"""Loss functions for object detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from scipy.optimize import linear_sum_assignment


class DetectionLoss(nn.Module):
    """Loss function for object detection with Hungarian matching."""
    
    def __init__(
        self,
        num_classes: int,
        bbox_loss_coef: float = 5.0,
        class_loss_coef: float = 1.0,
        giou_loss_coef: float = 2.0,
        eos_coef: float = 0.1  # Weight for no-object class
    ):
        """
        Initialize detection loss.
        
        Args:
            num_classes: Number of object classes
            bbox_loss_coef: Coefficient for bbox loss
            class_loss_coef: Coefficient for classification loss
            giou_loss_coef: Coefficient for GIoU loss
            eos_coef: Coefficient for no-object class
        """
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss_coef = bbox_loss_coef
        self.class_loss_coef = class_loss_coef
        self.giou_loss_coef = giou_loss_coef
        
        # Adjust weights for class imbalance
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions with 'pred_logits' and 'pred_boxes'
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        pred_logits = predictions['pred_logits']  # [B, num_queries, num_classes + 1]
        pred_boxes = predictions['pred_boxes']  # [B, num_queries, 4]
        
        batch_size = pred_logits.shape[0]
        
        # Perform Hungarian matching
        indices = self.match_predictions_to_targets(predictions, targets)
        
        # Compute classification loss
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=pred_logits.device
        )
        
        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )
        
        # Compute bbox losses
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Convert target boxes to [0, 1] range
        target_boxes = self._normalize_boxes(target_boxes, targets)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / max(target_classes_o.shape[0], 1)
        
        # Compute GIoU loss
        loss_giou = 1 - torch.diag(self._generalized_box_iou(
            self._box_cxcywh_to_xyxy(src_boxes),
            self._box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / max(target_classes_o.shape[0], 1)
        
        # Total loss
        losses = {
            'loss_ce': loss_ce * self.class_loss_coef,
            'loss_bbox': loss_bbox * self.bbox_loss_coef,
            'loss_giou': loss_giou * self.giou_loss_coef
        }
        losses['loss'] = sum(losses.values())
        
        return losses
    
    @torch.no_grad()
    def match_predictions_to_targets(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[tuple]:
        """Perform Hungarian matching between predictions and targets."""
        pred_logits = predictions['pred_logits']  # [B, num_queries, num_classes + 1]
        pred_boxes = predictions['pred_boxes']  # [B, num_queries, 4]
        
        batch_size, num_queries = pred_logits.shape[:2]
        
        # Flatten to compute cost matrices
        out_prob = pred_logits.flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes + 1]
        out_bbox = pred_boxes.flatten(0, 1)  # [B*num_queries, 4]
        
        indices = []
        
        for i, target in enumerate(targets):
            tgt_ids = target['labels']
            tgt_bbox = target['boxes']
            
            if len(tgt_ids) == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue
            
            # Normalize target boxes
            tgt_bbox_norm = self._normalize_boxes(tgt_bbox.unsqueeze(0), [target]).squeeze(0)
            
            # Classification cost
            cost_class = -out_prob[i * num_queries:(i + 1) * num_queries, tgt_ids]
            
            # L1 cost
            cost_bbox = torch.cdist(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox_norm, p=1)
            
            # GIoU cost
            cost_giou = -self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(out_bbox[i * num_queries:(i + 1) * num_queries]),
                self._box_cxcywh_to_xyxy(tgt_bbox_norm)
            )
            
            # Final cost matrix
            C = self.bbox_loss_coef * cost_bbox + self.class_loss_coef * cost_class + self.giou_loss_coef * cost_giou
            C = C.cpu()
            
            # Hungarian algorithm
            src_idx, tgt_idx = linear_sum_assignment(C)
            indices.append((torch.as_tensor(src_idx, dtype=torch.int64), torch.as_tensor(tgt_idx, dtype=torch.int64)))
        
        return indices
    
    def _get_src_permutation_idx(self, indices):
        """Get source permutation indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _normalize_boxes(self, boxes: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Normalize boxes to [0, 1] range assuming they're in pixel coordinates."""
        # For simplicity, assume boxes are already in [0, image_size] range
        # and normalize by image_size (this should be adapted based on your data)
        # Here we assume boxes are already in [cx, cy, w, h] format normalized
        return boxes
    
    def _box_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized IoU.
        
        Args:
            boxes1: [N, 4] in [x1, y1, x2, y2] format
            boxes2: [M, 4] in [x1, y1, x2, y2] format
            
        Returns:
            GIoU matrix [N, M]
        """
        # Compute intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        # Compute union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - inter
        
        iou = inter / union
        
        # Compute enclosing box
        lt_enclosing = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb_enclosing = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
        area_enclosing = wh_enclosing[:, :, 0] * wh_enclosing[:, :, 1]
        
        giou = iou - (area_enclosing - union) / area_enclosing
        
        return giou
