"""Loss functions for object detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from scipy.optimize import linear_sum_assignment
from abc import ABC, abstractmethod
from ..model.components import Utils


class BaseLoss(nn.Module, ABC):
    """Abstract base class for loss functions."""
    batch_count: int
    total_loss: float

    @abstractmethod
    def forward(self, predictions: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[
        str, torch.Tensor]:
        # Calculate loss(es) and return them as dictionary. The main loss must have key 'loss'
        pass

    @abstractmethod
    def reset_metrics(self):
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        pass


class DetectionLoss(BaseLoss):
    """Loss function for object detection with Hungarian matching."""
    total_loss_ce: float
    total_loss_bbox: float
    total_loss_giou: float

    def __init__(
            self,
            num_classes: int,
            bbox_loss_coef: float = 5.0,
            class_loss_coef: float = 1.0,
            giou_loss_coef: float = 2.0,
            eos_coef: float = 0.1, # Weight for no-object class
            device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # TODO: Make this follow commandline-input device
    ):
        """
        Initialize detection loss.
        
        Args:
            :param num_classes: Number of object classes
            :param bbox_loss_coef: Coefficient for bbox loss
            :param class_loss_coef: Coefficient for classification loss
            :param giou_loss_coef: Coefficient for GIoU loss
            :param eos_coef: Coefficient for no-object class
            :param device: Device to run loss computation on
        """
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss_coef = bbox_loss_coef
        self.class_loss_coef = class_loss_coef
        self.giou_loss_coef = giou_loss_coef

        self.device = device

        self.reset_metrics()

        # Adjust weights for class imbalance
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        empty_weight = empty_weight.to(self.device)
        self.register_buffer('empty_weight', empty_weight)

    def reset_metrics(self):
        self.total_loss = 0.0
        self.total_loss_ce = 0.0
        self.total_loss_bbox = 0.0
        self.total_loss_giou = 0.0
        self.batch_count = 0

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
        # Now, target_classes has shape [B, num_queries], where all non-used entries get class "None" and the matched entries get their correct class.

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )

        # Compute bbox losses
        idx = self._get_src_permutation_idx(indices) # TODO: Recomputation unnecessary??
        src_boxes = pred_boxes[idx] # Selecting the boxes selected by the hungarian matching
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # Expecting the target boxes to be 0-1 normalized
        # This selected the two chosen target boxes in correct order

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / max(target_classes_o.shape[0], 1) # Averaging by number of target boxes

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

        # Log the sums of the losses per epoch
        self.total_loss += losses['loss'].item()
        self.total_loss_ce += losses['loss_ce'].item()
        self.total_loss_bbox += losses['loss_bbox'].item()
        self.total_loss_giou += losses['loss_giou'].item()
        self.batch_count += 1

        return losses

    def get_metrics(self) -> dict[str, float]:
        return {"total_loss": self.total_loss / self.batch_count,
                "total_loss_ce": self.total_loss_ce / self.batch_count,
                "total_loss_bbox": self.total_loss_bbox / self.batch_count,
                "total_loss_giou": self.total_loss_giou / self.batch_count}

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
            tgt_ids = target['labels'] # [B, num_target_boxes]
            tgt_bbox = target['boxes'] # [B, num_target_boxes, 4]

            if len(tgt_ids) == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue

            # Classification cost
            cost_class = -out_prob[i * num_queries:(i + 1) * num_queries, tgt_ids]

            # L1 cost
            cost_bbox = torch.cdist(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox, p=1)

            # GIoU cost
            cost_giou = -self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(out_bbox[i * num_queries:(i + 1) * num_queries]),
                self._box_cxcywh_to_xyxy(tgt_bbox)
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

# TODO: Move this util stuff to bounding_box.py
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


class MaskedMSE(BaseLoss):

    def reset_metrics(self) -> None:
        self.total_loss: float = 0.0
        self.batch_count: int = 0

    def get_metrics(self) -> dict[str, float]:
        return {"total_loss": self.total_loss / self.batch_count}

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        dist = (pred - target) ** 2
        md = dist.mean(dim=-1)  # [B, N], mean loss per patch
        tmd = (md if mask is None else (
                    md * mask)).sum() / mask.sum()  # mean loss (on unmasked patches only for more focussed training)
        self.total_loss += tmd.item()
        self.batch_count += 1
        return {'loss': tmd}
