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
        empty_weight = torch.softmax(empty_weight, dim=-1) # Normalize to sum to 1 (experimental??)
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
            weight=self.empty_weight
        )

        # Compute bbox losses
        idx = self._get_src_permutation_idx(indices) # TODO: Recomputation unnecessary??
        src_boxes = pred_boxes[idx] # Selecting the boxes selected by the hungarian matching
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # Expecting the target boxes to be 0-1 normalized
        # This selected the two chosen target boxes in correct order

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / max(target_classes_o.shape[0], 1) # Averaging by number of target boxes

        # Compute GIoU loss
        giou_matrix = self._generalized_box_iou(
            self._box_cxcywh_to_xyxy(src_boxes),
            self._box_cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = 1 - torch.diag(giou_matrix)
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
            # cost_class = -out_prob[i * num_queries:(i + 1) * num_queries, tgt_ids]
            cost_class = -out_prob[i*num_queries:(i+1)*num_queries][:, tgt_ids]

            # L1 cost
            cost_bbox = torch.cdist(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox, p=1)

            # GIoU cost
            cost_giou = -self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(out_bbox[i * num_queries:(i + 1) * num_queries]),
                # self._box_cxcywh_to_xyxy(tgt_bbox),
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

class ClassificationLoss(BaseLoss):

    def reset_metrics(self) -> None:
        self.total_loss: float = 0.0
        self.batch_count: int = 0

    def get_metrics(self) -> dict[str, float]:
        return {"total_loss": self.total_loss / self.batch_count}

    def forward(self, pred_logits: torch.Tensor, target_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute cross-entropy loss for classification.
        Args:
            :param pred_logits: [B, num_classes] raw logits from the model
            :param target_labels: [B] ground truth class indices
        Returns:
            Dictionary with key 'loss' containing the computed loss tensor
        """
        ce_loss = F.cross_entropy(pred_logits, target_labels)
        self.total_loss += ce_loss.item()
        self.batch_count += 1
        return {'loss': ce_loss}

class YOLOv8Loss(BaseLoss):
    """YOLOv8 detection loss with focal loss and IoU-based bbox loss."""
    total_loss_cls: float
    total_loss_bbox: float
    total_loss_dfl: float

    def __init__(
            self,
            num_classes: int,
            bbox_loss_coef: float = 7.5,
            cls_loss_coef: float = 0.5,
            dfl_loss_coef: float = 1.5,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        """
        Initialize YOLOv8 detection loss.

        Args:
            num_classes: Number of object classes
            bbox_loss_coef: Coefficient for bbox loss
            cls_loss_coef: Coefficient for classification loss
            dfl_loss_coef: Coefficient for DFL (Distribution Focal Loss) loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            device: Device to run loss computation on
        """
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss_coef = bbox_loss_coef
        self.cls_loss_coef = cls_loss_coef
        self.dfl_loss_coef = dfl_loss_coef
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.device = device

        self.reset_metrics()

    def reset_metrics(self):
        self.total_loss = 0.0
        self.total_loss_cls = 0.0
        self.total_loss_bbox = 0.0
        self.total_loss_dfl = 0.0
        self.batch_count = 0

    def forward(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute YOLOv8 loss.

        Args:
            predictions: Model predictions with 'pred_logits' and 'pred_boxes'
            targets: Ground truth targets with 'labels' and 'boxes'

        Returns:
            Dictionary of losses
        """
        pred_logits = predictions['pred_logits']  # [B, num_queries, num_classes]
        pred_boxes = predictions['pred_boxes']  # [B, num_queries, 4]

        # Flatten batch dimension
        batch_size = pred_logits.shape[0]

        # Process targets
        target_classes_list = []
        target_boxes_list = []

        for target in targets:
            target_classes_list.append(target['labels'])
            target_boxes_list.append(target['boxes'])

        # Compute classification loss (focal loss)
        loss_cls = self._compute_focal_loss(pred_logits, target_classes_list)

        # Compute bounding box loss
        loss_bbox, loss_dfl = self._compute_bbox_loss(pred_boxes, target_boxes_list)

        # Total loss
        losses = {
            'loss_cls': loss_cls * self.cls_loss_coef,
            'loss_bbox': loss_bbox * self.bbox_loss_coef,
            'loss_dfl': loss_dfl * self.dfl_loss_coef
        }
        losses['loss'] = sum(losses.values())

        # Log metrics
        self.total_loss += losses['loss'].item()
        self.total_loss_cls += losses['loss_cls'].item()
        self.total_loss_bbox += losses['loss_bbox'].item()
        self.total_loss_dfl += losses['loss_dfl'].item()
        self.batch_count += 1

        return losses

    def get_metrics(self) -> dict[str, float]:
        return {
            "total_loss": self.total_loss / self.batch_count,
            "total_loss_cls": self.total_loss_cls / self.batch_count,
            "total_loss_bbox": self.total_loss_bbox / self.batch_count,
            "total_loss_dfl": self.total_loss_dfl / self.batch_count
        }

    def _compute_focal_loss(self, pred_logits: torch.Tensor, target_classes_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute focal loss for classification.

        Args:
            pred_logits: [B, num_queries, num_classes]
            target_classes_list: List of target class tensors

        Returns:
            Focal loss
        """
        batch_size, num_queries, num_classes = pred_logits.shape

        # Create target tensor filled with zeros (background class)
        target_classes = torch.zeros(batch_size, num_queries, dtype=torch.long, device=pred_logits.device)

        # Fill in positive targets
        for b_idx, target_cls in enumerate(target_classes_list):
            if len(target_cls) > 0:
                target_classes[b_idx, :len(target_cls)] = target_cls

        # Compute probabilities
        p = torch.sigmoid(pred_logits)  # [B, num_queries, num_classes]

        # Focal loss computation
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, F.one_hot(target_classes, num_classes).float(), reduction='none')
        focal_weight = (1 - p) ** self.focal_gamma
        focal_loss = self.focal_alpha * focal_weight * ce_loss

        return focal_loss.mean()

    def _compute_bbox_loss(self, pred_boxes: torch.Tensor, target_boxes_list: List[torch.Tensor]) -> tuple:
        """
        Compute bounding box loss (IoU + DFL).

        Args:
            pred_boxes: [B, num_queries, 4] in any format
            target_boxes_list: List of target box tensors

        Returns:
            Tuple of (bbox_loss, dfl_loss)
        """
        batch_size, num_queries, _ = pred_boxes.shape

        # Create padded target boxes
        max_targets = max(len(t) for t in target_boxes_list) if target_boxes_list else 0
        target_boxes_padded = torch.zeros(batch_size, num_queries, 4, device=pred_boxes.device)

        for b_idx, target_box in enumerate(target_boxes_list):
            if len(target_box) > 0:
                target_boxes_padded[b_idx, :len(target_box)] = target_box

        # Compute IoU loss
        pred_xyxy = self._box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = self._box_cxcywh_to_xyxy(target_boxes_padded)

        iou = self._compute_iou(pred_xyxy, target_xyxy)  # [B, num_queries, num_queries]

        # Use diagonal for matched pairs (simplified - in practice would use assignment)
        iou_loss = 1.0 - torch.diagonal(iou, dim1=1, dim2=2).mean()

        # DFL loss (simplified - distribution focal loss for localization)
        # In full YOLOv8, this involves regression to a distribution over discrete positions
        # Here we simplify to L1 loss on the boxes
        dfl_loss = F.l1_loss(pred_boxes, target_boxes_padded)

        return iou_loss, dfl_loss

    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU matrix between two sets of boxes.

        Args:
            boxes1: [B, N, 4] in [x1, y1, x2, y2] format
            boxes2: [B, M, 4] in [x1, y1, x2, y2] format

        Returns:
            IoU matrix [B, N, M]
        """
        # Compute intersection
        lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])  # [B, N, M, 2]
        rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])  # [B, N, M, 2]

        wh = (rb - lt).clamp(min=0)  # [B, N, M, 2]
        inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [B, N, M]

        # Compute union
        area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
        area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
        union = area1[:, :, None] + area2[:, None, :] - inter

        iou = inter / (union + 1e-7)

        return iou

    def _box_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
