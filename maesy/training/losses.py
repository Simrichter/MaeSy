"""Loss functions for object detection."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from scipy.optimize import linear_sum_assignment
from abc import ABC, abstractmethod


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
    total_loss_aux: float

    def __init__(
            self,
            num_classes: int,
            bbox_loss_coef: float = 5.0,
            class_loss_coef: float = 1.0,
            giou_loss_coef: float = 2.0,
            eos_coef: float = 0.1,  # Weight for no-object class
            aux_loss_coef: float = 1.0,
            line_loss_coef: float = 2.0,
            dn_loss_coef: float = 1.0,
            enable_line_detection: bool = False,
            line_class_id: int = -1,
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
            :param aux_loss_coef: Coefficient for weighting of auxiliary loss
            :param line_loss_coef: Coefficient for loss of line-detection head
            :param dn_loss_coef: Coefficient for auxiliary denoising loss
            :param enable_line_detection: Wether line detection head is enabled
            :param line_class_id: The class id of the lines
            :param device: Device to run loss computation on
        """
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss_coef = bbox_loss_coef
        self.class_loss_coef = class_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.aux_loss_coef = aux_loss_coef
        self.line_loss_coef = line_loss_coef
        self.dn_loss_coef = dn_loss_coef
        self.enable_line_detection = enable_line_detection
        self.line_class_id = line_class_id

        self.device = device

        self.reset_metrics()

        # Adjust weights for class imbalance
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        # empty_weight = torch.softmax(empty_weight, dim=-1) # Normalize to sum to 1 (experimental??)
        empty_weight = empty_weight.to(self.device)
        self.register_buffer('empty_weight', empty_weight)

    def reset_metrics(self):
        self.total_loss = 0.0
        self.total_loss_ce = 0.0
        self.total_loss_bbox = 0.0
        self.total_loss_giou = 0.0
        self.total_loss_aux = 0.0
        self.total_loss_line = 0.0
        self.total_loss_dn = 0.0
        self.batch_count = 0

    def _compute_single_output_losses(
            self,
            pred_logits: torch.Tensor,
            pred_boxes: torch.Tensor,
            targets: List[Dict[str, torch.Tensor]],
            pred_lines: torch.Tensor | None = None,
            indices: List[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if indices is None:
            predictions = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
            if pred_lines is not None:
                predictions["pred_lines"] = pred_lines
            indices = self.match_predictions_to_targets(predictions, targets) # List[Tuple[Tensor, Tensor]]

        all_target_boxes = []
        all_target_lines = []
        all_target_labels = []
        all_is_line = []

        for t, (_, tgt_idx) in zip(targets, indices):
            num_boxes = len(t['boxes'])

            # Identify which matched targets are lines
            is_line = tgt_idx >= num_boxes
            is_box = ~is_line

            # BOX indices
            box_idx = tgt_idx[is_box]

            # LINE indices (shift back!)
            line_idx = tgt_idx[is_line] - num_boxes

            # Boxes
            if is_box.any():
                all_target_boxes.append(t['boxes'][box_idx])

            # Lines
            if is_line.any():
                all_target_lines.append(t['line_points'][line_idx])

            # Labels (already correct)
            all_target_labels.append(t['labels'][tgt_idx])

            # Mask
            all_is_line.append(is_line)

        target_classes_o = torch.cat(all_target_labels)
        line_mask = torch.cat(all_is_line)

        target_boxes = torch.cat(all_target_boxes) if all_target_boxes else torch.empty(0, 4, device=pred_logits.device)
        target_lines = torch.cat(all_target_lines) if all_target_lines else torch.empty(0, 4, device=pred_logits.device)

        # Compute classification loss
        # target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes, # No-Object class
            dtype=torch.int64,
            device=pred_logits.device
        )

        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o
        # Now, target_classes has shape [B, num_queries], where all non-used entries get class "None" and the matched entries get their correct class.

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            label_smoothing=0.08 # TODO: Make controllable through config
        )

        num_matched_targets = target_classes_o.shape[0]
        if num_matched_targets > 0:
            # Compute bbox losses
            src_boxes = pred_boxes[idx] # Selecting the boxes selected by the hungarian matching
            src_lines = pred_lines[idx] if pred_lines is not None else None
            # This selected the two chosen target boxes in correct order

            box_mask = ~line_mask
            num_box_matches = int(box_mask.sum().item())
            num_line_matches = int(line_mask.sum().item())

            if box_mask.any():
                src_boxes_for_loss = src_boxes[box_mask]
                target_boxes_for_loss = target_boxes
                assert src_boxes_for_loss.shape[0] == target_boxes_for_loss.shape[0]
                loss_bbox = F.l1_loss(src_boxes_for_loss, target_boxes_for_loss, reduction='none')
                if num_box_matches > 0:
                    loss_bbox = loss_bbox.sum() / num_box_matches

                giou_matrix = self._generalized_box_iou(
                    self._box_cxcywh_to_xyxy(src_boxes_for_loss),
                    self._box_cxcywh_to_xyxy(target_boxes_for_loss)
                )
                loss_giou = 1 - torch.diag(giou_matrix)
                if num_box_matches > 0:
                    loss_giou = loss_giou.sum() / num_box_matches
            else:
                loss_bbox = torch.tensor(0.0, device=pred_logits.device)
                loss_giou = torch.tensor(0.0, device=pred_logits.device)

            if line_mask.any() and src_lines is not None:
                src_lines_for_loss = src_lines[line_mask]
                tgt_lines_for_loss = target_lines
                assert src_lines_for_loss.shape[0] == tgt_lines_for_loss.shape[0]
                # loss_line = F.l1_loss(src_lines_for_loss, tgt_lines_for_loss, reduction='none')
                # loss_line = loss_line.sum() / max(num_line_matches, 1)

                # Accounting for order invariance of line keypoints
                pred = src_lines_for_loss
                gt = tgt_lines_for_loss

                gt_swapped = torch.cat([gt[:, 2:], gt[:, :2]], dim=-1)

                loss1 = torch.abs(pred - gt).sum(dim=-1)
                loss2 = torch.abs(pred - gt_swapped).sum(dim=-1)

                loss_line = torch.min(loss1, loss2)

                if num_line_matches > 0:
                    loss_line = loss_line.sum() / num_line_matches
                else:
                    loss_line = torch.tensor(0.0, device=pred_logits.device)
            else:
                loss_line = torch.tensor(0.0, device=pred_logits.device)
        else:
            loss_bbox = torch.tensor(0.0, device=pred_logits.device)
            loss_giou = torch.tensor(0.0, device=pred_logits.device)
            loss_line = torch.tensor(0.0, device=pred_logits.device)

        return {
            'loss_ce': loss_ce * self.class_loss_coef,
            'loss_bbox': loss_bbox * self.bbox_loss_coef,
            'loss_giou': loss_giou * self.giou_loss_coef,
            'loss_line': loss_line * self.line_loss_coef,
        }

    def _compute_denoising_loss(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dn_outputs = predictions.get("dn_outputs")
        if not isinstance(dn_outputs, dict):
            zero = torch.tensor(0.0, device=predictions['pred_logits'].device)
            return {"loss_dn": zero}

        pred_logits = dn_outputs["pred_logits"]
        pred_boxes = dn_outputs["pred_boxes"]
        target_labels = dn_outputs["target_labels"]
        target_boxes = dn_outputs["target_boxes"]
        valid_mask = dn_outputs["target_valid_mask"]

        # Denoising targets are provided in aligned query order, so no Hungarian matching is required here.
        if not valid_mask.any():
            return {"loss_dn": torch.tensor(0.0, device=pred_logits.device)}

        logits_valid = pred_logits[valid_mask]
        labels_valid = target_labels[valid_mask]
        boxes_valid = pred_boxes[valid_mask]
        target_boxes_valid = target_boxes[valid_mask]

        loss_ce = F.cross_entropy(logits_valid, labels_valid, weight=self.empty_weight)

        if self.enable_line_detection and self.line_class_id >= 0:
            line_mask = labels_valid == self.line_class_id
        else:
            line_mask = torch.zeros_like(labels_valid, dtype=torch.bool)
        box_mask = ~line_mask

        if box_mask.any():
            loss_bbox = F.l1_loss(boxes_valid[box_mask], target_boxes_valid[box_mask])
            giou_matrix = self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(boxes_valid[box_mask]),
                self._box_cxcywh_to_xyxy(target_boxes_valid[box_mask]),
            )
            loss_giou = (1 - torch.diag(giou_matrix)).mean()
        else:
            loss_bbox = torch.tensor(0.0, device=pred_logits.device)
            loss_giou = torch.tensor(0.0, device=pred_logits.device)

        if line_mask.any() and "pred_lines" in dn_outputs and "target_lines" in dn_outputs:
            pred_lines = dn_outputs["pred_lines"][valid_mask]
            target_lines = dn_outputs["target_lines"][valid_mask]
            loss_line = F.l1_loss(pred_lines[line_mask], target_lines[line_mask])
        else:
            loss_line = torch.tensor(0.0, device=pred_logits.device)

        combined = (
            loss_ce * self.class_loss_coef
            + loss_bbox * self.bbox_loss_coef
            + loss_giou * self.giou_loss_coef
            + loss_line * self.line_loss_coef
        ) * self.dn_loss_coef
        return {"loss_dn": combined}

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
        main_indices = self.match_predictions_to_targets(
            {
                'pred_logits': predictions['pred_logits'],
                'pred_boxes': predictions['pred_boxes'],
                'pred_lines': predictions.get('pred_lines'),
            },
            targets,
        )
        losses = self._compute_single_output_losses(
            pred_logits=predictions['pred_logits'],
            pred_boxes=predictions['pred_boxes'],
            targets=targets,
            pred_lines=predictions.get('pred_lines'),
            indices=main_indices,
        )

        aux_total = torch.tensor(0.0, device=predictions['pred_logits'].device)
        aux_outputs = predictions.get('aux_outputs', [])
        for aux_idx, aux_prediction in enumerate(aux_outputs):
            if aux_prediction['pred_logits'].shape[:2] != predictions['pred_logits'].shape[:2]:
                raise ValueError(
                    "Auxiliary outputs must use the same [batch, num_queries] shape as main predictions to "
                    "reuse Hungarian assignments."
                )
            aux_losses = self._compute_single_output_losses(
                pred_logits=aux_prediction['pred_logits'],
                pred_boxes=aux_prediction['pred_boxes'],
                targets=targets,
                pred_lines=aux_prediction.get('pred_lines'),
                indices=main_indices,
            )
            aux_component_total = sum(aux_losses.values()) * self.aux_loss_coef
            losses[f'loss_ce_aux_{aux_idx}'] = aux_losses['loss_ce'] * self.aux_loss_coef
            losses[f'loss_bbox_aux_{aux_idx}'] = aux_losses['loss_bbox'] * self.aux_loss_coef
            losses[f'loss_giou_aux_{aux_idx}'] = aux_losses['loss_giou'] * self.aux_loss_coef
            losses[f'loss_line_aux_{aux_idx}'] = aux_losses['loss_line'] * self.aux_loss_coef
            losses[f'loss_aux_{aux_idx}'] = aux_component_total
            aux_total = aux_total + aux_component_total

        dn_loss = self._compute_denoising_loss(predictions)
        losses.update(dn_loss)

        losses['loss_aux'] = aux_total
        losses['loss'] = losses['loss_ce'] + losses['loss_bbox'] + losses['loss_giou'] + losses['loss_line'] + aux_total + losses['loss_dn']

        # Log the sums of the losses per epoch
        self.total_loss += losses['loss'].item()
        self.total_loss_ce += losses['loss_ce'].item()
        self.total_loss_bbox += losses['loss_bbox'].item()
        self.total_loss_giou += losses['loss_giou'].item()
        self.total_loss_aux += losses['loss_aux'].item()
        self.total_loss_line += losses['loss_line'].item()
        self.total_loss_dn += losses['loss_dn'].item()
        self.batch_count += 1

        return losses

    def get_metrics(self) -> dict[str, float]:
        return {"total_loss": self.total_loss / self.batch_count,
                "total_loss_ce": self.total_loss_ce / self.batch_count,
                "total_loss_bbox": self.total_loss_bbox / self.batch_count,
                "total_loss_giou": self.total_loss_giou / self.batch_count,
                "total_loss_aux": self.total_loss_aux / self.batch_count,
                "total_loss_line": self.total_loss_line / self.batch_count,
                "total_loss_dn": self.total_loss_dn / self.batch_count}

    @torch.no_grad()
    def match_predictions_to_targets(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: List[Dict[str, torch.Tensor]]
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """Perform Hungarian matching between predictions and targets."""
        pred_logits = predictions['pred_logits']  # [B, num_queries, num_classes + 1]
        pred_boxes = predictions['pred_boxes']  # [B, num_queries, 4]
        pred_lines = predictions.get('pred_lines')

        batch_size, num_queries = pred_logits.shape[:2]
        # Flatten to compute cost matrices
        out_prob = pred_logits.flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes + 1]
        out_bbox = pred_boxes.flatten(0, 1)  # [B*num_queries, 4]
        indices = []


        for i, target in enumerate(targets):
            tgt_ids = target['labels'] # [num_target_boxes+num_target_lines,]
            tgt_bbox = target['boxes'] # [num_target_boxes, 4]
            tgt_lines = target.get('line_points') # [num_target_lines, 4]

            if len(tgt_ids) == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue

            # L1 cost lines
            cost_line = torch.zeros(num_queries, len(tgt_ids), device=out_bbox.device)
            if (
                self.enable_line_detection
                and self.line_class_id >= 0
                and pred_lines is not None
                and tgt_lines is not None
                # and len(tgt_lines) == len(tgt_ids)
            ):
                assert tgt_ids.shape[0] == len(tgt_bbox) + len(tgt_lines) # Check if assumption of labels including ALL classes is fulfilled
                pred_l = pred_lines[i]  # [num_queries, 4]
                tgt_l = tgt_lines  # [num_lines, 4]

                tgt_l_swapped = torch.cat([tgt_l[:, 2:], tgt_l[:, :2]], dim=-1)

                dist1 = torch.cdist(pred_l, tgt_l, p=1)
                dist2 = torch.cdist(pred_l, tgt_l_swapped, p=1)

                cost_line[:, len(tgt_bbox):] = torch.min(dist1, dist2)
                # cost_line[:, len(tgt_bbox):] = torch.cdist(pred_lines[i], tgt_lines, p=1)

            # L1 cost bboxes
            cost_bbox = torch.zeros(num_queries, len(tgt_ids), device=out_bbox.device)
            cost_bbox[:, :len(tgt_bbox)] = torch.cdist(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox, p=1)

            # Classification cost
            cost_class = -out_prob[i * num_queries:(i + 1) * num_queries][:, tgt_ids]

            # GIoU cost
            cost_giou = torch.zeros(num_queries, len(tgt_ids), device=out_bbox.device)
            cost_giou[:, :len(tgt_bbox)] = -self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(out_bbox[i * num_queries:(i + 1) * num_queries]),
                self._box_cxcywh_to_xyxy(tgt_bbox)
            )
            if cost_giou.shape[1] > cost_giou.shape[0]:
                raise ValueError(f"Hungarian matching failed, more target boxes than predictions. Cost GIoU shape: {cost_giou.shape}")

            # print(f"Cost matrices for batch item {i}:")
            # print(cost_class.mean())
            # print(cost_bbox.mean())
            # print(cost_giou.mean())

            # Final cost matrix
            C = self.bbox_loss_coef * cost_bbox + self.line_loss_coef * cost_line + self.class_loss_coef * cost_class + self.giou_loss_coef * cost_giou
            C = C.cpu().numpy()
            # Check for invalid entries in C
            if np.isnan(C).any() or np.isinf(C).any() or np.isnan(C).any():
                print(f"C contains NaN, Inf or NegInf:\n{C}")
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
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        union = area1[:, None] + area2 - inter

        iou = inter / (union+1e-7)

        # Compute enclosing box
        lt_enclosing = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb_enclosing = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
        area_enclosing = wh_enclosing[:, :, 0] * wh_enclosing[:, :, 1] + 1e-7

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
