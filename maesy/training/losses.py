"""Loss functions for object detection."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
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
    total_loss_line: float
    total_loss_ellipse: float
    total_loss_aux: float
    total_loss_dn: float
    total_loss_enc: float

    def __init__(
            self,
            num_classes: int,
            bbox_loss_coef: float = 5.0,
            class_loss_coef: float = 1.0,
            giou_loss_coef: float = 2.0,
            eos_coef: float = 0.1,  # Weight for no-object class
            label_smoothing: float = 0.0,
            aux_loss_coef: float = 0.5,
            enc_loss_coef: float = 0.3,
            line_loss_coef: float = 2.0,
            line_angle_loss_coef: float = 0.5,
            line_length_loss_coef: float = 0.5,
            ellipse_loss_coef: float = 2.0,
            ellipse_shape_coef: float = 1.0,
            dn_loss_coef: float = 1.0,
            enable_line_detection: bool = False,
            line_class_id: int = -1,
            enable_ellipse_detection: bool = False,
            ellipse_class_id: int = -1,
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
            :param label_smoothing: Value for label smoothing in cross-entropy loss
            :param aux_loss_coef: Coefficient for weighting of auxiliary loss
            :param line_loss_coef: Coefficient for loss of line-detection head
            :param line_angle_loss_coef: Coefficient for line angle loss component
            :param line_length_loss_coef: Coefficient for line log-length loss component
            :param ellipse_loss_coef: Coefficient for loss of ellipse-detection head
            :param ellipse_shape_coef: Coefficient to downscale frobenius norm (shape-loss) of ellipses
            :param dn_loss_coef: Coefficient for auxiliary denoising loss
            :param enable_line_detection: Whether line detection head is enabled
            :param line_class_id: The class id of the lines
            :param enable_ellipse_detection: Whether ellipse detection head is enabled
            :param ellipse_class_id: The class id of the ellipses
            :param device: Device to run loss computation on (except from hungarian matching, wich must be CPU)
        """
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss_coef = bbox_loss_coef
        self.class_loss_coef = class_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.aux_loss_coef = aux_loss_coef
        self.enc_loss_coef = enc_loss_coef
        self.line_loss_coef = line_loss_coef
        self.line_angle_loss_coef = line_angle_loss_coef
        self.line_length_loss_coef = line_length_loss_coef
        self.ellipse_loss_coef = ellipse_loss_coef
        self.ellipse_shape_coef = ellipse_shape_coef
        self.dn_loss_coef = dn_loss_coef
        self.enable_line_detection = enable_line_detection
        self.line_class_id = line_class_id
        self.enable_ellipse_detection = enable_ellipse_detection
        self.ellipse_class_id = ellipse_class_id

        self.device = device

        self.reset_metrics()
        # Adjust weights for class imbalance
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        # empty_weight = torch.softmax(empty_weight, dim=-1) # Normalize to sum to 1 (experimental??)
        empty_weight = empty_weight.to(self.device)
        self.register_buffer('empty_weight', empty_weight)

        self.label_smoothing = label_smoothing

    def reset_metrics(self):
        """
            Resets all internally accumulated metrics
        """
        self.total_loss = 0.0
        self.total_loss_ce = 0.0
        self.total_loss_bbox = 0.0
        self.total_loss_giou = 0.0
        self.total_loss_aux = 0.0
        self.total_loss_line = 0.0
        self.total_loss_ellipse = 0.0
        self.total_loss_dn = 0.0
        self.total_loss_enc = 0.0
        self.batch_count = 0

    def _compute_single_output_losses(
            self,
            pred_logits: torch.Tensor,
            pred_boxes: torch.Tensor,
            targets: List[Dict[str, torch.Tensor]],
            pred_lines: Optional[torch.Tensor] = None,
            pred_ellipses: Optional[torch.Tensor] = None,
            indices: Optional[List[tuple[torch.Tensor, torch.Tensor]]]= None,
    ) -> Dict[str, torch.Tensor]:
        if indices is None:
            predictions = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
            if pred_lines is not None:
                predictions["pred_lines"] = pred_lines
            if pred_ellipses is not None:
                predictions["pred_ellipses"] = pred_ellipses
            with torch.autocast(device_type=self.device.type, enabled=False):
                indices = self.match_predictions_to_targets(predictions, targets) # List[Tuple[Tensor, Tensor]]

        all_target_boxes = []
        all_target_lines = []
        all_target_ellipses = []
        all_target_labels = []
        all_is_line = []
        all_is_ellipse = []

        for t, (_, tgt_idx) in zip(targets, indices):
            num_boxes = len(t['boxes'])
            num_lines = len(t['line_points']) if 'line_points' in t else 0
            num_ellipses = len(t['ellipses']) if 'ellipses' in t else 0

            assert len(tgt_idx) == num_boxes+num_lines+num_ellipses, "Length of matched target indices must equal total number of targets (boxes + lines + ellipses) for this target"

            # Identify which matched targets are lines
            is_line = (tgt_idx >= num_boxes) & (tgt_idx < num_boxes + num_lines)
            is_ellipse = tgt_idx >= (num_boxes + num_lines)
            is_box = ~(is_line | is_ellipse)

            # BOX indices
            box_idx = tgt_idx[is_box]

            # LINE indices (shifted back!)
            line_idx = tgt_idx[is_line] - num_boxes

            # ELLIPSE indices (shifted back!)
            ellipse_idx = tgt_idx[is_ellipse] - (num_boxes + num_lines)

            # Boxes
            if is_box.any():
                all_target_boxes.append(t['boxes'][box_idx])

            # Lines
            if is_line.any():
                all_target_lines.append(t['line_points'][line_idx])

            if is_ellipse.any():
                all_target_ellipses.append(t['ellipses'][ellipse_idx])

            # Labels (already correct)
            all_target_labels.append(t['labels'][tgt_idx])

            # Mask
            all_is_line.append(is_line)
            all_is_ellipse.append(is_ellipse)

        target_classes_o = torch.cat(all_target_labels)
        line_mask = torch.cat(all_is_line)
        ellipse_mask = torch.cat(all_is_ellipse)

        target_boxes = torch.cat(all_target_boxes) if all_target_boxes else torch.empty(0, 4, device=pred_logits.device)
        target_lines = torch.cat(all_target_lines) if all_target_lines else torch.empty(0, 4, device=pred_logits.device)
        target_ellipses = torch.cat(all_target_ellipses) if all_target_ellipses else torch.empty(0, 6, device=pred_logits.device)
        # Compute classification loss
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
            label_smoothing=self.label_smoothing
        )

        num_matched_targets = target_classes_o.shape[0]
        if num_matched_targets > 0:
            # Compute bbox losses
            src_boxes = pred_boxes[idx] # Selecting the boxes selected by the hungarian matching
            src_lines = pred_lines[idx] if pred_lines is not None else None
            src_ellipses = pred_ellipses[idx] if pred_ellipses is not None else None
            # This selected the two chosen target boxes in correct order)
            box_mask = ~ (line_mask | ellipse_mask)
            num_box_matches = int(box_mask.sum().item())
            num_line_matches = int(line_mask.sum().item())
            num_ellipse_matches = int(ellipse_mask.sum().item())

            if box_mask.any():
                src_boxes_for_loss = src_boxes[box_mask]
                target_boxes_for_loss = target_boxes
                assert src_boxes_for_loss.shape[0] == target_boxes_for_loss.shape[0]
                loss_bbox = F.l1_loss(src_boxes_for_loss, target_boxes_for_loss, reduction='none')
                if num_box_matches > 0:
                    loss_bbox = loss_bbox.sum() / num_box_matches

                giou_matrix = self._generalized_box_iou(src_boxes_for_loss, target_boxes_for_loss)
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

                endpoint_loss = torch.min(loss1, loss2)

                pred_vec = pred[:, 2:] - pred[:, :2]
                gt_vec = gt[:, 2:] - gt[:, :2]
                pred_len = torch.norm(pred_vec, dim=-1)
                gt_len = torch.norm(gt_vec, dim=-1)
                eps = 1e-6
                pred_unit = pred_vec / (pred_len.unsqueeze(-1) + eps)
                gt_unit = gt_vec / (gt_len.unsqueeze(-1) + eps)
                angle_alignment = torch.sum(pred_unit * gt_unit, dim=-1).abs().clamp(max=1.0)
                angle_loss = 1.0 - angle_alignment
                log_length_loss = torch.abs(torch.log(pred_len + eps) - torch.log(gt_len + eps))

                loss_line = endpoint_loss + self.line_angle_loss_coef * angle_loss + self.line_length_loss_coef * log_length_loss

                if num_line_matches > 0:
                    loss_line = loss_line.sum() / num_line_matches
                else:
                    loss_line = torch.tensor(0.0, device=pred_logits.device)
            else:
                loss_line = torch.tensor(0.0, device=pred_logits.device)

            if ellipse_mask.any() and src_ellipses is not None:
                pred = src_ellipses[ellipse_mask]
                gt = target_ellipses
                assert pred.shape[0] == gt.shape[0]


                center_loss = torch.abs(pred[:, :2] - gt[:, :2]).sum(dim=-1)
                # shape_loss = self.frobenius_per_sample(pred[:, 2:], gt[:, 2:])
                shape_loss = torch.abs(pred[:, 2:4] - gt[:, 2:4]).sum(dim=-1)
                rotation_loss = (pred[:, 4]-gt[:, 4])**2 + (pred[:, 5]-gt[:, 5])**2
                loss_ellipse = center_loss + self.ellipse_shape_coef * (shape_loss + rotation_loss)

                if num_ellipse_matches > 0:
                    loss_ellipse = loss_ellipse.sum() / num_ellipse_matches
                else:
                    loss_ellipse = torch.tensor(0.0, device=pred_logits.device)
            else:
                loss_ellipse = torch.tensor(0.0, device=pred_logits.device)

        else:
            loss_bbox = torch.tensor(0.0, device=pred_logits.device)
            loss_giou = torch.tensor(0.0, device=pred_logits.device)
            loss_line = torch.tensor(0.0, device=pred_logits.device)
            loss_ellipse = torch.tensor(0.0, device=pred_logits.device)

        return {
            'loss_ce': loss_ce * self.class_loss_coef,
            'loss_bbox': loss_bbox * self.bbox_loss_coef,
            'loss_giou': loss_giou * self.giou_loss_coef,
            'loss_line': loss_line * self.line_loss_coef,
            'loss_ellipse': loss_ellipse * self.ellipse_loss_coef,
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

        if self.enable_line_detection and self.ellipse_class_id >= 0:
            ellipse_class_mask = labels_valid == self.ellipse_class_id
        else:
            ellipse_class_mask = torch.zeros_like(labels_valid, dtype=torch.bool)

        # box_mask = ~line_mask
        box_mask = ~(line_mask | ellipse_class_mask)

        if box_mask.any():
            loss_bbox = F.l1_loss(boxes_valid[box_mask], target_boxes_valid[box_mask])
            giou_matrix = self._generalized_box_iou(boxes_valid[box_mask],                target_boxes_valid[box_mask],
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

    def _compute_encoder_dense_loss(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        enc_outputs = predictions.get("enc_outputs")
        zero = torch.tensor(0.0, device=predictions['pred_logits'].device)
        if not isinstance(enc_outputs, dict):
            return {
                "loss_ce_enc": zero,
                "loss_bbox_enc": zero,
                "loss_giou_enc": zero,
                "loss_line_enc": zero,
                "loss_ellipse_enc": zero,
                "loss_enc": zero,
            }

        pred_logits = enc_outputs.get("pred_logits")
        pred_boxes = enc_outputs.get("pred_boxes")
        if pred_logits is None or pred_boxes is None:
            return {
                "loss_ce_enc": zero,
                "loss_bbox_enc": zero,
                "loss_giou_enc": zero,
                "loss_line_enc": zero,
                "loss_ellipse_enc": zero,
                "loss_enc": zero,
            }

        with torch.autocast(device_type=self.device.type, enabled=False):
            enc_indices = self.match_predictions_to_targets(
                enc_outputs,
                targets
            )
        enc_losses = self._compute_single_output_losses(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            targets=targets,
            pred_lines=enc_outputs.get("pred_lines"),
            pred_ellipses=enc_outputs.get("pred_ellipses"),
            indices=enc_indices,
        )

        loss_ce_enc = enc_losses['loss_ce'] * self.enc_loss_coef
        loss_bbox_enc = enc_losses['loss_bbox'] * self.enc_loss_coef
        loss_giou_enc = enc_losses['loss_giou'] * self.enc_loss_coef
        loss_line_enc = enc_losses['loss_line'] * self.enc_loss_coef
        loss_ellipse_enc = enc_losses['loss_ellipse'] * self.enc_loss_coef
        return {
            "loss_ce_enc": loss_ce_enc,
            "loss_bbox_enc": loss_bbox_enc,
            "loss_giou_enc": loss_giou_enc,
            "loss_line_enc": loss_line_enc,
            "loss_ellipse_enc": loss_ellipse_enc,
            "loss_enc": loss_ce_enc + loss_bbox_enc + loss_giou_enc + loss_line_enc + loss_ellipse_enc,
        }

    def forward(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions with 'pred_logits', 'pred_boxes' and optionally pred_lines and pred_ellipses
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            main_indices = self.match_predictions_to_targets(
                predictions,
                targets,
            )
        losses = self._compute_single_output_losses(
            pred_logits=predictions['pred_logits'],
            pred_boxes=predictions['pred_boxes'],
            targets=targets,
            pred_lines=predictions.get('pred_lines'),
            pred_ellipses=predictions.get('pred_ellipses'),
            indices=main_indices,
        )

        aux_total = torch.tensor(0.0, device=predictions['pred_logits'].device)
        aux_outputs = predictions.get('aux_outputs', [])
        for aux_idx, aux_prediction in enumerate(aux_outputs):
            if aux_prediction['pred_logits'].shape[:2] != predictions['pred_logits'].shape[:2]:
                raise ValueError(
                    "Auxiliary outputs must use the same [batch, num_queries] shape as main predictions to reuse Hungarian assignments."
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

        enc_loss = self._compute_encoder_dense_loss(predictions, targets)
        losses.update(enc_loss)

        losses['loss_aux'] = aux_total
        losses['loss'] = (
            losses['loss_ce'] + losses['loss_bbox'] + losses['loss_giou'] + losses['loss_line'] + losses['loss_ellipse']
            + aux_total + losses['loss_dn'] + losses['loss_enc']
        )

        # Log the sums of the losses per epoch
        self.total_loss += losses['loss'].item()
        self.total_loss_ce += losses['loss_ce'].item()
        self.total_loss_bbox += losses['loss_bbox'].item()
        self.total_loss_giou += losses['loss_giou'].item()
        self.total_loss_aux += losses['loss_aux'].item()
        self.total_loss_line += losses['loss_line'].item()
        self.total_loss_ellipse += losses['loss_ellipse'].item()
        self.total_loss_dn += losses['loss_dn'].item()
        self.total_loss_enc += losses['loss_enc'].item()
        self.batch_count += 1

        return losses

    def get_metrics(self) -> dict[str, float]:
        return {"total_loss": self.total_loss / self.batch_count,
                "total_loss_ce": self.total_loss_ce / self.batch_count,
                "total_loss_bbox": self.total_loss_bbox / self.batch_count,
                "total_loss_giou": self.total_loss_giou / self.batch_count,
                "total_loss_aux": self.total_loss_aux / self.batch_count,
                "total_loss_line": self.total_loss_line / self.batch_count,
                "total_loss_ellipse": self.total_loss_ellipse / self.batch_count,
                "total_loss_dn": self.total_loss_dn / self.batch_count,
                "total_loss_enc": self.total_loss_enc / self.batch_count}

    @staticmethod
    def build_A_flat(ellipse_logits: torch.Tensor) -> torch.Tensor:
        """
            Calculates cholesky decomposition matrix L from the raw ellipse prediction outputs and reconstructs the A matrix.
            Since A is symmetric, only the values of the upper triangular matrix are returned.
            Softplus is applied to L_11 and L_22 logits to ensure positivity.

            Args:
                 :param ellipse_logits: Tensor containing the raw parameterization of the ellipses. Shape: [N, 5]
        """
        # e: [N, 5]
        a, b, c = ellipse_logits[:, 2], ellipse_logits[:, 3], ellipse_logits[:, 4]

        l11 = F.softplus(a) + 1e-6
        l21 = b
        l22 = F.softplus(c) + 1e-6

        # construct A explicitly (2x2)
        A11 = l11 ** 2
        A12 = l11 * l21
        A22 = l21 ** 2 + l22 ** 2

        # symmetric → store as 3-vector instead of 4
        # [A11, A12, A22]
        A_flat = torch.stack([A11, A12, A22], dim=-1)  # [N, 3]

        return A_flat

    @staticmethod
    def frobenius_per_sample(Ap: torch.Tensor, At: torch.Tensor) -> torch.Tensor:
        """
            Computes the Frobenius distance between two matrices of shape [K, 3]
            The tensors contain the upper triangular part of the A matrices of a cholesky decomposition for K predictions/targets.
            Ap, At: [K, 3] = [A11, A12, A22]

            Args:
                :param Ap: Predicted A matrices of shape [K, 3]
                :param At: Target A matrices of shape [K, 3]

            Returns:
                Tensor of shape [K,] containing the Frobenius distance between each pair of predicted and target A matrices.
        """
        w = 2 ** 0.5
        Ap = torch.stack([Ap[:, 0], w * Ap[:, 1], Ap[:, 2]], dim=-1)
        At = torch.stack([At[:, 0], w * At[:, 1], At[:, 2]], dim=-1)

        diff = Ap - At
        return torch.norm(diff, dim=-1)  # [K]

    @staticmethod
    def _pairwise_frobenius(Ap, At) -> torch.Tensor:
        """
            Compute pairwise Frobenius distances between two sets of matrices.
            Takes the A matrices of ellipses only, no center coordinates.
            A is represented only by A_11, A_12, A_22, so the input matrices have shape [N, 3] and [M, 3] respectively.

            Args:
                :param Ap: Predicted A matrices of shape [N, 3]
                :param At: Target A matrices of shape [M, 3]

            Returns:
                Pairwise Frobenius distances [N, M]
        """
        w = 2 ** 0.5 # sqrt(2) scaling factor for correct calculation
        Ap_scaled = torch.stack([Ap[:, 0], w * Ap[:, 1], Ap[:, 2]], dim=-1)
        At_scaled = torch.stack([At[:, 0], w * At[:, 1], At[:, 2]], dim=-1)
        # squared norms
        Ap_norm = (Ap_scaled**2).sum(dim=1, keepdim=True)      # [N, 1]
        At_norm = (At_scaled**2).sum(dim=1).unsqueeze(0)       # [1, M]
        # inner product
        prod = Ap @ At.t()                              # [N, M]
        # squared distance
        dist2 = Ap_norm + At_norm - 2 * prod             # [N, M]
        # numerical stability
        dist2 = torch.clamp(dist2, min=0.0)
        return torch.sqrt(dist2 + 1e-8)

    @torch.no_grad()
    def match_predictions_to_targets(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: List[Dict[str, torch.Tensor]]
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
            Perform Hungarian matching between predictions and targets.
            Expects boxes in format (x1 y1 x2 y2), lines in format (x y x y) and ellipses in format (cx cy log_a log_b sin2theta cos2theta)
        """
        pred_logits = predictions['pred_logits'].float()  # [B, num_queries, num_classes + 1]
        pred_boxes = predictions['pred_boxes'].float()  # [B, num_queries, 4]
        pred_lines = predictions.get('pred_lines')
        if pred_lines is not None:
            pred_lines = pred_lines.float()
        pred_ellipses = predictions.get('pred_ellipses')
        if pred_ellipses is not None:
            pred_ellipses = pred_ellipses.float()

        assert torch.isfinite(pred_boxes).all()
        assert torch.isfinite(pred_logits).all()
        assert pred_lines is None or torch.isfinite(pred_lines).all()
        assert pred_ellipses is None or torch.isfinite(pred_ellipses).all()

        batch_size, num_queries = pred_logits.shape[:2]
        # Flatten to compute cost matrices
        out_prob = pred_logits.float().flatten(0, 1).clamp(-20, 20).log_softmax(-1)  # [B*num_queries, num_classes + 1]
        out_bbox = pred_boxes.flatten(0, 1)  # [B*num_queries, 4]
        indices = []


        for i, target in enumerate(targets):
            tgt_ids = target['labels'] # [num_target_boxes+num_target_lines+num_ellipses,]
            tgt_bbox = target['boxes'] # [num_target_boxes, 4]
            tgt_lines = target.get('line_points') # [num_target_lines, 4]
            tgt_ellipses = target.get('ellipses') # [num_ellipses, 5]

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
                # assert tgt_ids.shape[0] == len(tgt_bbox) + len(tgt_lines) + len(tgt_ellipses) # Check if assumption of labels including ALL classes is fulfilled
                pred_l = pred_lines[i]  # [num_queries, 4]
                tgt_l = tgt_lines  # [num_lines, 4]

                tgt_l_swapped = torch.cat([tgt_l[:, 2:], tgt_l[:, :2]], dim=-1)

                dist1 = torch.cdist(pred_l, tgt_l, p=1)
                dist2 = torch.cdist(pred_l, tgt_l_swapped, p=1)

                pred_vec = pred_l[:, 2:] - pred_l[:, :2]
                tgt_vec = tgt_l[:, 2:] - tgt_l[:, :2]
                pred_len = torch.linalg.vector_norm(pred_vec, dim=-1)
                tgt_len = torch.linalg.vector_norm(tgt_vec, dim=-1)
                eps = 1e-6
                pred_unit = pred_vec / (pred_len.unsqueeze(-1) + eps)
                tgt_unit = tgt_vec / (tgt_len.unsqueeze(-1) + eps)
                angle_alignment = (pred_unit[:, None, :] * tgt_unit[None, :, :]).sum(dim=-1).abs().clamp(max=1.0)
                angle_cost = 1.0 - angle_alignment
                weight = torch.clamp(tgt_len/0.02, max=1.0) # weighting down lines that are shorter than 2% of the image size (~4,5 pixels at 224)
                angle_cost *= weight[None, :]
                log_len_pred = torch.log(pred_len + eps)
                log_len_tgt = torch.log(tgt_len + eps)
                log_len_cost = torch.abs(log_len_pred[:, None] - log_len_tgt[None, :])

                len_tgt_boxes = len(tgt_bbox)
                len_tgt_lines = len(tgt_lines)
                cost_line[:, len_tgt_boxes:len_tgt_boxes + len_tgt_lines] = (
                    torch.min(dist1, dist2)
                    + self.line_angle_loss_coef * angle_cost
                    + self.line_length_loss_coef * log_len_cost
                )
                # cost_line[:, len(tgt_bbox):] = torch.cdist(pred_lines[i], tgt_lines, p=1)

            cost_ellipse = torch.zeros(num_queries, len(tgt_ids), device=out_bbox.device)
            if (
                self.enable_ellipse_detection
                and self.ellipse_class_id >= 0
                and pred_ellipses is not None
                and tgt_ellipses is not None
            ):
                pred_e = pred_ellipses[i]  # [num_queries, 6]
                tgt_e = tgt_ellipses  # [num_ellipses, 6]
                len_tgt_lines = 0 if tgt_lines is None else len(tgt_lines)

                # ellipse cost mirrors the per-sample loss definition
                center_loss = torch.cdist(pred_e[:, :2], tgt_e[:, :2], p=1)
                shape_loss = torch.cdist(pred_e[:, 2:4], tgt_e[:, 2:4], p=1)
                rotation_loss = torch.cdist(pred_e[:, 4:6], tgt_e[:, 4:6], p=2) ** 2
                cost_ellipse[:, len(tgt_bbox)+len_tgt_lines:] = center_loss + self.ellipse_shape_coef * (shape_loss + rotation_loss)
                # cost_ellipse[:, len(tgt_bbox)+len_tgt_lines:] = torch.cdist(pred_e[:, :2], tgt_e[:, :2], p=1) + self.ellipse_shape_coef * self._pairwise_frobenius(pred_e[:, 2:], tgt_e[:, 2:])

            # L1 cost bboxes
            cost_bbox = torch.zeros(num_queries, len(tgt_ids), device=out_bbox.device)
            cost_bbox[:, :len(tgt_bbox)] = torch.cdist(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox, p=1)

            # Classification cost
            cost_class = -out_prob[i * num_queries:(i + 1) * num_queries][:, tgt_ids]

            # GIoU cost
            cost_giou = torch.zeros(num_queries, len(tgt_ids), device=out_bbox.device)
            cost_giou[:, :len(tgt_bbox)] = -self._generalized_box_iou(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox)
            if cost_giou.shape[1] > cost_giou.shape[0]:
                raise ValueError(f"Hungarian matching failed, more target boxes than predictions. Cost GIoU shape: {cost_giou.shape}")

            def _check(name, x):
                if not torch.isfinite(x).all():
                    print(f"\n🚨 {name} has NaNs/Infs")
                    print("min:", x.min().item())
                    print("max:", x.max().item())
                    raise RuntimeError(name)

            _check("cost_class", cost_class)
            _check("cost_bbox", cost_bbox)
            _check("cost_giou", cost_giou)

            # Final cost matrix
            C = self.bbox_loss_coef * cost_bbox + self.line_loss_coef * cost_line + self.ellipse_loss_coef * cost_ellipse + self.class_loss_coef * cost_class + self.giou_loss_coef * cost_giou
            C = C.cpu().numpy()
            # Check for invalid entries in C
            if np.isnan(C).any() or np.isinf(C).any() or np.isnan(C).any():
                print(f"C contains NaN, Inf or NegInf:\n{C}")
            # Hungarian algorithm
            src_idx, tgt_idx = linear_sum_assignment(C)
            indices.append((torch.as_tensor(src_idx, dtype=torch.int64), torch.as_tensor(tgt_idx, dtype=torch.int64)))

        return indices

    @staticmethod
    def _get_src_permutation_idx(indices):
        """Get source permutation indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    @staticmethod
    def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
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

    # def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    #     """
    #     Loss calculation. Calculates Pixel-level MSE with masking on patch level
    #     !Expects pred and target in patchified format!
    #
    #     Args:
    #         :param pred: prediction of the model in patchified shape [B, np, c*ps**2] (np = num_patches, ps = patch_size)
    #         :param target: correct target in patchified shape [B, np, c*ps**2] (np = num_patches, ps = patch_size)
    #         :param mask: Patch-level mask with '1' for every patch that should be considered in loss and '0' otherwise, shape [B, np]
    #     """
    #     dist = (pred - target) ** 2
    #     md = dist.mean(dim=-1)  # [B, N], mean loss per patch
    #     tmd = (md if mask is None else (md * mask)).sum() / mask.sum()  # mean loss (on unmasked patches only for more focussed training)
    #     self.total_loss += tmd.item()
    #     self.batch_count += 1
    #     return {'loss': tmd}

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Loss calculation. Calculates Pixel-level MSE with masking on pixel level
        All inputs are expected in [B, C, H, W] format.
        The mask is expected in [B, C, H, W] format as well !!! (often, the channels will be identical, but channel-level masking in possible)

        Args:
            :param pred: prediction of the model of shape [B, C, H, W]
            :param target: correct target in patchified shape [B, C, H, W]
            :param mask: Pixel-level mask with '1' for every pixel that should be considered in loss and '0' otherwise, shape [B, C, H, W]
        """
        dist = (pred - target) ** 2
        if mask is not None:
            masked_dist = dist * mask
            mean = masked_dist.sum() / mask.sum()
        else:
            mean = dist.mean()
        self.total_loss += mean.item()
        self.batch_count += 1
        return {'loss': mean}

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
        iou = self._compute_iou(pred_boxes, target_boxes_padded)  # [B, num_queries, num_queries]

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
