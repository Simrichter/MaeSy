"""Trainer for Object Detection models."""

from typing import Optional, List, Dict

import cv2
import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from maesy.evaluation.metrics import (
    compute_detection_metrics,
    prepare_targets_for_detection_metrics,
)
from maesy.evaluation.visualizer import draw_objects_in_tensor
from maesy.training import BaseTrainer


class DetectionTrainer(BaseTrainer):
    """Trainer for Object Detection Vision Transformer models.

    This trainer handles the training loop, loss computation, and optimization
    specific to object detection models using the DetectionLoss.
    """

    def _validation_start(self):
        self._val_predictions: List[Dict[str, torch.Tensor]] = []
        self._val_targets: List[Dict[str, torch.Tensor]] = []

    def _validation_step(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]], losses: Dict[str, torch.Tensor]):
        if targets is None:
            return
        decoded = losses.get("__decoded_predictions")
        prepared_targets = losses.get("__prepared_targets")
        if decoded is not None:
            self._val_predictions.extend(decoded)
        if prepared_targets is not None:
            self._val_targets.extend(prepared_targets)

    def _validation_finalize(self) -> Dict[str, float]:
        if len(self._val_predictions) == 0 or len(self._val_targets) == 0:
            return {}
        line_class_id = getattr(self.model.config, "line_class_id", None)
        if line_class_id is not None and line_class_id < 0:
            line_class_id = None
        return compute_detection_metrics(
            predictions=self._val_predictions,
            targets=self._val_targets,
            num_classes=self.model.config.num_classes,
            line_class_id=line_class_id,
        )

    def _render_target_boxes(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> torch.Tensor:
        img = image.detach().cpu()
        img = (img * self._IMAGENET_STD) + self._IMAGENET_MEAN
        img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        boxes = target["boxes"].detach().cpu()
        labels = target["labels"].detach().cpu()
        if boxes.numel() == 0:
            return img.float() / 255.0

        h, w = img.shape[-2:]
        boxes_xyxy = box_convert(boxes, "cxcywh", "xyxy")
        boxes_xyxy[:, (0, 2)] *= w
        boxes_xyxy[:, (1, 3)] *= h
        boxes_xyxy[:, (0, 2)] = boxes_xyxy[:, (0, 2)].clamp(0, w - 1)
        boxes_xyxy[:, (1, 3)] = boxes_xyxy[:, (1, 3)].clamp(0, h - 1)

        rendered_labels = [f"GT:C{label.item()}" for label in labels]
        drawn = draw_bounding_boxes(img, boxes_xyxy, labels=rendered_labels, colors="green", width=2)
        return drawn.float() / 255.0

    def _forward_model(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]], val: bool) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Input images [B, C, H, W]
            targets: List of target dictionaries containing 'boxes' and 'labels'
            val: Whether this is a validation pass

        Returns:
            Dictionary of losses from the loss function
        """
        # Get model predictions
        raw_out , predictions, _ = self.model.infer(images, targets=targets)

        # Compute loss
        losses = self.loss(raw_out, targets)

        if val and targets is not None:
            line_class_id: int | None = getattr(self.model.config, "line_class_id", None)
            if line_class_id is not None and line_class_id < 0:
                line_class_id = None
            ellipse_class_id: int | None = getattr(self.model.config, "ellipse_class_id", None)
            if ellipse_class_id is not None and ellipse_class_id < 0:
                ellipse_class_id = None

            losses["__decoded_predictions"] = predictions

            losses["__prepared_targets"] = prepare_targets_for_detection_metrics(
                targets,
                line_class_id=line_class_id,
                ellipse_class_id=ellipse_class_id,
            )

            # Unnormalize to RGB uint8 so draw_bounding_boxes always renders visible overlays.
            img = images[0].detach()
            img = (img * self._IMAGENET_STD) + self._IMAGENET_MEAN
            img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()
            losses["img_predictions"] = draw_objects_in_tensor(img, predictions[0]["boxes"], predictions[0]["labels"], predictions[0]["line_points"], predictions[0]["ellipses"], xyxy=True)/255

        return losses