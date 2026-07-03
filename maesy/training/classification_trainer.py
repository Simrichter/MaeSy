"""Trainer for Object Detection models."""

from typing import Optional, List, Dict

import torch

from maesy.evaluation.metrics import compute_classification_metrics
from maesy.training import BaseTrainer


class ClassificationTrainer(BaseTrainer):
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
        return compute_classification_metrics(self._val_predictions, self._val_targets)


    def _forward_model(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]], val: bool) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Input images [B, C, H, W]
            targets: Targets for classification (e.g., class labels) [B, num_classes]
            val: Whether this is a validation pass

        Returns:
            Dictionary of losses from the loss function
        """
        # Get model predictions
        predictions = self.model.forward(images)
        # Compute loss
        losses = self.loss(predictions, targets)
        losses["__decoded_predictions"] = predictions.argmax(dim=-1)
        losses["__prepared_targets"] = targets

        return losses