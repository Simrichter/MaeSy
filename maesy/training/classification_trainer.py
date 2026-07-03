"""Trainer for Object Detection models."""

from typing import Optional, List, Dict

import torch

from maesy.training import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """Trainer for Object Detection Vision Transformer models.

    This trainer handles the training loop, loss computation, and optimization
    specific to object detection models using the DetectionLoss.
    """

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

        return losses