"""Trainer for Object Detection models."""

from typing import Optional, List, Dict

import torch

from maesy.training import BaseTrainer


class DetectionTrainer(BaseTrainer):
    """Trainer for Object Detection Vision Transformer models.

    This trainer handles the training loop, loss computation, and optimization
    specific to object detection models using the DetectionLoss.
    """

    def forward_model(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]], val: bool) -> Dict[str, torch.Tensor]:
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
        predictions = self.model.forward(images)
        
        # Move targets to device
        if targets is not None:
            targets_device = []
            for target in targets:
                targets_device.append({
                    'boxes': target['boxes'].to(self.device, non_blocking=True),
                    'labels': target['labels'].to(self.device, non_blocking=True)
                })
            targets = targets_device
        
        # Compute loss
        losses = self.loss(predictions, targets)

        return losses