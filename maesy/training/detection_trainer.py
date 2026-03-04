"""Trainer for Object Detection models."""

from typing import Optional, List, Dict

import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from maesy.evaluation.visualizer import draw_boxes_in_image
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

        # Compute loss
        losses = self.loss(predictions, targets)

        if val:
            # return losses # TODO
            name_coding = {
                0: "Ball",  # TODO: Get this stuff from model config?
                1: "Robot",
                2: "PenaltyCross"  # (Keine 27 Beschriftungen erwünscht), alternativ: LineCrossing
            }

            pred_logits = predictions['pred_logits'][0]
            pred_boxes = predictions['pred_boxes'][0]
            mask = pred_logits.argmax(dim=-1) != 3
            filtered_logits = pred_logits#[mask] TODO
            filtered_boxes = box_convert(pred_boxes, "cxcywh", "xyxy") * images[0].shape[-1] # TODO [mask], Assuming square images and normalized box coordinates

            labels = [name_coding[l] for l in filtered_logits.argmax(dim=-1).cpu().tolist()]
            # print(f"Predicted labels: {labels}")
            # print(f"Predicted boxes: {filtered_boxes.cpu().tolist()}")
            img = images[0].cpu()
            # undo this transform: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img_prediction = draw_boxes_in_image(img, filtered_boxes, labels)
            losses["img_out"] = img_prediction

        return losses