from abc import ABC
from typing import Tuple, Dict

import torch.nn as nn
import torch

from .heads import BaseHead
from .backbones import BaseBackbone


class BaseModel(ABC, nn.Module):
    head: BaseHead
    backbone: BaseBackbone
    is_trainable: bool = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor | Dict[str, torch.Tensor]:
        out = self.backbone.forward(x, **kwargs)
        out = self.head.forward(out, **kwargs)
        return out

    def infer(self, images: torch.Tensor, targets: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Inference method for the model. By default, it just runs a forward pass and returns the raw outputs and targets.
        This can be overridden in specific model implementations to include post-processing steps (e.g., applying softmax, non-max suppression, etc.) before returning the results.

        Args:
            :param images: Input images [B, C, H, W]
            :param targets: Ground truth targets (format depends on the task)
            :param kwargs: Additional arguments for inference (e.g., confidence thresholds, etc.)

        Returns:
            raw_out: raw output of the model
            preds: post-processed output of the model
            targets: targets
        """
        raw_out = self.forward(images, **kwargs)
        preds = raw_out # At this point, model-specific post-processing steps can be applied (usually argmax on class logits, etc.)
        return raw_out, preds, targets

    def update_backbone_conf(self, *args, **kwargs) -> None:
        """
            Update the backbone configuration with new parameters and recreate the bakcbone's affected layers if necessary.
            Subclasses may specify their own arguments instead of kwargs
        """
        raise NotImplementedError("update_backbone_conf method of base_model.py was called, but is not implemented in specific model.")

    def update_head_conf(self, *args, **kwargs) -> None:
        """
        Update the head configuration with new parameters (e.g., number of classes, line class ID, etc.) and recreate the head's classification layers if necessary.
        Subclasses may specify their own arguments instead of kwargs
        """
        raise NotImplementedError("update_head_conf method of base_model.py was called, but is not implemented in specific model.")