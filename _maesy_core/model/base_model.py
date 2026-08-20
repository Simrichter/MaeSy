from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Dict, TypeVar, Generic

import torch.nn as nn
import torch

from .heads import BaseHead
from .backbones import BaseBackbone

@dataclass
class BaseConfig:
    """
    Base configuration class for models.
    This class can be extended to include specific configuration parameters for different model architectures.
    A unique 'type' field is included to identify the model type in the configuration.
    """
    type: str

ConfigT = TypeVar("ConfigT", bound=BaseConfig)

class BaseModel(ABC, nn.Module, Generic[ConfigT]):
    head: BaseHead
    backbone: BaseBackbone
    config: ConfigT
    is_trainable: bool = True

    def __init__(self, config: ConfigT) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor | Dict[str, torch.Tensor]:
        out = self.backbone.forward(x, **kwargs)
        return self.head.forward(out, **kwargs)

    def infer(self, images: torch.Tensor, targets: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

    def get_export_wrapper(self):
        """
            Returns a wrapper for the model that is suitable for exporting to ONNX.
            Efficiency optimizations or stripping from auxilliary training outputs can be done here
        """
        return self