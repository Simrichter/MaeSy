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
        """
            Forward pass through the model.
        """
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

    def get_device(self) -> torch.device:
        """
            Returns the device on which the model's parameters are located.
            This is useful for ensuring that inputs are moved to the same device as the model before inference.
        """
        return next(self.parameters()).device

    def get_input_dims(self) -> torch.Size:
        """
            Returns the input dimensions of the model as a torch.Size object.
            This is usually the input dimensions of the backbone, but can be overridden in specific model implementations if necessary.
        """
        return self.backbone.get_input_dims()

    def get_output_dims(self) -> Dict[str, torch.Size]:
        """
            Returns the output dimensions of all the model's outputs as a dictionary of torch.Size objects.
            This is usually the output dimensions of the head, but can be overridden in specific model implementations if necessary.
        """
        return self.head.get_output_dims()

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

    def get_model_hash(self) -> str:
        """
            Returns a hash of the model's configuration and parameters.
            This can be used to uniquely identify the model for caching or versioning purposes.
        """
        import hashlib
        import json

        # Create a hash of the model's configuration and parameters
        config_str = json.dumps(self.config.__dict__, sort_keys=True)
        params_str = json.dumps({k: v.tolist() for k, v in self.state_dict().items()}, sort_keys=True)
        combined_str = config_str + params_str
        return f"{self.head.config.type}_{self.backbone.config.type}_{hashlib.md5(combined_str.encode()).hexdigest()}"