"""Classification Vision Transformer for pretraining."""
from dataclasses import dataclass

import torch

from maesy.model.config import ModelConfig
from .base_model import BaseModel
from .heads import LinearHead, LinearHeadConfig
from .backbones import ResNetBackbone, ResNetBackboneConfig


@dataclass
class PatchClassificatorConfig(ModelConfig):
    """
    Configuration for Vision Transformer Detector model.
    """
    # Resnet backbone parameters
    resnet_version: str = "resnet50"
    feature_scale: str = "c3"

    # Classification head parameters
    head_in_dim: int = 2304
    num_classes: int = 3


class PatchClassificator(BaseModel):
    """ Network for patch classification
    """

    def __init__(self, config: PatchClassificatorConfig):
        """
        Initialize classification model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        head_config = LinearHeadConfig(
            input_dim=config.head_in_dim,
            num_classes=config.num_classes
        )
        self.backbone = ResNetBackbone(ResNetBackboneConfig(version=self.config.resnet_version, feature_scales=(self.config.feature_scale,)))
        self.backbonetype = "ResnetBackbone"

        self.head = LinearHead(head_config)
        self.headtype = "LinearHead"

    def forward(self, x):
        """Forward pass through the model."""
        features = self.backbone(x)
        # print("Features shape:", features[self.config.feature_scale].shape)  # Debug print to check feature shape
        out = self.head(torch.flatten(features[self.config.feature_scale], 1))
        return out

