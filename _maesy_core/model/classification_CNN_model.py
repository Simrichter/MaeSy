"""Classification Vision Transformer for pretraining."""
from dataclasses import dataclass

import torch

from .base_model import BaseModel, BaseConfig
from .heads import LinearHead, LinearHeadConfig
from .backbones import ResNetBackbone, ResNetBackboneConfig


@dataclass
class ClassificationCNNConfig(BaseConfig):
    """
    Configuration for Vision Transformer Detector model.
    """
    type: str = "classification_cnn"
    # ResNet backbone parameters
    resnet_version: str = "resnet50"

    # Classification head parameters
    embed_dim: int = 512
    num_classes: int = 3


class ClassificationCNN(BaseModel[ClassificationCNNConfig]):
    """Vision Transformer for image classification pretraining.

    This model implements standard supervised image classification pretraining
    using the ViT architecture.
    """

    def __init__(self, config: ClassificationCNNConfig):
        """
        Initialize classification model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        head_config = LinearHeadConfig(
            input_dim=config.embed_dim,
            num_classes=config.num_classes
        )
        self.backbone = ResNetBackbone(ResNetBackboneConfig(version="resnet18"))
        self.head = LinearHead(head_config)

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.backbone(x)
        # print("Features shape:", features.shape)  # Debug print to check feature shape
        out = self.head(torch.flatten(features, 1))
        return out

