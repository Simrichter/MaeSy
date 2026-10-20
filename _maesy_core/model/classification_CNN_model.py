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
    backbone_pretrained: bool = True

    # Classification head parameters
    embed_dim: int = 512
    num_classes: int = 3


class ClassificationCNN(BaseModel[ClassificationCNNConfig]):
    """ResNet-based image classifier."""

    def __init__(self, config: ClassificationCNNConfig):
        """Initialize the classifier with the configured ResNet variant."""
        super().__init__(config)
        feature_level = "c6"
        self.backbone = ResNetBackbone(
            ResNetBackboneConfig(
                version=config.resnet_version,
                pretrained=config.backbone_pretrained,
                feature_scales=(feature_level,),
            )
        )
        # The channel count differs across ResNet families (e.g. 512 for
        # ResNet-18 and 2048 for ResNet-50), so derive the head width from the
        # selected backbone feature instead of the legacy fixed embed_dim.
        head_config = LinearHeadConfig(
            input_dim=self.backbone.get_feature_channels()[0],
            num_classes=config.num_classes,
            feature_level=feature_level,
        )
        self.head = LinearHead(head_config)

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        """Return class logits with shape ``[batch_size, num_classes]``."""
        features = self.backbone(x, *args, **kwargs)
        feature_level = self.head.config.feature_level
        pooled_features = torch.flatten(features[feature_level], start_dim=1)
        return self.head({feature_level: pooled_features}, *args, **kwargs)["out"]

