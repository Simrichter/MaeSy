from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Size
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights, MobileNetV2, MobileNet_V2_Weights
from torchvision.models.quantization import mobilenet_v2
from wandb.util import downsample


@dataclass
class MobileNetBackboneConfig:
    version: str

class MobileNetBackbone(nn.Module):
    """ResNet Backbone for feature extraction."""

    def __init__(self, version: str, image_size: int, remove_layers: int = 1):
        """
        Initialize ResNet backbone.

        Args:
            :param version: Currently unused
            :param image_size: Input image size (assumed square) (currently unused)
            :param remove_layers: Number of layers to remove from the end (default: 1, removes the classification layer but keeps global average pooling)
        """
        super().__init__()
        self.type = f"MobileNetBackbone_{version}"
        self.config = MobileNetBackboneConfig(version=version)

        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.feature_dim = 1280
        self.spatial_feature_size = image_size // 32 # TODO: Validate this for different remove_layers values

        modules = list(model.children())[:-remove_layers] # Remove the last classification layer
        self.model = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            :param x: Input images [B, C, H, W]
        Returns:
            :return: Extracted features [B, feature_dim]
        """
        x = self.model(x)
        return x

    def get_feature_dims(self) -> Size:
        """
        Get the output feature dimension of the backbone.

        Returns:
            :return: Feature dimension
        """

        return torch.Size((self.feature_dim, self.spatial_feature_size, self.spatial_feature_size))