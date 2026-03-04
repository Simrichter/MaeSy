from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Size
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from wandb.util import downsample


@dataclass
class ResNetBackboneConfig:
    version: str

class ResNetBackbone(nn.Module):
    """ResNet Backbone for feature extraction."""

    def __init__(self, version: str, image_size: int, remove_layers: int = 1):
        """
        Initialize ResNet backbone.

        Args:
            :param version: ResNet version ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            :param image_size: Input image size (assumed square)
            :param remove_layers: Number of layers to remove from the end (default: 1, removes the classification layer but keeps global average pooling)
        """
        super().__init__()
        self.type = f"ResNetBackbone_{version}"
        self.config = ResNetBackboneConfig(version=version)

        model = None

        match remove_layers:
            case 1:
                self.spatial_feature_size = image_size // image_size
            case 2:
                self.spatial_feature_size = image_size // 32
            case 3:
                self.spatial_feature_size = image_size // 16
            case 4:
                self.spatial_feature_size = image_size // 8
            case _:
                raise ValueError(f"Unsupported remove_layers value: remove_layers={remove_layers}")

        match version:
            case'resnet18':
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.feature_dim = 512//(2**(remove_layers-2))
            case 'resnet34':
                model = resnet34(weights=ResNet34_Weights.DEFAULT)
                self.feature_dim = 512//(2**(remove_layers-2)) # TODO: Validate
            case 'resnet50':
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.feature_dim = 2048//(2**(remove_layers-2))
            case 'resnet101':
                model = resnet101(weights=ResNet101_Weights.DEFAULT)
                self.feature_dim = 2048//(2**(remove_layers-2)) # TODO: Validate
            case 'resnet152':
                model = resnet152(weights=ResNet152_Weights.DEFAULT)
                self.feature_dim = 2048//(2**(remove_layers-2)) # TODO: Validate
            case _:
                raise ValueError(f"Unsupported ResNet version: {version}")
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