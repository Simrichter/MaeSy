from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights

@dataclass
class ResNetBackboneConfig:
    version: str

class ResNetBackbone(nn.Module):
    """ResNet Backbone for feature extraction."""

    def __init__(self, version: str = 'resnet50'):
        """
        Initialize ResNet backbone.

        Args:
            :param version: ResNet version ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        """
        super().__init__()
        self.type = f"ResNetBackbone_{version}"
        self.config = ResNetBackboneConfig(version=version)

        model = None
        match version:
            case'resnet18':
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            case 'resnet34':
                model = resnet34(weights=ResNet34_Weights.DEFAULT)
            case 'resnet50':
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
            case 'resnet101':
                model = resnet101(weights=ResNet101_Weights.DEFAULT)
            case 'resnet152':
                model = resnet152(weights=ResNet152_Weights.DEFAULT)
            case _:
                raise ValueError(f"Unsupported ResNet version: {version}")

        modules = list(model.children())[:-1] # Remove the last classification layer
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
        # x = torch.flatten(x, 1)  # Flatten to [B, feature_dim]
        return x