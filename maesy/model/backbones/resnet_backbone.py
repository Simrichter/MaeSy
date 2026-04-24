from dataclasses import dataclass, field
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights


@dataclass
class ResNetBackboneConfig:
    """
        Config class for ResNet backbones

        Args:
            :param version: ResNet version ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            :param image_size: Input image size (assumed square)
            :param pretrained: Whether to use pre-trained weights
            :param feature_scales: Specify which feature scale levels to calculate and return during forward pass
    """
    version: str = "resnet50"
    image_size: int = 224
    pretrained: bool = True
    feature_scales: Tuple[str, ...] = field(default_factory = ("c3", "c4", "c5"))

class ResNetBackbone(nn.Module):
    """ResNet Backbone for feature extraction."""

    def __init__(self, config: ResNetBackboneConfig):
        """
        Initialize ResNet backbone.

        Args:
            :param config: ResNetBackboneConfig class that holds all necessary parameters
        """
        super().__init__()
        self.config = config
        self.type = f"ResNetBackbone_{self.config.version}"

        weights = None
        if config.pretrained:
            weights_map = {
                "resnet18": ResNet18_Weights.DEFAULT,
                "resnet34": ResNet34_Weights.DEFAULT,
                "resnet50": ResNet50_Weights.DEFAULT,
                "resnet101": ResNet101_Weights.DEFAULT,
                "resnet152": ResNet152_Weights.DEFAULT,
            }
            weights = weights_map[config.version]
        constructors = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
            "resnet152": resnet152,
        }
        if config.version not in constructors:
            raise ValueError(f"Unsupported ResNet version: {config.version}")

        self.calc_c3 = "c3" in self.config.feature_scales or "c4" in self.config.feature_scales or "c5" in self.config.feature_scales or "c6" in self.config.feature_scales
        self.calc_c4 = "c4" in self.config.feature_scales or "c5" in self.config.feature_scales or "c6" in self.config.feature_scales
        self.calc_c5 = "c5" in self.config.feature_scales or "c6" in self.config.feature_scales
        self.calc_c6 = "c6" in self.config.feature_scales

        self.feature_dim: Dict[str, int] = {"c3":128, "c4":256, "c5":512} if self.config.version in {"resnet18", "resnet34"} else {"c3":512, "c4":1024, "c5":2048}

        self.spatial_feature_size = {"c3":self.config.image_size // 8, "c4":self.config.image_size // 16, "c5":self.config.image_size // 32}

        model = constructors[config.version](weights=weights)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        if self.calc_c3:
            self.layer2 = model.layer2
        if self.calc_c4:
            self.layer3 = model.layer3
        if self.calc_c5:
            self.layer4 = model.layer4
        if self.calc_c6:
            self.avgPool = model.avgpool

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            :param x: Input images [B, C, H, W]
        Returns:
            :return: Dict of extracted features {scale: Tensor[B, feature_dim]}
        """
        x = self.stem(x)
        x = self.layer1(x)
        feature_maps = {}
        if self.calc_c3:
            c3 = self.layer2(x)
            feature_maps["c3"] = c3
            if self.calc_c4:
                c4 = self.layer3(c3)
                feature_maps["c4"] = c4
                if self.calc_c5:
                    c5 = self.layer4(c4)
                    feature_maps["c5"] = c5
                    if self.calc_c6:
                        c6 = self.avgPool(c5)
                        feature_maps["c6"] = c6
        return {k: feature_maps[k] for k in self.config.feature_scales}

    def get_feature_dims(self) -> Dict[str, torch.Size]:
        """
        Get the output feature dimension of the backbone.

        Returns:
            :return: Tuple of feature dimensions for every scale that the backbone will return
        """
        return {k: torch.Size((self.feature_dim[k], self.spatial_feature_size[k], self.spatial_feature_size[k])) for k in self.config.feature_scales}

    def get_feature_channels(self) -> Tuple[int, ...]:
        """
            Return the number of channels for each feature scale as a tuple
        """
        return tuple(self.feature_dim[k] for k in self.config.feature_scales)
