from dataclasses import dataclass, field
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.quantization import mobilenet_v2
import timm


@dataclass
class MobileNetBackboneConfig:
    """
    Config class for MobileNet backbones

    Args:
        :param version: MobileNet version ('mobilenetv2' is currently the only option)
        :param image_size: Input image size (assumed square)
        :param pretrained: Whether to use pre-trained weights
        :param feature_scales: Specify which feature scale levels to calculate and return during forward pass (following resnet naming scheme)
    """
    version: str = "mobilenetv2"
    image_size: int = 224
    pretrained: bool = True
    feature_scales: Tuple[str, ...] = field(default_factory=lambda: ("c3", "c4", "c5"))


class MobileNetBackbone(nn.Module):
    """MobileNet Backbone for feature extraction."""

    def _init_v2(self):
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if self.config.pretrained else None)
        # Feature dimensions for MobileNetV2 at each scale
        self.feature_dim: Dict[str, int] = {"c3": 32, "c4": 96, "c5": 1280}

        # Spatial sizes at each scale
        self.spatial_feature_size = {"c3": self.config.image_size // 8, "c4": self.config.image_size // 16, "c5": self.config.image_size // 32}

        # Extract layers based on stride reduction points
        # Layers 0-4: stem + layers up to stride-8, output 32 channels
        self.stem = nn.Sequential(*list(model.features.children())[:4])

        if self.calc_c3:
            # Layers 4-6: to stride-8, output 32 channels
            self.layer2 = nn.Sequential(*list(model.features.children())[4:7])

        if self.calc_c4:
            # Layers 4-13: from stride-8 to stride-16, output 96 channels
            self.layer3 = nn.Sequential(*list(model.features.children())[7:14])

        if self.calc_c5:
            # Layers 14-18: from stride-16 to stride-32, output 1280 channels
            self.layer4 = nn.Sequential(*list(model.features.children())[14:])


    def _init_v4(self):
        if self.config.version == 'mobilenetv4_s':
            model = timm.create_model('hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k', pretrained=self.config.pretrained, features_only=True)
            self.feature_dim: Dict[str, int] = {"c3": 64, "c4": 96, "c5": 128}
            self.spatial_feature_size = {"c3": self.config.image_size // 8, "c4": self.config.image_size // 16, "c5": self.config.image_size // 32}

        elif self.config.version == 'mobilenetv4_m':
            model = timm.create_model('hf_hub:timm/mobilenetv4_conv_medium.e500_r256_in1k', pretrained=True, features_only=True)
            self.feature_dim: Dict[str, int] = {"c3": 80, "c4": 160, "c5": 256}
            self.spatial_feature_size = {"c3": self.config.image_size // 8, "c4": self.config.image_size // 16, "c5": self.config.image_size // 32}
        else:
            raise ValueError(f"Unknown MobileNetv4 version: {self.config.version}")

        self.stem = nn.Sequential(model.conv_stem, model.bn1, model.act1)
        if self.calc_c3:
            self.layer2 = nn.Sequential(model.blocks[0], model.blocks[1]) # stride 8
        if self.calc_c4:
            self.layer3 = model.blocks[2] # stride 16
        if self.calc_c5:
            self.layer4 = model.blocks[3] # stride 32


    def __init__(self, config: MobileNetBackboneConfig):
        """
        Initialize MobileNet backbone.

        Args:
            :param config: MobileNetBackboneConfig class that holds all necessary parameters
        """
        super().__init__()
        self.config = config
        self.type = self.config.version

        self.calc_c3 = "c3" in self.config.feature_scales or "c4" in self.config.feature_scales or "c5" in self.config.feature_scales
        self.calc_c4 = "c4" in self.config.feature_scales or "c5" in self.config.feature_scales
        self.calc_c5 = "c5" in self.config.feature_scales

        if self.config.version.startswith("mobilenetv2"):
            self._init_v2()
        elif self.config.version.startswith("mobilenetv4"):
            self._init_v4()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            :param x: Input images [B, C, H, W]
        Returns:
            :return: Dict of extracted features {scale: Tensor[B, channels, H, W]}
        """
        x = self.stem(x)
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
        return {k: feature_maps[k] for k in self.config.feature_scales}

    def get_feature_dims(self) -> Dict[str, torch.Size]:
        """
        Get the output feature dimension of the backbone.

        Returns:
            :return: Dict of feature dimensions for every scale that the backbone will return
        """
        return {k: torch.Size((self.feature_dim[k], self.spatial_feature_size[k], self.spatial_feature_size[k])) for k in self.config.feature_scales}

    def get_feature_channels(self) -> Tuple[int, ...]: # TODO: Make dict[str, int]
        """
        Return the number of channels for each feature scale as a tuple
        """
        return tuple(self.feature_dim[k] for k in self.config.feature_scales)