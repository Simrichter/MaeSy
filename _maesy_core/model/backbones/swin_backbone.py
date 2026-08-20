from dataclasses import dataclass, field
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights, swin_t, swin_s, swin_b


@dataclass
class SWINBackboneConfig:
    """
    Config class for SWIN backbones

    Args:
        :param version: SWIN version ('swin_tiny', 'swin_small', 'swin_base') # TODO: Maybe expand to Swin_v2 options?
        :param image_size: Input image size (assumed square)
        :param pretrained: Whether to use pre-trained weights
        :param feature_scales: Specify which feature scale levels to calculate and return during forward pass Selection of: ['c3', 'c4', 'c5']
    """
    version: str = "swin_tiny"  # Options: 'swin_tiny', 'swin_small', 'swin_base'
    type = f"SWINBackbone_{version}"
    image_size: int = 224
    pretrained: bool = True
    feature_scales: Tuple[str, ...] = field(default_factory = ("c3", "c4", "c5"))


def _to_nchw(x):
    return x.permute(0, 3, 1, 2).contiguous()


class SWINBackbone(nn.Module):
    """ResNet Backbone for feature extraction."""

    def __init__(self, config: SWINBackboneConfig, remove_layers: int = 1):
        """
        Initialize SWIN backbone.

        Args:
            :param config: The SWINBackboneConfig
        """
        super().__init__()
        self.config = config

        weights = None
        if config.pretrained:
            weights_map = {
                "swin_tiny": Swin_T_Weights.DEFAULT,
                "swin_small": Swin_S_Weights.DEFAULT,
                "swin_base": Swin_B_Weights.DEFAULT,
            }
            weights = weights_map[config.version]
        constructors = {
            "swin_tiny": swin_t,
            "swin_small": swin_s,
            "swin_base": swin_b
        }
        if config.version not in constructors:
            raise ValueError(f"Unsupported SWIN version: {config.version}")

        self.calc_c3 = "c3" in self.config.feature_scales or "c4" in self.config.feature_scales or "c5" in self.config.feature_scales
        self.calc_c4 = "c4" in self.config.feature_scales or "c5" in self.config.feature_scales
        self.calc_c5 = "c5" in self.config.feature_scales

        self.feature_dim: Dict[str, int] = {"c3":192, "c4":384, "c5":768}  # For Swin Tiny, Small and Base the feature dimensions are the same for the respective layers

        self.spatial_feature_size = {"c3":self.config.image_size // 8, "c4":self.config.image_size // 16, "c5":self.config.image_size // 32}

        model = constructors[config.version](weights=weights)

        self.stem = model.features[0]  # The initial patch embedding layer
        self.layer1 = model.features[1]

        if self.calc_c3:
            self.layer2 = nn.Sequential(
            model.features[2],  # PatchMerging
            model.features[3],  # Stage
        )
        if self.calc_c4:
            self.layer3 = nn.Sequential(
            model.features[4],
            model.features[5],
        )
        if self.calc_c5:
            self.layer4 = nn.Sequential(
            model.features[6],
            model.features[7],
        )

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
        return {k: _to_nchw(feature_maps[k]) for k in self.config.feature_scales}

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
