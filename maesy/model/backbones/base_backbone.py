from dataclasses import dataclass
from typing import Protocol, Tuple, Dict, List
import torch
from torchvision.transforms.v2 import Transform


@dataclass
class BaseConfig:
    """Base configuration for backbones."""
    pass

class BaseBackbone(Protocol):
    """
        Base type for backbones.
        Only used for typing, since this class is a Protocol.
        Backbone implementation must follow this interface design, but they cannot inherit it
    """
    type: str
    config: BaseConfig
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
            Forward pass through the backbone. Should return a dict of feature maps for each requested feature scale, e.g. {"c3": c3_features, "c4": c4_features, "c5": c5_features}
        """
        raise NotImplementedError("get_feature_dims")

    def get_feature_dims(self) -> Dict[str, torch.Size]:
        """
            Return the feature dimensions of the backbone for each feature scale as a dict {scale: feature_dim}
        """
        raise NotImplementedError("get_feature_dims")

    def get_feature_channels(self) -> Tuple[int, ...]:
        """
            Return the number of channels for each feature scale as a tuple
        """
        raise NotImplementedError("get_feature_channels")

    def get_transforms(self) -> List[Transform]:
        """
            Returns a list of backbone-specific transforms that will be applied to the input tensor.
            Especially helpful with pretrained weights that expect certain normalizations
        """
        raise NotImplementedError("get_transforms")