from dataclasses import dataclass
from typing import Protocol, Tuple, Dict
import torch

@dataclass
class BaseConfig:
    """Base configuration for backbones."""
    pass

class BaseBackbone(Protocol):
    type: str
    config: BaseConfig
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ...

    def get_feature_dims(self) -> Dict[str, torch.Size]:
        ...

    def get_feature_channels(self) -> Tuple[int, int, int]:
        ...