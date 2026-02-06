from dataclasses import dataclass
from typing import Protocol
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