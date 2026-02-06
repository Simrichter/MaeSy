from dataclasses import dataclass
from typing import Protocol
import torch


@dataclass
class BaseHeadConfig:
    pass


class BaseHead(Protocol):
    type: str
    config: BaseHeadConfig

    def forward(self, features: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ...
