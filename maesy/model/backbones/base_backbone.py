from typing import Protocol
import torch

class BaseBackbone(Protocol):

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ...