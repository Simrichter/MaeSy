from typing import Protocol
import torch

class BaseHead(Protocol):

    def forward(self, features: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ...