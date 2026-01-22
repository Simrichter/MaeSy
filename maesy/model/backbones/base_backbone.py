from typing import Protocol
import torch

class BaseBackbone(Protocol):

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x, **kwargs)