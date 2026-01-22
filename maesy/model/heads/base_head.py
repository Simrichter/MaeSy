from typing import Protocol
import torch

class BaseHead(Protocol):

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def __call__(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(features, **kwargs)