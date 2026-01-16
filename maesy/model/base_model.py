from abc import ABC, abstractmethod
from typing import Protocol, Any
import torch.nn as nn
import torch
from .heads import BaseHead
from .backbones import BaseBackbone


class BaseModel(ABC, nn.Module):
    head: BaseHead
    backbone: BaseBackbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    # def __call__(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.forward(x)
