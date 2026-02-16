from abc import ABC

import torch.nn as nn
import torch

from .heads import BaseHead
from .backbones import BaseBackbone


class BaseModel(ABC, nn.Module):
    head: BaseHead
    backbone: BaseBackbone

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = self.backbone.forward(x, **kwargs)
        out = self.head.forward(out, **kwargs)
        return out

    def infer(self, images, targets, **kwargs):
        return self.forward(images, **kwargs).detach(), targets
