from abc import ABC, abstractmethod
import torch.nn as nn
from .heads import BaseHead
from .backbones import BaseBackbone


class BaseModel(ABC, nn.Module):

    @abstractmethod
    def __init__(self):
        super(BaseModel, self).__init__()
        self.backbone: BaseBackbone = None
        self.head: BaseHead = None

    def forward(self, x):
        return self.head(self.backbone(x))
