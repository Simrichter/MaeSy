from abc import ABC, abstractmethod
import torch.nn as nn

class BaseBackbone(ABC, nn.Module):

    @abstractmethod
    def __init__(self):
        super(BaseBackbone, self).__init__()

    @abstractmethod
    def forward(self,x):
        pass