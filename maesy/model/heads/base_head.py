from abc import ABC, abstractmethod
import torch.nn as nn

class BaseHead(ABC, nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,img):
        pass