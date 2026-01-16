import torch.nn as nn
from dataclasses import dataclass

@dataclass
class LinearHeadConfig:
    embed_dim: int = 256
    num_classes: int = 6

class LinearHead(nn.Module):
    def __init__(self, config: LinearHeadConfig):
        super().__init__()
        self.linear = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, x):
        return self.linear(x)