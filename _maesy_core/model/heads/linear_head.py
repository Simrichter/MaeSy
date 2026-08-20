from typing import Dict

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class LinearHeadConfig:
    type = "LinearHead"

    input_dim: int = 256
    num_classes: int = 6
    feature_level: str = "c5"

class LinearHead(nn.Module):
    def __init__(self, config: LinearHeadConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.input_dim, config.num_classes)


    def forward(self, features: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        return {"out": self.linear(features[self.config.feature_level])}
