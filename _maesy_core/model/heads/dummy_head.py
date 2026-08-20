from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from _maesy_core.model.heads.base_head import BaseHeadConfig


@dataclass
class DummyHeadConfig(BaseHeadConfig):
    type = "DummyHead"

class DummyHead(nn.Module):
    """
    A dummy head that performs no operation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.config = DummyHeadConfig()

    def forward(self, features: Dict[str, torch.Tensor], *args,  **kwargs) -> Dict[str, torch.Tensor]:
        """
        Dummy forward pass that returns the input as is.
        :param features: Input features
        :return: Unchanged input features
        """
        return features
