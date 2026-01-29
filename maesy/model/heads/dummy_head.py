import torch
import torch.nn as nn

from maesy.model.heads import BaseHead


class DummyHead(nn.Module):
    """
    A dummy head that performs no operation.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, features: torch.Tensor, *args,  **kwargs) -> torch.Tensor:
        """
        Dummy forward pass that returns the input as is.
        :param features: Input features
        :return: Unchanged input features
        """
        return features
