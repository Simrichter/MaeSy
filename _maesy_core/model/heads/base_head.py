from dataclasses import dataclass
from typing import Protocol, Dict
import torch


@dataclass
class BaseHeadConfig(Protocol):
    """
        Base type for head configurations.
    """
    type: str


class BaseHead(Protocol):
    """
        Base type for heads.
        Only used for typing, since this class is a Protocol.
        Head implementation must follow this interface design, but they cannot inherit it
    """
    config: BaseHeadConfig

    def forward(self, features: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
            Forward pass through the head. Should return the final output of the model, e.g. class logits, bounding box predictions, etc.
        """
        raise NotImplementedError("forward")
