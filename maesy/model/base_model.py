from abc import ABC
import torch.nn as nn
import torch
from .heads import BaseHead
from .backbones import BaseBackbone


class BaseModel(ABC, nn.Module):
    head: BaseHead
    backbone: BaseBackbone

    # def preprocess(self, x: torch.Tensor) -> torch.Tensor:
    #     """Model-specific preprocessing like positional embeddings or patchification. If not overwritten, behaves as identity.
    #
    #     Args:
    #         x: Input images [B, C, H, W]
    #
    #     Returns:
    #         Preprocessed images of desired shape
    #     """
    #     return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(
                self.backbone(
                    self.backbone.preprocess(
                        x
                    )
                )
        )
