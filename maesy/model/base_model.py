from abc import ABC
from typing import Any

import torch.nn as nn
import torch
from torch import Tensor

from .heads import BaseHead
from .backbones import BaseBackbone


class BaseModel(ABC, nn.Module):
    head: BaseHead
    backbone: BaseBackbone

    def preprocess(self, x: torch.Tensor) -> tuple[Tensor, dict[str, Any]]:
        """Model-specific preprocessing like patchification or input masking.
        If not overwritten, behaves as identity.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            x: Preprocessed images of desired shape
            preprocess_data: Further information that might be used in postprocessing
        """
        return x, {}

    def postprocess(self, model_out, preprocess_data: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """Model-specific postprocessing function to turn the output of the model into its final form.
        If not overwritten, behaves as identity.

        Args:
            model_out: The raw output of the model
            preprocess_data: Information from preprocessing that is used for postprocessing

        Returns:
            final_out: The postprocessed output of the model
        """
        return model_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(
                self.backbone(
                    self.backbone.preprocess(
                        x
                    )
                )
        )
