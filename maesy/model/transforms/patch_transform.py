import torch
import torch.nn as nn
from ..components import Utils

class PatchifyTransform(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Utils.patchify(x, image_size=x.shape[2], patch_size=self.patch_size, in_channels=x.shape[1])