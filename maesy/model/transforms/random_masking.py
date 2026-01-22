from typing import Tuple

import torch
import torch.nn as nn

class RandomMasking(nn.Module):
    def __init__(self, mask_ratio: float):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Perform random masking.

            Args:
                x: [B, N, D] - input sequence

            Returns:
                x_masked: [B, N * (1 - mask_ratio), D] - masked sequence
                mask: [B, N] - binary mask (0 is keep, 1 is remove)
                ids_restore: [B, N] - indices to restore original order
            """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
