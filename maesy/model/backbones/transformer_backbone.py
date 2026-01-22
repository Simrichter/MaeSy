from typing import Tuple

import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from ..components import TransformerBlock, Utils

@dataclass
class TransformerBackboneConfig:
    """Configuration for Vision Transformer Backbone"""

    # Image parameters
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Transformer parameters
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        assert self.image_size % self.patch_size == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"
        self.num_patches = (self.image_size // self.patch_size) ** 2


class TransformerBackbone(nn.Module):
    def __init__(self, config: TransformerBackboneConfig):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Linear(config.patch_size**2*config.in_channels, config.embed_dim)

        # Class token

        # Positional encoding
        self.pos_embed = Utils.get_sinusoidal_encoding(config.num_patches, config.embed_dim)
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(**asdict(config)) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

        # Utils.init_weights(self)

    def forward(self, x: torch.Tensor, ids_shuffle: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Preprocessed tokens [B, N_visible, patch_size*patch_size*in_channels]
            .param ids_shuffle: Indices that were used to shuffle the patches [B, N]
        Returns:
            x: Encoded visible patches (+cls token) [B, N_visible, D]
        """
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        B, N, D = x.shape

        x = x + torch.gather(self.pos_embed.repeat(B, 1, 1), dim=1, index=ids_shuffle[:, :N].unsqueeze(-1).repeat(1, 1, D))
        # x = x + self.pos_embed[:, :]

        x = self.pos_dropout(x)

        # Transformer encoder
        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)

        return x