from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from ..components import PatchEmbedding, TransformerBlock, Utils

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
        # self.patch_embed = PatchEmbedding(**asdict(config))
        self.patch_embed = nn.Linear(config.embed_dim, config.embed_dim)

        # Class token
        # self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        # Positional encoding
        # self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches + 1, config.embed_dim) * 0.02)
        self.pos_embed = Utils.get_sinusoidal_encoding(config.num_patches, config.embed_dim)
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(**asdict(config)) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

        Utils.init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Preprocessed tokens [B, N_visible + 1, D]

        Returns:
            x: Encoded visible patches (+cls token) [B, N_visible(+1), D]
        """
        x = self.pos_dropout(x)

        # Transformer encoder
        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)

        return x

    def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess input images by applying patch embedding, positional embedding, random masking and attaching a class token

        Args:
            x: Input images [B, C, H, W]
        Returns:
            x: Preprocessed tokens [B, N_visible + 1, D]
            mask: Binary mask
            ids_restore: Indices to restore original order
        """
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add positional encoding (without cls token)
        x = x + self.pos_embed[:, 1:, :]

# TODO: use ids_restore + mask such that random_masking can be done outside the model

        # Masking: length -> length * (1 - mask_ratio)
        x, mask, ids_restore = self.random_masking(x, self.config.mask_ratio)

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x, mask, ids_restore