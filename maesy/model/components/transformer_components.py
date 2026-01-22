from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""

    def __init__(self, patch_size: int, embed_dim: int, in_channels: int, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolutional layer for patch embedding
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        x = self.projection(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return self.dropout(x + self.pos_embed)

class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    class MultiHeadAttention(nn.Module):
        """Multi-head self-attention layer."""

        def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
            super().__init__()
            assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) % num_heads ({num_heads}) is not zero ({embed_dim % num_heads})"

            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim ** -0.5

            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.attn_dropout = nn.Dropout(dropout)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.proj_dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            B, N, C = x.shape

            # Compute Q, K, V
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)

            # Combine heads
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_dropout(x)

            return x

    class MLP(nn.Module):
        """MLP layer."""

        def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.1):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = nn.GELU()
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_features, in_features)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.dropout2(x)
            return x

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1, attention_dropout: float = 0.1, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = self.MultiHeadAttention(
            embed_dim,
            num_heads,
            attention_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = self.MLP(
            embed_dim,
            int(embed_dim * mlp_ratio),
            dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Utils:
    """Utility functions for transformer components."""

    @staticmethod
    def get_sinusoidal_encoding(context_length: int, embedding_dimension: int, ) -> torch.Tensor:
        """
        This function creates a positional embedding using sinusoidal encoding matrix
        :returns Tensor of shape [context_length, embed_dim]
        """

        def get_single_encoding(embed_dim, pos):
            """
            This function returns an encoding vector for a given position.
            It is used as a helper function to create the positional embedding matrix and to extend it if needed in the GPT model.
            """
            return [np.sin(pos / np.power(10000, i / embed_dim)) if i % 2 == 0 else np.cos(
                pos / np.power(10000, (i - 1) / embed_dim)) for i in range(embed_dim)]

        return torch.FloatTensor(
            [get_single_encoding(embedding_dimension, i) for i in range(context_length)]).unsqueeze(0)

    @staticmethod
    def init_weights(module: nn.Module):
        """Initialize weights."""
        # Initialize patch embedding projection
        w = module.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls token and pos embedding
        nn.init.normal_(module.cls_token, std=0.02)
        nn.init.normal_(module.pos_embed, std=0.02)

        # Initialize linear layers
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def patchify(imgs: torch.Tensor, image_size: int, patch_size: int, in_channels: int = 3) -> torch.Tensor:
        """
        Convert images to patches.

        Args:
            imgs: [B, C, H, W]
            image_size: Size of the image (assumes square images)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for colored images)

        Returns:
            patches: [B, num_patches, patch_size**2 * C]
        """
        p = patch_size
        h = w = image_size // p

        x = imgs.reshape(imgs.shape[0], in_channels, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * in_channels)
        return x

    @staticmethod
    def unpatchify(x: torch.Tensor, image_size: int, patch_size: int, in_channels: int = 3) -> torch.Tensor:
        """
        Convert patches to images.

        Args:
            x: [B, num_patches, patch_size**2 * C]
            image_size: Size of the image (assumes square images)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for colored images)

        Returns:
            imgs: [B, C, H, W]
        """
        p = patch_size
        h = w = image_size // p

        x = x.reshape(x.shape[0], h, w, p, p, in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], in_channels, h * p, w * p)
        return imgs

    @staticmethod
    def random_masking(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Perform random masking.

            Args:
                x: [B, N, P**2*C] - input sequence
                mask_ratio: float - ratio of patches to mask

            Returns:
                x_masked: [B, N * (1 - mask_ratio), P**2*C] - masked sequence
                mask: [B, N] - binary mask (0 is keep, 1 is remove)
                ids_restore: [B, N] - indices to restore original order
            """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

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

        return x_masked, mask, ids_shuffle