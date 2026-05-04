from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaeMultiscaleDecoderConfig:
    embed_dim: int
    num_patches: int
    patch_size: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attention_dropout: float
    num_layers: int
    in_channels: int
    feature_dims: Dict[str, int]
    feature_scales: Tuple[str, ...] = field(default_factory=lambda: ("c3", "c4", "c5"))
    use_skip_connections: bool = True
    skip_scales: Tuple[str, ...] = field(default_factory=lambda: ("c3", "c4"))
    window_size: int = 7


class _WindowAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        out, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)
        return out


class _SwinDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float, attention_dropout: float, window_size: int, shift_size: int):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _WindowAttention(embed_dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        x = x.view(b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size * self.window_size, c)
        return windows

    def _window_reverse(self, windows: torch.Tensor, b: int, h: int, w: int, c: int) -> torch.Tensor:
        x = windows.view(b, h // self.window_size, w // self.window_size, self.window_size, self.window_size, c)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)

    def _build_shift_attention_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        mask_windows = self._window_partition(img_mask).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % self.window_size != 0 or w % self.window_size != 0:
            raise ValueError(
                f"Feature map size ({h}, {w}) must be divisible by window_size={self.window_size} for SW-MSA without padding"
            )
        x = x.permute(0, 2, 3, 1).contiguous()

        residual = x
        x = self.norm1(x)
        attn_mask = None
        if self.shift_size > 0:
            attn_mask = self._build_shift_attention_mask(h, w, x.device)
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = self._window_partition(x)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(b, 1, 1)
        windows = self.attn(windows, attn_mask=attn_mask)
        x = self._window_reverse(windows, b, h, w, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2).contiguous()

class MaeMultiscaleDecoder(nn.Module):
    def __init__(self, config: MaeMultiscaleDecoderConfig):
        super().__init__()
        self.type = "MaeMultiscaleDecoder"
        self.config = config

        patch_grid = int(config.num_patches ** 0.5)
        if patch_grid * patch_grid != config.num_patches:
            raise ValueError("num_patches must be a perfect square for multiscale decoding")
        self.patch_grid = patch_grid

        self.input_projections = nn.ModuleDict(
            {
                scale: nn.Conv2d(channels, config.embed_dim, kernel_size=1)
                for scale, channels in config.feature_dims.items()
            }
        )
        self.blocks = nn.ModuleList(
            [
                _SwinDecoderBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    window_size=config.window_size,
                    shift_size=0 if i % 2 == 0 else config.window_size // 2,
                )
                for i in range(config.num_layers)
            ]
        )
        self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches, config.embed_dim) * 0.02)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.pred = nn.Linear(config.embed_dim, config.patch_size ** 2 * config.in_channels)

    def _resize_to_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=(self.patch_grid, self.patch_grid), mode="bilinear", align_corners=False)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decode multiscale feature maps to patch reconstructions.

        Args:
            x: Dict of backbone feature maps (NCHW)
        Returns:
            Reconstructed patches [B, num_patches, patch_size**2 * C]
        """
        deepest_scale = self.config.feature_scales[-1]
        if deepest_scale not in x:
            raise KeyError(f"Missing deepest decoder feature scale '{deepest_scale}'")

        decoded = self._resize_to_patch_grid(self.input_projections[deepest_scale](x[deepest_scale]))

        if self.config.use_skip_connections:
            for scale in self.config.skip_scales:
                if scale in x and scale in self.input_projections:
                    decoded = decoded + self._resize_to_patch_grid(self.input_projections[scale](x[scale]))

        for block in self.blocks:
            decoded = block(decoded)

        tokens = decoded.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens + self.pos_embed)
        return self.pred(tokens)
