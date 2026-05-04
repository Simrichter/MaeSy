import torch.nn as nn
from dataclasses import dataclass, asdict
import torch
from ..components import TransformerBlock, Utils


@dataclass
class MaeDecoderHeadConfig:
    embed_dim: int
    num_patches: int
    patch_size: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attention_dropout: float
    num_layers: int
    in_channels: int


class MaeDecoderHead(nn.Module):
    def __init__(self, config: MaeDecoderHeadConfig):
        super().__init__()
        self.type = "DecoderHead"
        self.config = config
        # Decoder components
        self.decoder_embed = nn.Linear(config.embed_dim, config.embed_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        # Positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, config.num_patches + 1, config.embed_dim) * 0.02)

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(**asdict(config)) for _ in range(config.num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)

        # Prediction head - reconstruct pixels
        self.decoder_pred = nn.Linear(config.embed_dim, config.patch_size ** 2 * config.in_channels)

        # Utils.init_weights(self)

    def _init_weights(self):
        """Initialize weights."""
        # Initialize patch embedding like nn.Linear
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize other parameters
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, ids_shuffle: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            x: Encoded visible patches [B, N_visible + 1, D]
            ids_shuffle: Indices that were used to shuffle the patches [B, num_patches]

        Returns:
            x: Reconstructed patches [B, num_patches, patch_size**2 * C]
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add positional encoding
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predict pixel values
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x
