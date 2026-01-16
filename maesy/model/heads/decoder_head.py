import torch.nn as nn
from dataclasses import dataclass, asdict
import torch
from ..components import TransformerBlock, Utils

@dataclass
class DecoderHeadConfig:
    embed_dim: int
    num_patches: int
    patch_size: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attention_dropout: float
    num_layers: int
    in_channels: int

class DecoderHead(nn.Module):
    def __init__(self, config: DecoderHeadConfig):
        super().__init__()

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

        Utils.init_weights(self)

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