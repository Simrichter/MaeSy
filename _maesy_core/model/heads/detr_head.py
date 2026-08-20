import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple

from _maesy_core.model.components import Utils


@dataclass
class DETRHeadConfig:
    """Configuration for Detection Head."""

    type = "DETRHead"

    feature_stage: str = "c5"
    feature_channels: int = 1024
    embed_dim: int = 128
    spatial_feature_size: Tuple[int] = (7,7)
    num_classes: int = 80
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    hidden_dim_out_layers: int = 256

    # Encoder parameters
    num_heads_encoder: int = 8
    num_encoder_layers: int = 6


    num_queries: int = 100
    num_decoder_layers: int = 6
    num_heads_decoder: int = 8
    enable_auxiliary_losses: bool = True

    # def __post_init__(self):
    #     self.num_patches = self.spatial_feature_size[0] * self.spatial_feature_size[1]


class DETRHead(nn.Module):
    """Detection head for object detection using transformer decoder.

    This head takes encoded features from a backbone and produces
    bounding box predictions and class logits using a transformer decoder
    with learned object queries.
    """

    def __init__(self, config: DETRHeadConfig):
        super().__init__()
        self.config = config

        # initial projection to connect backbone output to head input
        self.input_projection = torch.nn.Conv2d(self.config.feature_channels, config.embed_dim, kernel_size=1)

        # self.register_buffer('pos_embed', )
        pos_embed = Utils.get_2d_sinusoidal_encoding(*self.config.spatial_feature_size, self.config.embed_dim)

        self.encoder = DetrEncoder(pos_embed, config.embed_dim, config.num_heads_encoder, config.num_encoder_layers, config.mlp_ratio, config.dropout)
        self.decoder = DetrDecoder(pos_embed, config.embed_dim, config.num_queries, config.num_heads_decoder, config.num_decoder_layers, config.mlp_ratio, config.dropout)

        # Classification head
        self.class_embed = MLP(config.embed_dim, config.hidden_dim_out_layers, config.num_classes + 1, config.dropout)

        # Bounding box regression head
        self.bbox_embed = MLP(config.embed_dim, config.hidden_dim_out_layers, 4, config.dropout)

    def forward(self, features: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through detection head.

        Args:
            features: Encoded features from backbone [B, H'*W', embed_dim]

        Returns:
            Dictionary containing:
                - pred_logits: Class predictions [B, num_queries, num_classes + 1]
                - pred_boxes: Bounding box predictions [B, num_queries, 4]
        """

        features = self.input_projection(features[self.config.feature_stage])  # [B, feature_dim, H', W'] -> [B, embed_dim, H', W']
        _, _, h, w = features.shape
        features = features.flatten(2).transpose(1, 2)  # [B, embed_dim, H', W'] -> [B, embed_dim, H'*W'] -> [B, H'*W', embed_dim]

        encoder_out = self.encoder(features)  # [B, H'*W', D]
        decoder_out, decoder_intermediates = self.decoder(
            encoder_out,
            return_intermediate=self.config.enable_auxiliary_losses,
        )

        # Predict classes and bounding boxes
        pred_logits = self.class_embed(decoder_out)  # [B, num_queries, num_classes + 1]
        pred_boxes = self.bbox_embed(decoder_out).sigmoid()  # [B, num_queries, 4] normalized to [0, 1]

        outputs = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }

        if self.config.enable_auxiliary_losses and len(decoder_intermediates) > 1:
            aux_outputs: List[Dict[str, torch.Tensor]] = []
            for hidden_state in decoder_intermediates[:-1]:
                aux_outputs.append({
                    'pred_logits': self.class_embed(hidden_state),
                    'pred_boxes': self.bbox_embed(hidden_state).sigmoid(),
                })
            outputs['aux_outputs'] = aux_outputs

        return outputs

class DetrEncoder(nn.Module):
    """Transformer encoder for DETR."""
    def __init__(self, pos_embed, embed_dim, num_heads, num_encoder_layers, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        self.register_buffer('pos_embed', pos_embed)

        self.encoder_blocks = nn.ModuleList([
            DetrEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_encoder_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for block in self.encoder_blocks:
            x = block(x, self.pos_embed.expand(x.shape[0], -1, -1))
        return x

class DetrDecoder(nn.Module):
    """Transformer decoder for DETR."""
    def __init__(self, pos_embed, embed_dim, num_queries, num_heads, num_decoder_layers, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embed', pos_embed)

        self.decoder_blocks = nn.ModuleList([
            DetrDecoderLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_decoder_layers)
        ])

    def forward(self, features: torch.Tensor, return_intermediate: bool = False) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass."""
        query_pos_enc = self.query_embed.weight.unsqueeze(0)

        queries = self.query_dropout(self.query_embed.weight).unsqueeze(0).repeat(features.shape[0], 1, 1)

        # Add query noise during training to prevent query collapse
        if self.training:
            noise = torch.randn_like(query_pos_enc) * 0.1
            query_pos_enc = query_pos_enc + noise

        intermediate_outputs: List[torch.Tensor] = []
        for block in self.decoder_blocks:
            queries = block(queries, features, query_pos_enc, self.pos_embed.expand(features.shape[0], -1, -1))
            if return_intermediate:
                intermediate_outputs.append(queries)

        return queries, intermediate_outputs

class DetrEncoderLayer(nn.Module):
    """Single layer of the DETR encoder."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.self_attn = DetrSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, pos_enc) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        attn_out = self.self_attn(self.norm1(x), pos_enc)
        x = x + attn_out

        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x

class DetrDecoderLayer(nn.Module):
    """Single layer of the DETR decoder."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.self_attn = DetrSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = DetrCrossAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, enc_features: torch.Tensor, pos_enc_query: torch.Tensor, pos_enc_feature: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        attn_out = self.self_attn(self.norm1(queries), pos_enc_query)
        queries = queries + attn_out

        # Cross-attention
        cross_attn_out = self.cross_attn(self.norm2(queries), enc_features, pos_enc_query, pos_enc_feature)
        queries = queries + cross_attn_out

        # MLP
        mlp_out = self.mlp(self.norm3(queries))
        queries = queries + mlp_out

        return queries

class DetrSelfAttention(nn.Module):
    """Multi-head self-attention layer for DETR."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) % num_heads ({num_heads}) is not zero ({embed_dim % num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qk = nn.Linear(embed_dim, embed_dim * 2)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.attention = Attention(embed_dim, self.scale, dropout)

    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, N, C = x.shape

        # Compute Q, K, V
        qk = self.qk(x+pos_enc).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [2, B, num_heads, N, head_dim]
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N, head_dim]
        q, k, v = qk[0], qk[1], v

        x = self.attention(q, k, v)

        return x

class DetrCrossAttention(nn.Module):
    """Multi-head cross-attention layer for DETR."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) % num_heads ({num_heads}) is not zero ({embed_dim % num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.attention = Attention(embed_dim, self.scale, dropout)

    def forward(self, queries: torch.Tensor, enc_features: torch.Tensor, pos_enc_query: torch.Tensor, pos_enc_feature: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, N, C = queries.shape

        # Compute Q, K, V
        q = self.q(queries + pos_enc_query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, nq, head_dim]
        k = self.k(enc_features + pos_enc_feature).reshape(B, enc_features.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, np, head_dim]
        v = self.v(enc_features).reshape(B, enc_features.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]

        x = self.attention(q, k, v)

        return x

class Attention(nn.Module):
    """Multi-head attention layer."""
    def __init__(self, embed_dim, scale, dropout: float = 0.1):
        super().__init__()

        self.scale = scale

        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, nh, N, C = q.shape

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, nh*C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class MLP(nn.Module):
    """Simple MLP."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(x)