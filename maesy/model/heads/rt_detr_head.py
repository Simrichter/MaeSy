from dataclasses import dataclass
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from torch import nn

from maesy.model.components import Utils


@dataclass
class RTDETRHeadConfig:
    feature_channels: Tuple[int, int, int]
    num_classes: int = 80
    num_queries: int = 100
    embed_dim: int = 256
    num_decoder_layers: int = 6
    decoder_num_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1
    hidden_dim_out_layers: int = 256
    enable_auxiliary_losses: bool = True


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AIFIBlock(nn.Module):
    """Single-scale attention block used on the highest-level feature map."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        return self.block(x + pos_embed)


class RTDETRDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        q = self.norm1(query + query_pos)
        self_attn_out, _ = self.self_attn(q, q, q)
        query = query + self_attn_out

        q = self.norm2(query + query_pos)
        cross_out, _ = self.cross_attn(q, memory, memory)
        query = query + cross_out

        query = query + self.ffn(self.norm3(query))
        return query


class RTDETRHead(nn.Module):
    """Hybrid-encoder + iterative decoder head inspired by RT-DETR."""

    def __init__(self, config: RTDETRHeadConfig):
        super().__init__()
        self.type = "RTDETRHead"
        self.config = config

        self.input_proj = nn.ModuleList([nn.Conv2d(c, config.embed_dim, kernel_size=1) for c in config.feature_channels])

        # Cross-scale fusion (top-down + bottom-up)
        self.fpn_td_4 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)
        self.fpn_td_3 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)
        self.pan_down_3 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, stride=2, padding=1)
        self.pan_down_4 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, stride=2, padding=1)
        self.pan_out_4 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)
        self.pan_out_5 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)

        self.aifi = AIFIBlock(
            embed_dim=config.embed_dim,
            num_heads=config.decoder_num_heads,
            mlp_ratio=config.decoder_mlp_ratio,
            dropout=config.decoder_dropout,
        )

        self.encoder_class_head = nn.Linear(config.embed_dim, config.num_classes + 1)
        self.encoder_box_head = MLP(config.embed_dim, config.hidden_dim_out_layers, 4)

        self.query_content = nn.Embedding(config.num_queries, config.embed_dim)
        self.decoder_layers = nn.ModuleList(
            [
                RTDETRDecoderLayer(
                    embed_dim=config.embed_dim,
                    num_heads=config.decoder_num_heads,
                    mlp_ratio=config.decoder_mlp_ratio,
                    dropout=config.decoder_dropout,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.decoder_class_heads = nn.ModuleList(
            [nn.Linear(config.embed_dim, config.num_classes + 1) for _ in range(config.num_decoder_layers)]
        )
        self.decoder_box_heads = nn.ModuleList(
            [MLP(config.embed_dim, config.hidden_dim_out_layers, 4) for _ in range(config.num_decoder_layers)]
        )

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))

    @staticmethod
    def _build_positional_encoding(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        pos = Utils.get_2d_sinusoidal_encoding(h, w, c, device=feat.device)
        return pos.expand(b, -1, -1)

    def _hybrid_encode(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = [proj(feat) for proj, feat in zip(self.input_proj, features.values())]

        b, c, h, w = p5.shape
        p5_tokens = p5.flatten(2).transpose(1, 2)
        p5_tokens = self.aifi(p5_tokens, self._build_positional_encoding(p5))
        p5 = p5_tokens.transpose(1, 2).reshape(b, c, h, w)

        p4 = self.fpn_td_4(p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        p3 = self.fpn_td_3(p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))

        n4 = self.pan_out_4(p4 + self.pan_down_3(p3))
        n5 = self.pan_out_5(p5 + self.pan_down_4(n4))
        return [p3, n4, n5]

    def _flatten_memory(self, fused_features: List[torch.Tensor]) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for feat in fused_features:
            token = feat.flatten(2).transpose(1, 2)
            token = token + self._build_positional_encoding(feat)
            tokens.append(token)
        return torch.cat(tokens, dim=1)

    def _select_queries(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_logits = self.encoder_class_head(memory)
        enc_boxes = self.encoder_box_head(memory).sigmoid()

        foreground_scores = enc_logits[..., :-1].sigmoid().amax(dim=-1)
        topk = torch.topk(foreground_scores, k=self.config.num_queries, dim=1)

        gather_idx = topk.indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        selected_memory = torch.gather(memory, dim=1, index=gather_idx)

        box_idx = topk.indices.unsqueeze(-1).expand(-1, -1, 4)
        reference_boxes = torch.gather(enc_boxes, dim=1, index=box_idx)

        query = selected_memory + self.query_content.weight.unsqueeze(0)
        return query, reference_boxes

    def forward(self, features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        fused = self._hybrid_encode(features)
        memory = self._flatten_memory(fused)

        query, reference_boxes = self._select_queries(memory)
        reference_logits = self._inverse_sigmoid(reference_boxes)

        logits_per_layer: List[torch.Tensor] = []
        boxes_per_layer: List[torch.Tensor] = []

        query_pos = self.query_content.weight.unsqueeze(0).expand(query.shape[0], -1, -1)
        for layer, cls_head, box_head in zip(self.decoder_layers, self.decoder_class_heads, self.decoder_box_heads):
            query = layer(query, memory, query_pos)
            pred_logits = cls_head(query)
            pred_boxes = (box_head(query) + reference_logits).sigmoid()
            logits_per_layer.append(pred_logits)
            boxes_per_layer.append(pred_boxes)
            reference_logits = self._inverse_sigmoid(pred_boxes.detach())

        outputs = {
            "pred_logits": logits_per_layer[-1],
            "pred_boxes": boxes_per_layer[-1],
        }

        if self.config.enable_auxiliary_losses and len(logits_per_layer) > 1:
            outputs["aux_outputs"] = [
                {"pred_logits": cls, "pred_boxes": box}
                for cls, box in zip(logits_per_layer[:-1], boxes_per_layer[:-1])
            ]

        return outputs