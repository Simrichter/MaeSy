from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn

from maesy.model.components import Utils, FusionBlock
from maesy.model_tools.debug_helpers import check_finite


@dataclass
class RTDETRHeadConfig:
    feature_channels: Tuple[int, ...]
    num_classes: int = 80
    num_queries: int = 100
    embed_dim: int = 256
    num_decoder_layers: int = 6
    decoder_num_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1
    hidden_dim_out_layers: int = 256
    hidden_dim_dense_heads: int = 256
    num_feature_levels: int = 3
    num_deformable_points: int = 4
    enable_denoising: bool = False
    denoising_num_queries: int = 0
    denoising_label_noise_ratio: float = 0.0 # TODO: Maybe activate? Maybe not
    denoising_box_noise_scale: float = 0.4
    enable_line_detection: bool = False
    line_class_id: int = -1
    enable_ellipse_detection: bool = False
    ellipse_class_id: int = -1
    enable_auxiliary_losses: bool = True
    lightweight_fusion: bool = True
    num_rep_blocks_in_fusion: int = 3
    debug = False



class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, custom_init = False, last_activation: str = "None"):
        super().__init__()
        str_to_act = {"None": nn.Identity(), "ReLU": nn.ReLU(), "Sigmoid": nn.Sigmoid(), "Tanh": nn.Tanh()}
        layers: List[nn.Module] = []
        in_dim = input_dim
        for idx in range(num_layers - 1):
            lin_layer = nn.Linear(in_dim, hidden_dim)
            if custom_init: # TODO: Check if this is useful
                torch.nn.init.normal_(lin_layer.weight, std=0.01)
                torch.nn.init.constant_(lin_layer.bias, 0)
            layers.append(lin_layer)
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(str_to_act.get(last_activation, nn.Identity()))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MaskedAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
            An implementation of Multi-Head Attention.
            Since RT-DETR only uses it for self-attention, some simplifications are done
            Supports nested tensors (preferred) or a mask for padded tensors.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        # self.k_proj = nn.Linear(embed_dim, embed_dim)
        # self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.packed_proj = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.attn_dropout = nn.Dropout(dropout)
        # self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # b, nq, _ = query.shape

        # assumes: query is key and key is value (Self-Attention only)
        assert query is key and key is value, "Error: MaskedAttention module currently only supports Self-Attention"
        result = self.packed_proj(query)
        query, key, value = torch.chunk(result, 3, dim=-1)
        q = query.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1,2)
        k = key.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1,2)
        v = value.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1,2)

        # with torch.amp.autocast("cuda", enabled=False):
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        attn_output = attn_output.transpose(1,2).flatten(-2) # Shape: (N, nq, embed_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # check_finite("attn before mask", attn)
        # if mask is not None:
        #     attn = attn.masked_fill(mask[:, None, :, :], float("-inf"))
        # attn = attn.softmax(dim=-1)
        check_finite("attn after sdpa", attn_output)
        # attn = self.attn_dropout(attn)

        # out = (attn @ v).transpose(1, 2).reshape(b, nq, self.embed_dim)
        out = self.out_proj(attn_output)
        return out


class AIFIBlock(nn.Module):
    """Single-scale attention block used on the highest-level feature map."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MaskedAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x + pos_embed)
        x = x + self.self_attn(normed, normed, normed)
        x = x + self.ffn(self.norm2(x))
        return x


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable attention"""

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        b, num_queries, _ = query.shape
        _, value_len, _ = value.shape

        value = self.value_proj(value).reshape(b, value_len, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).reshape(b, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(b, num_queries, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(dim=-1).reshape(b, num_queries, self.num_heads, self.num_levels, self.num_points)
        check_finite("Deformable attn weights", attention_weights)
        normalizer = torch.tensor(
            [[w, h] for h, w in spatial_shapes],
            dtype=query.dtype,
            device=query.device,
        ).view(1, 1, 1, self.num_levels, 1, 2)
        # sampling_locations = reference_points[:, :, None, None, None, :2] + sampling_offsets / normalizer # TODO: make this more elegant! (move the :2 decision to earlier place)
        assert reference_points.shape[-1] == 2, "To many coordinates for 2D reference_points"
        sampling_locations = reference_points[:, :, None, None, None, :] + sampling_offsets / normalizer # dividing by height/width of the respective feature maps to transform to make offsets resolution-independent
        check_finite("deformable sampling locs", sampling_locations)
        output = torch.zeros(
            b,
            num_queries,
            self.num_heads,
            self.head_dim,
            dtype=query.dtype,
            device=query.device,
        )

        start_idx = 0
        for level_idx, (height, width) in enumerate(spatial_shapes):
            hw = height * width
            value_level = value[:, start_idx:start_idx + hw]
            start_idx += hw

            value_level = value_level.reshape(b, height, width, self.num_heads, self.head_dim)
            value_level = value_level.permute(0, 3, 4, 1, 2).reshape(b * self.num_heads, self.head_dim, height, width)

            grid = sampling_locations[:, :, :, level_idx]
            grid = (grid * 2.0 - 1.0).permute(0, 2, 1, 3, 4).reshape(b * self.num_heads, num_queries, self.num_points, 2)

            sampled = F.grid_sample(
                value_level,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(b, self.num_heads, self.head_dim, num_queries, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)

            level_weights = attention_weights[:, :, :, level_idx].unsqueeze(-1)
            output = output + (sampled * level_weights).sum(dim=3)

        output = output.reshape(b, num_queries, self.embed_dim)
        check_finite("deformable output", output)
        return self.output_proj(output)


class RTDETRDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float, num_levels: int, num_points: int):
        super().__init__()
        self.self_attn = MaskedAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        query_pos: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed_query = self.norm1(query + query_pos)
        query = query + self.self_attn(normed_query, normed_query, normed_query, mask=attention_mask)

        normed_query = self.norm2(query + query_pos)
        with torch.amp.autocast("cuda", enabled=False):
            query = query + self.cross_attn(
                query=normed_query,
                value=memory,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
            )

        query = query + self.ffn(self.norm3(query))
        return query


class RTDETRHead(nn.Module):
    """Hybrid encoder + deformable decoder head"""

    def create_class_heads(self):
        """
        Create classification heads for each decoder layer.
        This is separated from __init__ to allow for creating new classification heads after loading pretrained weights
        """
        self.decoder_class_heads = nn.ModuleList(
            # [nn.Linear(self.config.embed_dim, self.config.num_classes + 1) for _ in range(self.config.num_decoder_layers)]
            [MLP(self.config.embed_dim, self.config.hidden_dim_out_layers, self.config.num_classes+1, custom_init=True) for _ in range(self.config.num_decoder_layers)]
        )

        self.encoder_class_head = MLP(self.config.embed_dim, self.config.hidden_dim_dense_heads, self.config.num_classes + 1) # dense head for query selection
        # self.encoder_class_head = nn.Linear(self.config.embed_dim, self.config.num_classes + 1)

    def __init__(self, config: RTDETRHeadConfig):
        super().__init__()
        self.encoder_class_head: MLP | None = None # Forward-definition with None. Is initialized in create_class_heads function
        self.decoder_class_heads: nn.ModuleList | None = None
        self.type = "RTDETRHead"
        self.config = config
        if len(config.feature_channels) != config.num_feature_levels:
            raise ValueError("feature_channels length must match num_feature_levels")

        self.input_proj = nn.ModuleList([nn.Conv2d(c, config.embed_dim, kernel_size=1) for c in config.feature_channels])
        self.level_embeddings = nn.Parameter(torch.zeros(config.num_feature_levels, config.embed_dim))

        # Cross-scale fusion (top-down + bottom-up)
        if self.config.lightweight_fusion:
            self.fpn_td_4 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)
            self.fpn_td_3 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)
            self.pan_down_3 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, stride=2, padding=1)
            self.pan_down_4 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, stride=2, padding=1)
            self.pan_out_4 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)
            self.pan_out_5 = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)
        else:
            self.before_upsample1 = nn.ModuleList([
                nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(config.embed_dim),
                nn.SiLU()
            ])
            self.before_upsample2 = nn.ModuleList([
                nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(config.embed_dim),
                nn.SiLU()
            ])
            self.fusion1 = FusionBlock(self.config.num_rep_blocks_in_fusion, self.config.embed_dim)
            self.fusion2 = FusionBlock(self.config.num_rep_blocks_in_fusion, self.config.embed_dim)
            self.fusion3 = FusionBlock(self.config.num_rep_blocks_in_fusion, self.config.embed_dim)
            self.fusion4 = FusionBlock(self.config.num_rep_blocks_in_fusion, self.config.embed_dim)
            self.downsample_conv1 = nn.ModuleList([
                nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(config.embed_dim),
                nn.SiLU()
            ])
            self.downsample_conv2 = nn.ModuleList([
                nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(config.embed_dim),
                nn.SiLU()
            ])

        self.aifi = AIFIBlock(
            embed_dim=config.embed_dim,
            num_heads=config.decoder_num_heads,
            mlp_ratio=config.decoder_mlp_ratio,
            dropout=config.decoder_dropout,
        )

        # self.encoder_class_head = MLP(config.embed_dim, config.hidden_dim_dense_heads, config.num_classes+1)# nn.Linear(config.embed_dim, config.num_classes + 1) # dense head for query selection
        # encoder_class_head is instantiated in create_class_heads() method!
        self.encoder_box_head = MLP(config.embed_dim, config.hidden_dim_dense_heads, 4) # dense head for reference boxes
        self.encoder_line_head = MLP(config.embed_dim, config.hidden_dim_dense_heads, 4) # dense head for reference lines
        self.encoder_ellipse_head = MLP(config.embed_dim, config.hidden_dim_dense_heads, 6) # dense head for reference ellipses

        self.query_content = nn.Embedding(config.num_queries, config.embed_dim)
        self.reference_point_proj = MLP(2, config.embed_dim, config.embed_dim, num_layers=2)
        self.decoder_layers = nn.ModuleList(
            [
                RTDETRDecoderLayer(
                    embed_dim=config.embed_dim,
                    num_heads=config.decoder_num_heads,
                    mlp_ratio=config.decoder_mlp_ratio,
                    dropout=config.decoder_dropout,
                    num_levels=config.num_feature_levels,
                    num_points=config.num_deformable_points,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

        self.create_class_heads()

        self.decoder_box_heads = nn.ModuleList(
            [MLP(config.embed_dim, config.hidden_dim_out_layers, 4) for _ in range(config.num_decoder_layers)]
        )

        self.decoder_line_heads = nn.ModuleList([MLP(config.embed_dim, config.hidden_dim_out_layers, 4) for _ in range(config.num_decoder_layers)]) if config.enable_line_detection else None

        self.decoder_ellipse_heads = nn.ModuleList([MLP(config.embed_dim, config.hidden_dim_out_layers, 6) for _ in range(config.num_decoder_layers)]) if config.enable_ellipse_detection else None

        if config.enable_denoising and config.denoising_num_queries > 0:
            self.dn_query_embedding = nn.Embedding(config.num_classes+1, config.embed_dim)
        else:
            self.dn_query_embedding = None

        self._pos_encoding_cache: Dict[Tuple[int, int, str, int, str], torch.Tensor] = {}

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x) - torch.log(1.0 - x) # Division-safe version of torch.log(x / (1.0 - x))

    def _build_positional_encoding(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        cache_key = (h, w, feat.device.type, feat.device.index or -1, str(feat.dtype))

        if torch.onnx.is_in_onnx_export():
            pos = Utils.get_2d_sinusoidal_encoding(h, w, c, device=feat.device).to(dtype=feat.dtype)
        else:
            pos = self._pos_encoding_cache.get(cache_key)
            if pos is None:
                pos = Utils.get_2d_sinusoidal_encoding(h, w, c, device=feat.device).to(dtype=feat.dtype)
                self._pos_encoding_cache[cache_key] = pos
        return pos.expand(b, -1, -1)

    @staticmethod
    def _ordered_feature_keys(features: Dict[str, torch.Tensor]) -> List[str]:
        preferred = ["c3", "c4", "c5", "c6"]
        ordered = [key for key in preferred if key in features]
        if len(ordered) == len(features):
            return ordered
        return sorted(features.keys())

    def _hybrid_encode(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        ordered_keys = self._ordered_feature_keys(features)
        if len(ordered_keys) < self.config.num_feature_levels:
            raise ValueError(
                f"Expected at least {self.config.num_feature_levels} feature levels, got {len(ordered_keys)}"
            )
        projected = [proj(features[key]) for proj, key in zip(self.input_proj, ordered_keys)]
        p3, p4, p5 = projected

        b, c, h, w = p5.shape
        p5_tokens = p5.flatten(2).transpose(1, 2)
        p5_tokens = self.aifi(p5_tokens, self._build_positional_encoding(p5))
        p5 = p5_tokens.transpose(1, 2).reshape(b, c, h, w)

        if self.config.lightweight_fusion:
            p4 = self.fpn_td_4(p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
            p3 = self.fpn_td_3(p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))

            n4 = self.pan_out_4(p4 + self.pan_down_3(p3))
            n5 = self.pan_out_5(p5 + self.pan_down_4(n4))
        else:
            p4 = self.fusion1(torch.cat((F.interpolate(self.before_upsample1(p5), size=p4.shape[-2:], mode="nearest"), p4)))
            p3 = self.fusion2(torch.cat((F.interpolate(self.before_upsample2(p4), size=p3.shape[-2:], mode="nearest"), p3)))

            n4 = self.fusion3(torch.cat((self.downsample_conv1(p3), p4)))
            n5 = self.fusion4(torch.cat((self.downsample_conv2(n4), p5)))
        return [p3, n4, n5]

    def _flatten_memory(self, fused_features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        tokens: List[torch.Tensor] = []
        spatial_shapes: List[Tuple[int, int]] = []

        for level_idx, feat in enumerate(fused_features):
            _, _, h, w = feat.shape
            spatial_shapes.append((h, w))

            token = feat.flatten(2).transpose(1, 2)
            token = token + self._build_positional_encoding(feat)
            token = token + self.level_embeddings[level_idx].view(1, 1, -1)
            tokens.append(token)

        return torch.cat(tokens, dim=1), spatial_shapes

    def _select_queries(self, memory: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        references: Dict[str, torch.Tensor] = {}
        enc_outputs: Dict[str, torch.Tensor] = {}
        enc_logits = self.encoder_class_head(memory)

        foreground_scores = enc_logits[..., :-1].sigmoid().amax(dim=-1)
        topk_idx = torch.topk(foreground_scores, k=self.config.num_queries, dim=1).indices.unsqueeze(-1)

        gather_idx = topk_idx.expand(-1, -1, memory.shape[-1])
        selected_memory = torch.gather(memory, dim=1, index=gather_idx)
        enc_outputs["pred_logits"] = torch.gather(
            enc_logits,
            dim=1,
            index=topk_idx.expand(-1, -1, enc_logits.shape[-1]),
        )

        references["pred_logits"] = enc_outputs["pred_logits"]

        selected_refs = topk_idx.expand(-1, -1, 4)
        enc_box_logits = self.encoder_box_head(memory)
        enc_box_normed = enc_box_logits.sigmoid()
        # weigh based on the remaining space to ensure [0, 1], interpret the result as width and height from top left corner
        enc_boxes = torch.cat((enc_box_normed[..., 0:2], enc_box_normed[..., 0:2] + enc_box_normed[..., 2:4] * (1.0- enc_box_normed[..., 0:2])), dim=-1)

        assert torch.isfinite(enc_boxes).all()
        references["reference_box_logits"] = torch.gather(enc_box_logits, dim=1, index=selected_refs)
        references["reference_boxes"] = torch.gather(enc_boxes, dim=1, index=selected_refs)
        enc_outputs["pred_boxes"] = references["reference_boxes"]
        if self.config.enable_line_detection:
            enc_line_logits = self.encoder_line_head(memory)
            enc_lines = enc_line_logits.sigmoid()
            references["reference_line_logits"] = torch.gather(enc_line_logits, dim=1, index=selected_refs)
            references["reference_lines"] = torch.gather(enc_lines, dim=1, index=selected_refs)
            enc_outputs["pred_lines"] = references["reference_lines"]
        if self.config.enable_ellipse_detection:
            enc_ellipse_logits = self.encoder_ellipse_head(memory)
            # enc_ellipses[:, :, -2:] = F.tanh(enc_ellipses[:, :, -2:])
            enc_ellipses = enc_ellipse_logits.copy()
            enc_ellipses[:, :, :4] = enc_ellipses[:, :, :4].sigmoid()
            references["reference_ellipse_logits"] = torch.gather(enc_ellipse_logits, dim=1, index=topk_idx.expand(-1, -1, 6))
            references["reference_ellipses"] = torch.gather(enc_ellipses, dim=1, index=topk_idx.expand(-1, -1, 6))
            enc_outputs["pred_ellipses"] = references["reference_ellipses"]

        enc_outputs["selected_indices"] = topk_idx.squeeze(-1)

        query = selected_memory + self.query_content.weight.unsqueeze(0)
        return query, references, enc_outputs

    def _build_denoising_queries(
        self,
        targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype, # TODO: Why use dtype inconsistently? i.e. dtype=torch.long / dtype=dtype
    ) -> Dict[str, torch.Tensor] | None:
        batch_size = len(targets)
        nq_boxes: int = max(len(t.get("boxes", []))for t in targets) # use of generator expression avoids costly creation of intermediate list
        nq_lines: int = max(len(t.get("line_points", [])) for t in targets)
        nq_ellipses: int = max(len(t.get("ellipses",[])) for t in targets)
        total_nq = nq_boxes+nq_lines+nq_ellipses

        dn_ref_box_logits = torch.zeros(batch_size, total_nq, 4, device=device, dtype=dtype)
        dn_ref_boxes = torch.full((batch_size, total_nq, 4), 0.5, device=device, dtype=dtype) # Use 0.5 to now blow up inverse sigmoid. Should not matter anyways due to masking
        dn_tgt_boxes = torch.zeros(batch_size, total_nq, 4, device=device, dtype=dtype)
        dn_box_mask = torch.full((batch_size, total_nq), False, device=device, dtype=torch.bool)

        dn_ref_line_logits = torch.zeros(batch_size, total_nq, 4, device=device, dtype=dtype)
        dn_ref_lines = torch.full((batch_size, total_nq, 4), 0.5, device=device, dtype=dtype)
        dn_tgt_lines = torch.zeros(batch_size, total_nq, 4, device=device, dtype=dtype)
        dn_line_mask = torch.full((batch_size, total_nq), False, device=device, dtype=torch.bool)

        dn_masking = torch.full((batch_size, total_nq, total_nq), fill_value=True, device=device, dtype=torch.bool)

        dn_labels = torch.full((batch_size, total_nq), fill_value=self.config.num_classes, device=device, dtype=torch.long,)
        dn_class_logits = torch.full((batch_size, total_nq, self.config.num_classes + 1), float("-inf"), device=device, dtype=torch.float)
        dn_valid = torch.zeros(batch_size, total_nq, device=device, dtype=torch.bool) # mask used to ignore padding entries in self-attention

        for batch_idx, target in enumerate(targets):
            boxes = target.get("boxes", torch.empty((0, 4), device=device, dtype=dtype))
            labels = target.get("labels", torch.empty((0,), device=device, dtype=torch.long))
            lines = target.get("line_points", torch.empty((0, 4), device=device, dtype=torch.float))
            ellipses = target.get("ellipses", torch.empty((0, 6), device=device, dtype=dtype))

            box_noise = (torch.rand_like(boxes) * 2.0 - 1.0) * self.config.denoising_box_noise_scale
            sel_boxes = (boxes + box_noise).clamp(0.0, 1.0)
            assert boxes.shape[0] <= nq_boxes
            boxes_end = sel_boxes.shape[0]
            dn_ref_boxes[batch_idx, :boxes_end, :] = sel_boxes
            dn_ref_box_logits[batch_idx, :boxes_end, :] = self._inverse_sigmoid(sel_boxes)
            dn_tgt_boxes[batch_idx, :boxes_end, :] = boxes
            dn_box_mask[batch_idx, :boxes_end] = True
            # dn_masking[batch_idx, :boxes_end, :boxes.shape[0]] = False # Set to False to allow attention
            dn_valid[batch_idx, :boxes_end] = True

            line_noise = (torch.rand_like(lines) * 2.0 - 1.0) * self.config.denoising_box_noise_scale # TODO: Make parameter distinct for lines + ellipses
            sel_lines = (lines + line_noise).clamp(0.0, 1.0)
            assert lines.shape[0] <= nq_lines
            lines_end = boxes_end+sel_lines.shape[0]
            dn_ref_lines[batch_idx, boxes_end:lines_end, :] = sel_lines
            dn_ref_line_logits[batch_idx, boxes_end:lines_end, :] = self._inverse_sigmoid(sel_lines)
            dn_tgt_lines[batch_idx, boxes_end:lines_end, :] = lines
            dn_line_mask[batch_idx, boxes_end:lines_end] = True
            # dn_masking[batch_idx, boxes_end:lines_end, boxes_end:boxes_end+lines_end] = False # Set to False to allow attention
            dn_valid[batch_idx, boxes_end:lines_end] = True

            if self.config.denoising_label_noise_ratio > 0:
                noise_mask = torch.rand_like(labels.float()) < self.config.denoising_label_noise_ratio
                random_labels = torch.randint(0, self.config.num_classes, size=labels.shape, device=device)
                labels = torch.where(noise_mask, random_labels, labels)

            assert len(labels)<=total_nq
            dn_labels[batch_idx, :lines_end] = labels
            dn_class_logits[batch_idx, torch.arange(len(labels)), labels] = float("inf")
        dn_masking = ~dn_valid[:, None, :].expand(
            batch_size,
            total_nq,
            total_nq
        )
        dn_queries = self.dn_query_embedding(dn_labels)
        return {
            "dn_query": dn_queries,
            "reference_boxes": dn_ref_boxes,
            "reference_box_logits": dn_ref_box_logits,
            "target_boxes": dn_tgt_boxes,
            "box_mask": dn_box_mask,
            "reference_lines": dn_ref_lines,
            "reference_line_logits": dn_ref_line_logits,
            "target_lines": dn_tgt_lines,
            "line_mask": dn_line_mask,
            "dn_labels": dn_labels,
            "pred_logits": dn_class_logits,
            "dn_valid": dn_valid,
            "dn_self_attention_mask": dn_masking
        }

    def _decoder_stack(
        self,
        query: torch.Tensor,
        references: Dict[str, torch.Tensor],
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        class_logits = references["pred_logits"].sigmoid()
        reference_boxes = references.get("reference_boxes", torch.zeros(*class_logits.shape[:2], 4, device=class_logits.device)) #.detach() # Detach from gradient graph due to instability issues.
        # reference_box_mask = references.get("box_mask", torch.full(reference_boxes.shape[:2], True, device=class_logits.device, dtype=torch.bool))
        reference_lines = references.get("reference_lines", torch.zeros(*class_logits.shape[:2], 4, device=class_logits.device))
        # reference_line_mask = references.get("line_mask", torch.full(reference_lines.shape[:2], True, device=class_logits.device, dtype=torch.bool))
        reference_ellipses = references.get("reference_ellipses", torch.zeros(*class_logits.shape[:2], 6, device=class_logits.device))
        # reference_ellipse_mask = references.get("ellipse_mask", torch.full(reference_ellipses.shape[:2], True, device=class_logits.device, dtype=torch.bool))

        reference_box_logits = references.get("reference_box_logits") # self._inverse_sigmoid(reference_boxes) #TODO: Cleanup if this works
        line_reference_logits = references.get("reference_line_logits") # self._inverse_sigmoid(references.get("reference_lines")) # self._inverse_sigmoid(reference_boxes)
        ellipse_reference_logits = references.get("reference_ellipse_logits") # self._inverse_sigmoid(references.get("reference_ellipses"))

        logits_per_layer: List[torch.Tensor] = []
        boxes_per_layer: List[torch.Tensor] = []
        lines_per_layer: List[torch.Tensor] = []
        ellipses_per_layer: List[torch.Tensor] = []

        assert self.decoder_class_heads is not None, "Problem in class_head initialization detected. Decoder_class_heads are still uninitialized."
        for layer_idx, (layer, cls_head, box_head) in enumerate(zip(self.decoder_layers, self.decoder_class_heads, self.decoder_box_heads)):

            # # build a weighting system to combine reference points from the three geometries
            box_classes = [
                c for c in range(class_logits.shape[-1])
                if c not in {
                    self.config.line_class_id,
                    self.config.ellipse_class_id,
                }
            ]
            box_logit = (class_logits[:, :, box_classes].max(dim=-1)[0]).unsqueeze(-1)
            line_logit = class_logits[:, :, self.config.line_class_id].unsqueeze(-1) if self.config.line_class_id != -1 else torch.full((*class_logits.shape[:2], 1), 0.0, device=class_logits.device)
            ellipse_logit = class_logits[:, :, self.config.ellipse_class_id].unsqueeze(-1) if self.config.ellipse_class_id != -1 else torch.full((*class_logits.shape[:2], 1), 0.0, device=class_logits.device)

            weighting = torch.cat([box_logit, line_logit, ellipse_logit], dim=-1)
            weighting /= weighting.sum(dim=-1, keepdim=True) + 1e-8 # normalize to sum to 1, add small epsilon to avoid division by zero

            reference_points = weighting[:, :, 0].unsqueeze(-1)*(reference_boxes[..., :2] + reference_boxes[..., 2:4]) / 2
            reference_points += weighting[:, :, 1].unsqueeze(-1)*(reference_lines[..., :2] + reference_lines[..., 2:4])/2
            reference_points += weighting[:, :, 2].unsqueeze(-1)*reference_ellipses[..., :2]
            query_pos_encoding = self.reference_point_proj(reference_points)
            query = layer(
                query=query,
                memory=memory,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                query_pos=query_pos_encoding,
                attention_mask=self_attention_mask,
            )
            pred_logits = cls_head(query)
            # assert torch.isfinite(pred_logits).all(), pred_logits
            box_delta = box_head(query)
            if box_delta is None:
                raise ValueError("box_delta is None")
            if reference_box_logits is None:
                raise ValueError("reference_box_logits is None")
            pred_box_normed = (box_delta + reference_box_logits).sigmoid()
            pred_boxes = torch.cat((pred_box_normed[..., 0:2], pred_box_normed[..., 0:2] + pred_box_normed[..., 2:4] * (1.0- pred_box_normed[..., 0:2])), dim=-1)
            logits_per_layer.append(pred_logits)
            boxes_per_layer.append(pred_boxes)

            if self.decoder_line_heads is not None and line_reference_logits is not None:
                line_delta = self.decoder_line_heads[layer_idx](query)
                pred_lines = (line_delta + line_reference_logits).sigmoid()
                lines_per_layer.append(pred_lines)
                line_reference_logits = line_delta.detach() + line_reference_logits.detach()

            if self.decoder_ellipse_heads is not None and ellipse_reference_logits is not None:
                tmp_ellipse_pred = self.decoder_ellipse_heads[layer_idx](query)
                tmp_ellipse_pred[:, :, -2:] = F.tanh(tmp_ellipse_pred[:, :, -2:])
                pred_ellipses = (tmp_ellipse_pred + ellipse_reference_logits)
                ellipses_per_layer.append(pred_ellipses)
                ellipse_reference_logits = tmp_ellipse_pred.detach() + ellipse_reference_logits.detach()

            reference_box_logits = box_delta.detach() + reference_box_logits.detach()
            reference_boxes = pred_boxes.detach()

        decoded: Dict[str, List[torch.Tensor]] = {
            "pred_logits": logits_per_layer,
            "pred_boxes": boxes_per_layer,
        }
        if len(lines_per_layer) > 0:
            decoded["pred_lines"] = lines_per_layer

        if len(ellipses_per_layer) > 0:
            decoded["pred_ellipses"] = ellipses_per_layer

        return decoded

    def forward(self, features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]]:
        """
            Forward pass of the rt-detr head

            Args:
                :param features: Dict that contains the multiscale features from the backbone. Expects B, C, H, W format. Keys should be in the form of c{level} (e.g. c3, c4, c5)

            Returns:
             outputs: Dict containing raw outputs
        """
        fused = self._hybrid_encode(features)
        memory, spatial_shapes = self._flatten_memory(fused)

        query, references, enc_outputs = self._select_queries(memory)
        main_decoded = self._decoder_stack(query, references, memory, spatial_shapes)
        logits_per_layer = main_decoded["pred_logits"]
        boxes_per_layer = main_decoded["pred_boxes"]
        lines_per_layer = main_decoded.get("pred_lines", [])
        ellipses_per_layer = main_decoded.get("pred_ellipses", [])
        check_finite("logits_per_layer", logits_per_layer)
        check_finite("boxes_per_layer", boxes_per_layer)
        outputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]] = {
            "pred_logits": logits_per_layer[-1],
            "pred_boxes": boxes_per_layer[-1],
            "enc_outputs": enc_outputs,
        }
        if len(lines_per_layer) > 0:
            outputs["pred_lines"] = lines_per_layer[-1]
        if len(ellipses_per_layer) > 0:
            outputs["pred_ellipses"] = ellipses_per_layer[-1]

        if self.config.enable_auxiliary_losses and len(logits_per_layer) > 1:
            aux_outputs: List[Dict[str, torch.Tensor]] = []
            for aux_idx, (cls, box) in enumerate(zip(logits_per_layer[:-1], boxes_per_layer[:-1])): # TODO: Simplify this?? Should just be a copy of the lists??
                aux_output = {"pred_logits": cls, "pred_boxes": box}
                if len(lines_per_layer) > aux_idx:
                    aux_output["pred_lines"] = lines_per_layer[aux_idx]
                if len(ellipses_per_layer) > aux_idx:
                    aux_output["pred_ellipses"] = ellipses_per_layer[aux_idx]
                aux_outputs.append(aux_output)
            outputs["aux_outputs"] = aux_outputs

        targets = kwargs.get("targets")
        if self.training and self.config.enable_denoising and isinstance(targets, list):
            dn_pack = self._build_denoising_queries(
                targets=targets,
                device=memory.device,
                dtype=memory.dtype,
            )
            if dn_pack is not None:
                # dn_references: Dict[str, torch.Tensor] = {"reference_boxes": dn_pack["dn_ref_boxes"], "reference_lines": dn_pack["dn_ref_lines"], "reference_ellipses": dn_pack["dn_ref_ellipses"]}

                dn_decoded = self._decoder_stack(dn_pack["dn_query"], dn_pack, memory, spatial_shapes, self_attention_mask=dn_pack["dn_self_attention_mask"])
                dn_output: Dict[str, torch.Tensor] = {
                    "pred_logits": dn_decoded["pred_logits"][-1],
                    "pred_boxes": dn_decoded["pred_boxes"][-1],
                    "target_labels": dn_pack["dn_labels"],
                    "target_boxes": dn_pack["target_boxes"],
                    "target_lines": dn_pack["target_lines"],
                    "target_valid_mask": dn_pack["dn_valid"],
                }
                if "pred_lines" in dn_decoded:
                    dn_output["pred_lines"] = dn_decoded["pred_lines"][-1]
                    dn_output["target_lines"] = dn_pack["target_lines"]
                outputs["dn_outputs"] = dn_output

        return outputs