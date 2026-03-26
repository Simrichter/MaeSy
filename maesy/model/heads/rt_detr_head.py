from dataclasses import dataclass
from typing import Tuple, List, Dict, Union

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
    num_feature_levels: int = 3
    num_deformable_points: int = 4
    enable_denoising: bool = False
    denoising_num_queries: int = 0
    denoising_label_noise_ratio: float = 0.2
    denoising_box_noise_scale: float = 0.4
    enable_line_detection: bool = False
    line_class_id: int = -1
    enable_auxiliary_losses: bool = True


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for idx in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ManualAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        b, nq, _ = query.shape
        nk = key.shape[1]

        q = self.q_proj(query).reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(b, nq, self.embed_dim)
        out = self.proj_dropout(self.out_proj(out))
        return out


class AIFIBlock(nn.Module):
    """Single-scale attention block used on the highest-level feature map."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = ManualAttention(embed_dim, num_heads, dropout)
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
        sampling_offsets = self.sampling_offsets(query).reshape(
            b, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            b, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(dim=-1).reshape(
            b, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        normalizer = torch.tensor(
            [[w, h] for h, w in spatial_shapes],
            dtype=query.dtype,
            device=query.device,
        ).view(1, 1, 1, self.num_levels, 1, 2)
        sampling_locations = reference_points[:, :, None, None, None, :2] + sampling_offsets / normalizer

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
        return self.output_proj(output)


class RTDETRDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float, num_levels: int, num_points: int):
        super().__init__()
        self.self_attn = ManualAttention(embed_dim, num_heads, dropout)
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
    ) -> torch.Tensor:
        normed_query = self.norm1(query + query_pos)
        query = query + self.self_attn(normed_query, normed_query, normed_query)

        normed_query = self.norm2(query + query_pos)
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

    def __init__(self, config: RTDETRHeadConfig):
        super().__init__()
        self.type = "RTDETRHead"
        self.config = config
        if len(config.feature_channels) != config.num_feature_levels:
            raise ValueError("feature_channels length must match num_feature_levels")

        self.input_proj = nn.ModuleList([nn.Conv2d(c, config.embed_dim, kernel_size=1) for c in config.feature_channels])
        self.level_embeddings = nn.Parameter(torch.zeros(config.num_feature_levels, config.embed_dim))

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
        self.decoder_class_heads = nn.ModuleList(
            [nn.Linear(config.embed_dim, config.num_classes + 1) for _ in range(config.num_decoder_layers)]
        )
        self.decoder_box_heads = nn.ModuleList(
            [MLP(config.embed_dim, config.hidden_dim_out_layers, 4) for _ in range(config.num_decoder_layers)]
        )
        if config.enable_line_detection:
            self.decoder_line_heads = nn.ModuleList(
                [MLP(config.embed_dim, config.hidden_dim_out_layers, 4) for _ in range(config.num_decoder_layers)]
            )
        else:
            self.decoder_line_heads = None

        if config.enable_denoising and config.denoising_num_queries > 0:
            self.dn_query_content = nn.Embedding(config.denoising_num_queries, config.embed_dim)
        else:
            self.dn_query_content = None

        self._pos_encoding_cache: Dict[Tuple[int, int, str, int, str], torch.Tensor] = {}

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))

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

        p4 = self.fpn_td_4(p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        p3 = self.fpn_td_3(p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))

        n4 = self.pan_out_4(p4 + self.pan_down_3(p3))
        n5 = self.pan_out_5(p5 + self.pan_down_4(n4))
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

    def _build_denoising_queries(
        self,
        targets: List[Dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor] | None:
        if self.dn_query_content is None:
            return None

        dn_q = self.config.denoising_num_queries
        dn_boxes = torch.zeros(batch_size, dn_q, 4, device=device, dtype=dtype)
        dn_labels = torch.full(
            (batch_size, dn_q),
            fill_value=self.config.num_classes,
            device=device,
            dtype=torch.long,
        )
        dn_valid = torch.zeros(batch_size, dn_q, device=device, dtype=torch.bool)
        dn_lines = torch.zeros(batch_size, dn_q, 4, device=device, dtype=dtype)

        for batch_idx, target in enumerate(targets):
            boxes = target.get("boxes", torch.empty((0, 4), device=device, dtype=dtype)).to(device=device, dtype=dtype)
            labels = target.get("labels", torch.empty((0,), device=device, dtype=torch.long)).to(device=device, dtype=torch.long)
            lines = target.get("line_points", boxes).to(device=device, dtype=dtype)

            if boxes.shape[0] == 0:
                continue

            repeats = (dn_q + boxes.shape[0] - 1) // boxes.shape[0]
            selected_indices = torch.randperm(boxes.shape[0], device=device).repeat(repeats)[:dn_q]

            sel_boxes = boxes[selected_indices]
            sel_labels = labels[selected_indices]
            sel_lines = lines[selected_indices]

            box_noise = (torch.rand_like(sel_boxes) * 2.0 - 1.0) * self.config.denoising_box_noise_scale
            sel_boxes = (sel_boxes + box_noise).clamp(0.0, 1.0)

            line_noise = (torch.rand_like(sel_lines) * 2.0 - 1.0) * self.config.denoising_box_noise_scale
            sel_lines = (sel_lines + line_noise).clamp(0.0, 1.0)

            if self.config.denoising_label_noise_ratio > 0:
                noise_mask = torch.rand_like(sel_labels.float()) < self.config.denoising_label_noise_ratio
                random_labels = torch.randint(0, self.config.num_classes, size=sel_labels.shape, device=device)
                sel_labels = torch.where(noise_mask, random_labels, sel_labels)

            dn_boxes[batch_idx] = sel_boxes
            dn_labels[batch_idx] = sel_labels
            dn_valid[batch_idx] = True
            dn_lines[batch_idx] = sel_lines

        dn_query = self.dn_query_content.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return {
            "dn_query": dn_query,
            "dn_boxes": dn_boxes,
            "dn_labels": dn_labels,
            "dn_valid": dn_valid,
            "dn_lines": dn_lines,
        }

    def _decode_queries(
        self,
        query: torch.Tensor,
        reference_boxes: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
    ) -> Dict[str, List[torch.Tensor]]:
        reference_logits = self._inverse_sigmoid(reference_boxes)
        line_reference_logits = self._inverse_sigmoid(reference_boxes)

        logits_per_layer: List[torch.Tensor] = []
        boxes_per_layer: List[torch.Tensor] = []
        lines_per_layer: List[torch.Tensor] = []

        for layer_idx, (layer, cls_head, box_head) in enumerate(zip(self.decoder_layers, self.decoder_class_heads, self.decoder_box_heads)):
            query_pos = self.reference_point_proj(reference_boxes[..., :2])
            query = layer(
                query=query,
                memory=memory,
                reference_points=reference_boxes,
                spatial_shapes=spatial_shapes,
                query_pos=query_pos,
            )
            pred_logits = cls_head(query)
            tmp_box_pred = box_head(query)
            pred_boxes = (tmp_box_pred + reference_logits).sigmoid()

            logits_per_layer.append(pred_logits)
            boxes_per_layer.append(pred_boxes)

            if self.decoder_line_heads is not None:
                tmp_line_pred = self.decoder_line_heads[layer_idx](query)
                pred_lines = (tmp_line_pred + line_reference_logits).sigmoid()
                lines_per_layer.append(pred_lines)
                line_reference_logits = tmp_line_pred.detach() + line_reference_logits.detach()

            reference_logits = tmp_box_pred.detach() + reference_logits.detach()
            reference_boxes = pred_boxes.detach()

        decoded: Dict[str, List[torch.Tensor]] = {
            "pred_logits": logits_per_layer,
            "pred_boxes": boxes_per_layer,
        }
        if len(lines_per_layer) > 0:
            decoded["pred_lines"] = lines_per_layer
        return decoded

    def forward(self, features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        fused = self._hybrid_encode(features)
        memory, spatial_shapes = self._flatten_memory(fused)

        query, reference_boxes = self._select_queries(memory)
        main_decoded = self._decode_queries(query, reference_boxes, memory, spatial_shapes)
        logits_per_layer = main_decoded["pred_logits"]
        boxes_per_layer = main_decoded["pred_boxes"]
        lines_per_layer = main_decoded.get("pred_lines", [])

        outputs: Dict[str, Union[torch.Tensor, List[Dict[str, torch.Tensor]]]] = {
            "pred_logits": logits_per_layer[-1],
            "pred_boxes": boxes_per_layer[-1],
        }
        if len(lines_per_layer) > 0:
            outputs["pred_lines"] = lines_per_layer[-1]

        if self.config.enable_auxiliary_losses and len(logits_per_layer) > 1:
            aux_outputs: List[Dict[str, torch.Tensor]] = []
            for aux_idx, (cls, box) in enumerate(zip(logits_per_layer[:-1], boxes_per_layer[:-1])):
                aux_output = {"pred_logits": cls, "pred_boxes": box}
                if len(lines_per_layer) > aux_idx:
                    aux_output["pred_lines"] = lines_per_layer[aux_idx]
                aux_outputs.append(aux_output)
            outputs["aux_outputs"] = aux_outputs

        targets = kwargs.get("targets")
        if self.training and self.config.enable_denoising and isinstance(targets, list):
            dn_pack = self._build_denoising_queries(
                targets=targets,
                batch_size=memory.shape[0],
                device=memory.device,
                dtype=memory.dtype,
            )
            if dn_pack is not None:
                dn_decoded = self._decode_queries(dn_pack["dn_query"], dn_pack["dn_boxes"], memory, spatial_shapes)
                dn_output: Dict[str, torch.Tensor] = {
                    "pred_logits": dn_decoded["pred_logits"][-1],
                    "pred_boxes": dn_decoded["pred_boxes"][-1],
                    "target_labels": dn_pack["dn_labels"],
                    "target_boxes": dn_pack["dn_boxes"],
                    "target_valid_mask": dn_pack["dn_valid"],
                }
                if "pred_lines" in dn_decoded:
                    dn_output["pred_lines"] = dn_decoded["pred_lines"][-1]
                    dn_output["target_lines"] = dn_pack["dn_lines"]
                outputs["dn_outputs"] = dn_output

        return outputs