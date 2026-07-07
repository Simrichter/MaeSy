"""RT-DETR style detector integrated with the MaeSy BaseModel contracts."""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import torch
from torch import nn

from .backbones import ResNetBackbone, ResNetBackboneConfig, MobileNetBackbone, MobileNetBackboneConfig, SWINBackbone, SWINBackboneConfig
from .base_model import BaseModel
from .heads import RTDETRHead, RTDETRHeadConfig
from maesy.dataset import sanitize_xyxy


@dataclass
class RTDETRConfig:
    type: str = "RT-DETR"
    image_size: int = 224
    backbone_version: str = "resnet50"
    backbone_pretrained: bool = True
    feature_scales: Tuple[str, str, str] = ("c3", "c4", "c5")

    num_classes: int = 80
    num_queries: int = 100
    softmax_activated: bool = False

    embed_dim: int = 256
    num_decoder_layers: int = 6
    decoder_num_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1
    hidden_dim_out_layers: int = 256
    num_deformable_points: int = 4
    enable_denoising: bool = False
    denoising_num_queries: int = 0
    denoising_label_noise_ratio: float = 0.2
    denoising_box_noise_scale: float = 0.4
    enable_line_detection: bool = False
    line_class_id: int = -1
    enable_ellipse_detection: bool = False
    ellipse_class_id: int = -1
    enable_auxiliary_losses: bool = True
    lightweight_fusion: bool = True
    num_rep_blocks_in_fusion: int = 3


def _decode_detr_predictions(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_lines: torch.Tensor | None = None,
    pred_ellipses: torch.Tensor | None = None,
    line_class_id: int | None = None,
    ellipse_class_id: int | None = None,
    no_object_class: int | None = None,
    score_threshold: float = 0.0,
    softmax_activated: bool = True
) -> List[Dict[str, torch.Tensor]]:
    """
    Decode DETR outputs to a list of ``xyxy`` detections per image.

    Args:
        :param pred_logits: Tensor of cls logits
        :param pred_boxes: Tensor of predicted box coordinates
        :param pred_lines: Optional Tensor of predicted line endpoint coordinates
        :param pred_ellipses: Optional Tensor of predicted ellipses in cholesky format
        :param line_class_id: Int specifying the cls_id for lines
        :param ellipse_class_id: Int specifying the cls_id for ellipses
        :param no_object_class: Int specifying the cls_id for no objects. Defaults to last cls_id
        :param score_threshold: Threshold to filter predictions with too low confidence
        :param softmax_activated: whether the class probabilities are meant as logits for softmax (or sigmoid if set to false)

    Returns:
        List of dicts containing:
            {"boxes": boxes_xyxy,
            "labels": det_labels,
            "scores": det_scores,
            "line_points": line_points,
            "line_labels": line_labels,
            "line_scores": line_scores}
    """
    if no_object_class is None:
        no_object_class = pred_logits.shape[-1] - 1

    probs = pred_logits.softmax(-1) if softmax_activated else pred_logits.sigmoid()
    scores, labels = probs.max(-1)
    decoded: List[Dict[str, torch.Tensor]] = []

    for img_idx in range(pred_logits.shape[0]):
        mask = labels[img_idx] != no_object_class
        if score_threshold > 0.0:
            mask = mask & (scores[img_idx] >= score_threshold)

        masked_boxes = pred_boxes[img_idx][mask].detach().cpu().float()
        masked_labels = labels[img_idx][mask].detach().cpu().long()
        masked_scores = scores[img_idx][mask].detach().cpu().float()

        # Route geometry by matched class: bbox classes use pred_boxes, line class uses pred_lines, ellipses use pred_ellipses.
        if line_class_id is not None and pred_lines is not None:
            bbox_mask = masked_labels != line_class_id
            line_mask = masked_labels == line_class_id
        else:
            bbox_mask = torch.ones_like(masked_labels, dtype=torch.bool)
            line_mask = torch.zeros_like(masked_labels, dtype=torch.bool)
        if ellipse_class_id is not None and pred_ellipses is not None:
            bbox_mask = bbox_mask & (masked_labels != ellipse_class_id)
            ellipse_mask = masked_labels == ellipse_class_id
        else:
            ellipse_mask = torch.zeros_like(masked_labels, dtype=torch.bool)

        boxes_xyxy, valid = sanitize_xyxy(masked_boxes[bbox_mask])
        if bbox_mask.any():
            det_labels = masked_labels[bbox_mask][valid]
            det_scores = masked_scores[bbox_mask][valid]
        else:
            det_labels = torch.empty((0,), dtype=torch.long)
            det_scores = torch.empty((0,), dtype=torch.float32)

        if line_mask.any() and pred_lines is not None:
            line_points = pred_lines[img_idx][mask][line_mask].detach().cpu().float().clamp(0.0, 1.0)
            line_labels = masked_labels[line_mask]
            line_scores = masked_scores[line_mask]
        else:
            line_points = torch.empty((0, 4), dtype=torch.float32)
            line_labels = torch.empty((0,), dtype=torch.long)
            line_scores = torch.empty((0,), dtype=torch.float32)

        if ellipse_mask.any() and pred_ellipses is not None:
            ellipses = pred_ellipses[img_idx][mask][ellipse_mask].detach().cpu().float()
            ellipse_labels = masked_labels[ellipse_mask]
            ellipse_scores = masked_scores[ellipse_mask]
        else:
            ellipses = torch.empty((0, 6), dtype=torch.float32)
            ellipse_labels = torch.empty((0,), dtype=torch.long)
            ellipse_scores = torch.empty((0,), dtype=torch.float32)

        decoded.append({
            "boxes": boxes_xyxy,
            "labels": det_labels,
            "scores": det_scores,
            "line_points": line_points,
            "line_labels": line_labels,
            "line_scores": line_scores,
            "ellipses": ellipses,
            "ellipse_labels": ellipse_labels,
            "ellipse_scores": ellipse_scores
        })

    return decoded


class RTDETR(BaseModel):
    """RT-DETR detector"""

    def __init__(self, config: RTDETRConfig):
        super().__init__()
        self.config = config

        if self.config.backbone_version.startswith("resnet"):
            bbone_conf = ResNetBackboneConfig(
                version=self.config.backbone_version,
                image_size=self.config.image_size,
                pretrained=self.config.backbone_pretrained,
                feature_scales=self.config.feature_scales
            )
            self.backbone = ResNetBackbone(bbone_conf)
        elif self.config.backbone_version.startswith("swin"):
            bbone_conf = SWINBackboneConfig(
                version=self.config.backbone_version,
                image_size=self.config.image_size,
                pretrained=self.config.backbone_pretrained,
                feature_scales=self.config.feature_scales
            )
            self.backbone = SWINBackbone(bbone_conf)
        elif self.config.backbone_version.startswith("mobilenet"):
            bbone_conf = MobileNetBackboneConfig(
                version=self.config.backbone_version,
                image_size=self.config.image_size,
                pretrained=self.config.backbone_pretrained,
                feature_scales=self.config.feature_scales
            )
            self.backbone = MobileNetBackbone(bbone_conf)
        else:
            raise ValueError(f"Unknown backbone version {self.config.backbone_version}")

        head_conf = RTDETRHeadConfig(
                feature_channels=self.backbone.get_feature_channels(), # 512, 1024, 2048
                num_classes=config.num_classes,
                num_queries=config.num_queries,
                embed_dim=config.embed_dim,
                num_decoder_layers=config.num_decoder_layers,
                decoder_num_heads=config.decoder_num_heads,
                decoder_mlp_ratio=config.decoder_mlp_ratio,
                decoder_dropout=config.decoder_dropout,
                hidden_dim_out_layers=config.hidden_dim_out_layers,
                num_feature_levels=len(config.feature_scales),
                num_deformable_points=config.num_deformable_points,
                enable_denoising=config.enable_denoising,
                denoising_num_queries=config.denoising_num_queries,
                denoising_label_noise_ratio=config.denoising_label_noise_ratio,
                denoising_box_noise_scale=config.denoising_box_noise_scale,
                enable_line_detection=config.enable_line_detection,
                line_class_id=config.line_class_id,
                enable_ellipse_detection=config.enable_ellipse_detection,
                ellipse_class_id=config.ellipse_class_id,
                enable_auxiliary_losses=config.enable_auxiliary_losses,
                lightweight_fusion=config.lightweight_fusion,
                num_rep_blocks_in_fusion=config.num_rep_blocks_in_fusion
            )

        self.head:RTDETRHead = RTDETRHead(head_conf)

        print(
            f"Created RT-DETR model with backbone {self.backbone.type} and head {self.head.type}"
            # f"\n Feature channels: {self.backbone.get_feature_channels()}"
        )

    def update_head_conf(self, num_classes: Optional[int] = None, special_classes: Dict[str, int] = None) -> None:
        """
            Update the configs with new number of classes or special class IDs and create a new classification head

            Args:
                :param num_classes: Number of classes
                :param special_classes: Dict of special class IDs to be changed. If no change desired leave entry None (Or entire dict)

        """
        changed = False
        if num_classes is not None:
            self.config.num_classes = num_classes
            self.head.config.num_classes = num_classes
            changed = True

        if special_classes is not None:
            line_class_id = special_classes.get("line_class_id", None)
            if line_class_id is not None:
                self.config.line_class_id = line_class_id
                self.head.config.line_class_id = line_class_id
                changed = True

            ellipse_class_id = special_classes.get("ellipse_class_id", None)
            if ellipse_class_id is not None:
                self.config.ellipse_class_id = ellipse_class_id
                self.head.config.ellipse_class_id = ellipse_class_id
                changed = True

        if changed:
            self.head.create_class_heads()

    def infer(self, images: torch.Tensor, targets: Dict[str, torch.Tensor], **kwargs):
        score_threshold = kwargs.pop("score_threshold", 0.3)
        raw_out = self.forward(images, **kwargs)

        predictions = _decode_detr_predictions(
            pred_logits=raw_out["pred_logits"],
            pred_boxes=raw_out["pred_boxes"],
            pred_lines=raw_out.get("pred_lines"),
            pred_ellipses=raw_out.get("pred_ellipses"),
            line_class_id=self.config.line_class_id,
            ellipse_class_id=self.config.ellipse_class_id,
            no_object_class=raw_out["pred_logits"].shape[-1] - 1,
            score_threshold=score_threshold,
            softmax_activated=self.config.softmax_activated
        )

        return raw_out, predictions, targets

    def get_export_wrapper(self):
        # return self
        return _ExportRTDETRWrapper(self)
    # TODO: Make nice such that the BaseModel version of this automatically gives the model-specific exportWrapper if existent
    def get_output_names(self):
        return ["boxes", "labels", "scores", "line_points", "line_labels", "line_scores"]  # TODO: Check/make dynamic

class _ExportRTDETRWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # _, outputs, _ = self.model.infer(x, None, score_threshold=0.0)
        outputs = self.model.forward(x)
        return outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_lines"]

    def get_output_names(self):
        # return  ["boxes", "labels", "scores", "line_points", "line_labels", "line_scores"] # TODO: Check/make dynamic
        return ["logits", "boxes", "lines"]