"""RT-DETR style detector integrated with the MaeSy BaseModel contracts."""

from dataclasses import dataclass
from typing import Tuple

from .backbones import ResNetBackbone
from .backbones.resnet_backbone import ResNetBackboneConfig
from .base_model import BaseModel
from .heads import RTDETRHead, RTDETRHeadConfig


@dataclass
class RTDETRConfig:
    type: str = "RT-DETR"
    image_size: int = 224
    resnet_version: str = "resnet50"
    backbone_pretrained: bool = True
    feature_scales: Tuple[str, str, str] = ("c3", "c4", "c5")

    num_classes: int = 80
    num_queries: int = 100

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
    enable_auxiliary_losses: bool = True

    # Additional settings
    bbox_loss_coef: float = 5.0
    class_loss_coef: float = 1.0
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1
    aux_loss_coef: float = 1.0
    line_loss_coef: float = 2.0
    dn_loss_coef: float = 1.0


class RTDETR(BaseModel):
    """RT-DETR style detector with the same output contract as DETR."""

    def __init__(self, config: RTDETRConfig):
        super().__init__()
        self.config = config

        bbone_conf = ResNetBackboneConfig(
            version=self.config.resnet_version,
            image_size=self.config.image_size,
            pretrained=self.config.backbone_pretrained,
            feature_scales=self.config.feature_scales
        )
        self.backbone = ResNetBackbone(bbone_conf)

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
                enable_auxiliary_losses=config.enable_auxiliary_losses,
            )

        self.head = RTDETRHead(head_conf)

        print(
            f"Created RT-DETR model with backbone {self.backbone.type} and head {self.head.type}"
            # f"\n Feature channels: {self.backbone.get_feature_channels()}"
        )

    def infer(self, images, targets, **kwargs):
        out = self.forward(images, **kwargs)
        out["pred_logits"] = out["pred_logits"].softmax(-1).detach()
        out["pred_boxes"] = out["pred_boxes"].detach()
        if "pred_lines" in out:
            out["pred_lines"] = out["pred_lines"].detach()
        return out, targets

