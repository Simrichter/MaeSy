import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .base_model import BaseModel
from .backbones import (
    MobileNetBackbone,
    MobileNetBackboneConfig,
    ResNetBackbone,
    ResNetBackboneConfig,
    SWINBackbone,
    SWINBackboneConfig,
)
from .components import Utils
from .heads import MaeMultiscaleDecoder, MaeMultiscaleDecoderConfig


@dataclass
class MaeMultiscaleConfig:
    """
    Config class for the MaeMultiscale model, which is a masked autoencoder based on multi-scale backbones for self-supervised learning on vision tasks.
    This config includes parameters for the backbone architecture (e.g., ResNet, SWIN, MobileNet), input image size, and which feature scales to extract from the backbone.
    Also contains configuration for the multiscale decoder head.
    """
    type: str = "mae_multiscale"
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = field(init=False) # Auto-calculated in post_init
    in_channels: int = 3

    backbone_version: str = "resnet50"
    backbone_pretrained: bool = True
    feature_scales: Tuple[str, str, str] = ("c3", "c4", "c5")
    use_decoder_skip_connections: bool = True
    decoder_skip_scales: Tuple[str, ...] = field(default_factory=lambda: ("c3", "c4"))

    decoder_embed_dim: int = 384
    decoder_num_layers: int = 4
    decoder_num_heads: int = 6
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1
    decoder_attention_dropout: float = 0.1
    decoder_window_size: int = 7

    def __post_init__(self):
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (self.image_size // self.patch_size) ** 2


class MaskedAutoencoderMultiscale(BaseModel):
    def __init__(self, config: MaeMultiscaleConfig) -> None:
        super().__init__()
        self.config = config

        if self.config.backbone_version.startswith("resnet"):
            bbone_conf = ResNetBackboneConfig(
                version=self.config.backbone_version,
                image_size=self.config.image_size,
                pretrained=self.config.backbone_pretrained,
                feature_scales=self.config.feature_scales,
            )
            self.backbone = ResNetBackbone(bbone_conf)
        elif self.config.backbone_version.startswith("swin"):
            bbone_conf = SWINBackboneConfig(
                version=self.config.backbone_version,
                image_size=self.config.image_size,
                pretrained=self.config.backbone_pretrained,
                feature_scales=self.config.feature_scales,
            )
            self.backbone = SWINBackbone(bbone_conf)
        elif self.config.backbone_version.startswith("mobilenet"):
            bbone_conf = MobileNetBackboneConfig(
                version=self.config.backbone_version,
                image_size=self.config.image_size,
                pretrained=self.config.backbone_pretrained,
                feature_scales=self.config.feature_scales,
            )
            self.backbone = MobileNetBackbone(bbone_conf)
        else:
            raise ValueError(f"Unknown backbone version {self.config.backbone_version}")

        feature_dims: Dict[str, int] = {
            scale: dims[0]
            for scale, dims in self.backbone.get_feature_dims().items()
        }
        head_conf = MaeMultiscaleDecoderConfig(
            embed_dim=config.decoder_embed_dim,
            num_patches=config.num_patches,
            patch_size=config.patch_size,
            num_heads=config.decoder_num_heads,
            mlp_ratio=config.decoder_mlp_ratio,
            dropout=config.decoder_dropout,
            attention_dropout=config.decoder_attention_dropout,
            num_layers=config.decoder_num_layers,
            in_channels=config.in_channels,
            feature_dims=feature_dims,
            feature_scales=config.feature_scales,
            use_skip_connections=config.use_decoder_skip_connections,
            skip_scales=config.decoder_skip_scales,
            window_size=config.decoder_window_size,
        )
        self.head = MaeMultiscaleDecoder(head_conf)

        print(
            f"Created multiscale MAE model with backbone {self.backbone.type} and head {self.head.type}"
        )

    def forward(self, x: torch.Tensor):
        """Forward pass expects already-masked (or unmasked) images from the trainer."""
        out = super().forward(x)
        return Utils.unpatchify(out, self.config.image_size, self.config.patch_size, self.config.in_channels)

    def reconstruct(self, out, orig_images=None, **kwargs):
        out = out.detach() * kwargs["mask"].unsqueeze(-1)
        model_out = Utils.unpatchify(out, self.config.image_size, self.config.patch_size, self.config.in_channels)
        if orig_images is not None:
            orig_patches = Utils.patchify(orig_images, self.config.image_size, self.config.patch_size, self.config.in_channels)
            given_patches = orig_patches * (1 - kwargs["mask"]).unsqueeze(-1)
            imgs_masked = Utils.unpatchify(given_patches, self.config.image_size, self.config.patch_size, self.config.in_channels)
            model_out = model_out + imgs_masked
            model_out = torch.cat((imgs_masked, model_out), dim=-1)
        return model_out

    def infer(self, images, targets, **kwargs):
        predictions, additional_data = self.forward(images, **kwargs)
        img_preds = self.reconstruct(predictions, orig_images=images, **additional_data)
        img_preds = torch.cat((images, img_preds), dim=-1)
        return img_preds.detach(), targets
