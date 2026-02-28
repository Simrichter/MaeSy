"""Vision Transformer for Object Detection using BaseModel framework."""

from dataclasses import dataclass
import torch

from .base_model import BaseModel
from .backbones import TransformerBackbone, TransformerBackboneConfig, ResNetBackbone
from .heads import ViTDetectionHead, DetectionHeadConfig
from .components import Utils
from .heads.detr_head import DETRHeadConfig, DETRHead


@dataclass
class DETRConfig:
    """Configuration for Vision Transformer Detector model."""

    # Image parameters
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    resnet_version: str = "resnet18"
    freeze_backbone: bool = True

    # Transformer backbone parameters
    embed_dim: int = 128
    num_layers: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Detection head parameters
    num_classes: int = 80
    num_queries: int = 100
    num_decoder_layers: int = 6
    decoder_num_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1
    hidden_dim: int = 256

    # Loss weights (for compatibility with loss functions)
    bbox_loss_coef: float = 5.0
    class_loss_coef: float = 1.0
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1


    def __post_init__(self):
        """Validate configuration."""
        assert self.image_size % self.patch_size == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"
        self.num_patches = (self.image_size // self.patch_size) ** 2


class DETR(BaseModel):
    """Vision Transformer for Object Detection.

    This model implements a ViT-based object detection architecture following
    the BaseModel framework by combining:
    - TransformerBackbone: Encodes image patches into feature representations
    - DetectionHead: Decodes features into bounding box predictions
    """

    def __init__(self, config: DETRConfig):
        """
        Initialize ViT Detector model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Create backbone configuration
        self.backbone = ResNetBackbone(version=self.config.resnet_version, remove_layers=2)
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = torch.nn.Conv2d(512, config.embed_dim, kernel_size=1) #TODO: Make this more flexible for different backbones

        # Create detection head configuration
        head_config = DETRHeadConfig(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            num_queries=config.num_queries,
            num_decoder_layers=config.num_decoder_layers,
            num_heads_encoder=config.decoder_num_heads,
            num_heads_decoder=config.decoder_num_heads,
            mlp_ratio=config.decoder_mlp_ratio,
            dropout=config.decoder_dropout,
            hidden_dim=config.hidden_dim
        )
        self.head = DETRHead(head_config)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Forward pass through the model.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary containing:
                - pred_logits: Class predictions [B, num_queries, num_classes + 1]
                - pred_boxes: Bounding box predictions [B, num_queries, 4]
        """
        features = self.backbone(x) # [B, C, H, W] -> [B, 512, H', W']

        features = self.projection(features) # [B, 512, H', W'] -> [B, embed_dim, H', W']
        features = features.flatten(2).transpose(1, 2) # [B, embed_dim, H', W'] -> [B, embed_dim, H'*W'] -> [B, H'*W', embed_dim]

        out = self.head(features)

        return out

    def infer(self, images, targets, **kwargs):
        out = self.forward(images, **kwargs)
        out['pred_logits'] = out['pred_logits'].softmax(-1)[..., :-1].detach()
        out['pred_boxes'] = out['pred_boxes'].detach()
        return out, targets
