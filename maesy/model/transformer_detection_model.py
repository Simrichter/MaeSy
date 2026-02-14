from dataclasses import dataclass

import torch

from maesy.model import BaseModel
from maesy.model.backbones import TransformerBackboneConfig, TransformerBackbone
from maesy.model.components import Utils
from maesy.model.heads import DetectionHead, DetectionHeadConfig


@dataclass
class TransformerDetectorConfig:
    """Configuration for Vision Transformer Detector model."""

    # Image parameters
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Transformer parameters
    embed_dim: int = 768
    num_layers: int = 12
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

    def __post_init__(self):
        """Validate configuration."""
        assert self.image_size % self.patch_size == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"
        self.num_patches = (self.image_size // self.patch_size) ** 2

class TransformerDetectionModel(BaseModel):
    """Transformer-based Object Detection Model.
    
    This model can utilize a pretrained MAE TransformerBackbone for object detection.
    It combines TransformerBackbone with DetectionHead following the BaseModel framework.
    """

    def __init__(self, config: TransformerDetectorConfig):
        """
        Initialize Transformer Detection model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        backbone_config = TransformerBackboneConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout
        )
        self.backbone = TransformerBackbone(backbone_config)

        head_config = DetectionHeadConfig(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            num_queries=config.num_queries,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.decoder_num_heads,
            mlp_ratio=config.decoder_mlp_ratio,
            dropout=config.decoder_dropout,
            hidden_dim=config.hidden_dim
        )
        self.head = DetectionHead(head_config)


    def forward(self, x, **kwargs):
        """
        Forward pass through the model.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dictionary containing:
                - pred_logits: Class predictions [B, num_queries, num_classes + 1]
                - pred_boxes: Bounding box predictions [B, num_queries, 4]
        """
        x = Utils.patchify(x, self.config.image_size, self.config.patch_size)
        # No masking for detection - use sequential ids
        ids_shuffle = Utils.create_sequential_ids(x.shape[0], x.shape[1], device=x.device)
        out = super().forward(x, ids_shuffle=ids_shuffle)
        return out
