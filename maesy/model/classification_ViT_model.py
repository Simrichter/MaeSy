"""Classification Vision Transformer for pretraining."""
from dataclasses import dataclass

from maesy.model.config import ModelConfig
from .base_model import BaseModel
from .heads import LinearHead, LinearHeadConfig
from .backbones import TransformerBackbone, TransformerBackboneConfig
@dataclass
class ClassificationViTConfig(ModelConfig):
    """
    Configuration for Vision Transformer Detector model.
    """

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

    # Classification head parameters
    hidden_dim: int = 256
    num_classes: int = 3


class ClassificationViT(BaseModel):
    """Vision Transformer for image classification pretraining.
    
    This model implements standard supervised image classification pretraining
    using the ViT architecture.
    """
    
    def __init__(self, config: ClassificationViTConfig):
        """
        Initialize classification model.
        
        Args:
            config: Model configuration
        """
        super().__init__()

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

        head_config = LinearHeadConfig(
            input_dim=config.embed_dim,
            num_classes=config.num_classes
        )

        self.backbone = TransformerBackbone(backbone_config)
        self.backbonetype = "TransformerBackbone"

        self.head = LinearHead(head_config)
        self.headtype = "LinearHead"

