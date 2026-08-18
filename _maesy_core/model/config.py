"""Model configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for Vision Transformer Detector model."""
    
    # Image parameters
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    
    # Transformer parameters
    head_in_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Detection parameters
    num_classes: int = 80
    num_queries: int = 100  # Number of object queries
    
    # Detection head parameters
    hidden_dim: int = 256
    num_decoder_layers: int = 6
    
    # Loss weights
    bbox_loss_coef: float = 5.0
    class_loss_coef: float = 1.0
    giou_loss_coef: float = 2.0
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.image_size % self.patch_size == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"
        self.num_patches = (self.image_size // self.patch_size) ** 2
