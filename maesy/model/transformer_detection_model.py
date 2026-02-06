from dataclasses import dataclass

import torch

from maesy.model import BaseModel
from maesy.model.backbones import TransformerBackboneConfig, TransformerBackbone
from maesy.model.components import Utils
from maesy.model.heads import DummyHead


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
    decoder_embed_dim: int = 768
    decoder_num_layers: int = 8
    decoder_mpl_ratio = 4.0
    decoder_dropout = 0.1
    decoder_attention_dropout = 0.1

    def __post_init__(self):
        """Validate configuration."""
        assert self.image_size % self.patch_size == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"
        self.num_patches = (self.image_size // self.patch_size) ** 2

class TransformerDetectionModel(BaseModel):

    def __init__(self, config: TransformerDetectorConfig):
        """
            Initialize MAE model.

            Args:
                config: Model configuration
                decoder_embed_dim: Decoder embedding dimension
                decoder_num_layers: Number of decoder layers
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

        self.head = DummyHead()


    def forward(self, x, **kwargs):
        x = Utils.patchify(x, self.config.image_size, self.config.patch_size)
        # x, mask, ids_shuffle = Utils.random_masking(x, **kwargs)
        out = super().forward(x, **{"ids_shuffle": torch.arange(0, x.shape[1]).unsqueeze(0)})  # to(device=self.device, non_blocking=True)
        return out
