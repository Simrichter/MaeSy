"""Masked Autoencoder Vision Transformer for pretraining."""
from dataclasses import dataclass
from .base_model import BaseModel
from .backbones import TransformerBackbone, TransformerBackboneConfig
from .components import Utils
from .heads import DecoderHead, DecoderHeadConfig

@dataclass
class MAEConfig:
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
    hidden_dim: int = 256
    num_decoder_layers: int = 6

    def __post_init__(self):
        """Validate configuration."""
        assert self.image_size % self.patch_size == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"
        self.num_patches = (self.image_size // self.patch_size) ** 2

class MaskedAutoencoderViT(BaseModel):
    """Masked Autoencoder Vision Transformer.
    
    This model implements a MAE architecture
    Its usage is a pretraining where random patches are masked
    and the model learns to reconstruct them.
    """

    def __init__(self, config: MAEConfig, decoder_embed_dim: int = 768, decoder_num_layers: int = 8, decoder_mpl_ratio=4.0,
                 decoder_dropout=0.1, decoder_attention_dropout=0.1):
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

        head_config = DecoderHeadConfig(
            embed_dim=decoder_embed_dim,
            num_patches=backbone_config.num_patches,
            patch_size=config.patch_size,
            num_heads=config.num_heads,
            mlp_ratio=decoder_mpl_ratio,
            dropout=decoder_dropout,
            attention_dropout=decoder_attention_dropout,
            num_layers=decoder_num_layers,
            in_channels=config.in_channels
        )
        self.head = DecoderHead(head_config)

    def forward(self, x, **kwargs):
        x = Utils.patchify(x, self.config.image_size, self.config.patch_size)
        x, mask, ids_shuffle = Utils.random_masking(x, **kwargs)
        out = super().forward(x, **{"mask": mask, "ids_shuffle": ids_shuffle}) # to(device=self.device, non_blocking=True)
        return out, {"mask": mask, "ids_shuffle": ids_shuffle}

    def reconstruct(self, out, orig_images = None, **kwargs):
        out = out.detach() * kwargs['mask'].unsqueeze(-1)
        if orig_images is not None:
            orig_patches = Utils.patchify(orig_images, self.config.image_size, self.config.patch_size)
            out += orig_patches * (1 - kwargs['mask']).unsqueeze(-1)

        model_out = Utils.unpatchify(out, self.config.image_size,
                                     self.config.patch_size)
        return model_out