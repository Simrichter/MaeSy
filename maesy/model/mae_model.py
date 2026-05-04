"""Masked Autoencoder Vision Transformer for pretraining."""
from dataclasses import dataclass

import torch

from .base_model import BaseModel
from .backbones import TransformerBackbone, TransformerBackboneConfig
from .components import Utils
from .heads import MaeDecoderHead, MaeDecoderHeadConfig

@dataclass
class MAEConfig:
    """
    Configuration for Masked Autoencoder model.
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

class MaskedAutoencoderViT(BaseModel):
    """Masked Autoencoder Vision Transformer.
    
    This model implements a MAE architecture
    Its usage is a pretraining where random patches are masked
    and the model learns to reconstruct them.
    """

    def __init__(self, config: MAEConfig):
        """
        Initialize MAE model.
        
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

        head_config = MaeDecoderHeadConfig(
            embed_dim=config.decoder_embed_dim,
            num_patches=backbone_config.num_patches,
            patch_size=config.patch_size,
            num_heads=config.num_heads,
            mlp_ratio=config.decoder_mpl_ratio,
            dropout=config.decoder_dropout,
            attention_dropout=config.decoder_attention_dropout,
            num_layers=config.decoder_num_layers,
            in_channels=config.in_channels
        )
        self.head = MaeDecoderHead(head_config)

    def forward(self, x, **kwargs):
        x = Utils.patchify(x, self.config.image_size, self.config.patch_size)
        x, mask, ids_shuffle = Utils.random_masking(x, **kwargs)
        out = super().forward(x, **{"ids_shuffle": ids_shuffle}) # to(device=self.device, non_blocking=True)
        return out, {"mask": mask, "ids_shuffle": ids_shuffle}

    def reconstruct(self, out, orig_images = None, **kwargs):
        out = out.detach() * kwargs['mask'].unsqueeze(-1)
        model_out = Utils.unpatchify(out, self.config.image_size, self.config.patch_size)
        model_out = torch.clamp(model_out, 0, 255)  # Clamp to valid color range
        if orig_images is not None:
            orig_patches = Utils.patchify(orig_images, self.config.image_size, self.config.patch_size)
            given_patches = orig_patches * (1 - kwargs['mask']).unsqueeze(-1)
            # model_out += given_patches
            imgs_masked = Utils.unpatchify(given_patches, self.config.image_size, self.config.patch_size)
            model_out += imgs_masked
            model_out = torch.cat((imgs_masked, model_out), dim=-1)
        return model_out

    def infer(self, images, targets, **kwargs):
        # Get predictions
        predictions, additional_data = self.forward(images, **kwargs)
        img_preds = self.reconstruct(predictions, orig_images=images, **additional_data)

        # patches = Utils.patchify(images, self.model.config.image_size, self.model.config.patch_size)
        # imgs_masked = Utils.unpatchify(patches * (1 - additional_data["mask"]).unsqueeze(-1),
        #                                self.model.config.image_size, self.model.config.patch_size)
        img_preds = torch.cat((images, img_preds), dim=-1)

        return img_preds.detach(), targets
