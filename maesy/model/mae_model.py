"""Masked Autoencoder Vision Transformer for pretraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

from maesy.model.config import ModelConfig
from maesy.model.vision_transformer import PatchEmbedding, TransformerBlock


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder Vision Transformer.
    
    This model implements MAE pretraining where random patches are masked
    and the model learns to reconstruct them.
    """
    
    def __init__(self, config: ModelConfig, decoder_embed_dim: int = 768, decoder_num_layers: int = 8):
        """
        Initialize MAE model.
        
        Args:
            config: Model configuration
            decoder_embed_dim: Decoder embedding dimension
            decoder_num_layers: Number of decoder layers
        """
        super().__init__()
        self.config = config
        self.decoder_embed_dim = decoder_embed_dim
        
        # Encoder components (same as VisionTransformer)
        self.patch_embed = PatchEmbedding(config)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        
        # Positional embedding for encoder
        self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches + 1, config.embed_dim) * 0.02)
        
        # Transformer encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)
        
        # Decoder components
        self.decoder_embed = nn.Linear(config.embed_dim, decoder_embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim) * 0.02)
        
        # Positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, config.num_patches + 1, decoder_embed_dim) * 0.02)
        
        # Decoder transformer blocks
        # Create decoder config based on encoder config but with decoder-specific dimensions
        # Copy all parameters from original config to ensure compatibility
        decoder_config = ModelConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=decoder_embed_dim,
            num_layers=decoder_num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            num_classes=config.num_classes,
            num_queries=config.num_queries,
            hidden_dim=config.hidden_dim,
            num_decoder_layers=config.num_decoder_layers,
            bbox_loss_coef=config.bbox_loss_coef,
            class_loss_coef=config.class_loss_coef,
            giou_loss_coef=config.giou_loss_coef
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_config) for _ in range(decoder_num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head - reconstruct pixels
        self.decoder_pred = nn.Linear(decoder_embed_dim, config.patch_size ** 2 * config.in_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize patch embedding like nn.Linear
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize other parameters
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            imgs: [B, C, H, W]
            
        Returns:
            patches: [B, num_patches, patch_size**2 * C]
        """
        p = self.config.patch_size
        h = w = self.config.image_size // p
        
        x = imgs.reshape(imgs.shape[0], self.config.in_channels, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * self.config.in_channels)
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches to images.
        
        Args:
            x: [B, num_patches, patch_size**2 * C]
            
        Returns:
            imgs: [B, C, H, W]
        """
        p = self.config.patch_size
        h = w = self.config.image_size // p
        
        x = x.reshape(x.shape[0], h, w, p, p, self.config.in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], self.config.in_channels, h * p, w * p)
        return imgs
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking.
        
        Args:
            x: [B, N, D] - input sequence
            mask_ratio: Ratio of patches to mask
            
        Returns:
            x_masked: [B, N * (1 - mask_ratio), D] - masked sequence
            mask: [B, N] - binary mask (0 is keep, 1 is remove)
            ids_restore: [B, N] - indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with masking.
        
        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask
            
        Returns:
            x: Encoded visible patches [B, N_visible, D]
            mask: Binary mask
            ids_restore: Indices to restore original order
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add positional encoding (without cls token)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking: length -> length * (1 - mask_ratio)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Encoded visible patches [B, N_visible + 1, D]
            ids_restore: Indices to restore original order
            
        Returns:
            x: Reconstructed patches [B, num_patches, patch_size**2 * C]
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add positional encoding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MAE.
        
        Args:
            imgs: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask
            
        Returns:
            loss: Reconstruction loss
            pred: Predicted patches
            mask: Binary mask
        """
        # Validate input dimensions
        if imgs.shape[2] != imgs.shape[3]:
            raise ValueError(f"MAE expects square images, got {imgs.shape[2]}x{imgs.shape[3]}")
        
        if imgs.shape[2] != self.config.image_size:
            raise ValueError(f"Image size {imgs.shape[2]} doesn't match config image_size {self.config.image_size}")
        
        # Encode with masking
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        
        # Decode
        pred = self.forward_decoder(latent, ids_restore)
        
        # Compute loss
        target = self.patchify(imgs)
        
        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches (on masked patches only for more focussed training)
        
        return loss, pred, mask
    
    def get_encoder(self) -> nn.Module:
        """
        Get the encoder part for transfer learning.
        
        Returns:
            Encoder module with encoder weights
        """
        # This returns a module that can be used to initialize VisionTransformerDetector
        return self
