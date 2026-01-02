"""Classification Vision Transformer for pretraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.vision_transformer import PatchEmbedding, TransformerBlock


class ClassificationViT(nn.Module):
    """Vision Transformer for image classification pretraining.
    
    This model implements standard supervised image classification pretraining
    using the ViT architecture.
    """
    
    def __init__(self, config: ModelConfig, num_classes: int = 1000):
        """
        Initialize classification model.
        
        Args:
            config: Model configuration
            num_classes: Number of classes for classification
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches + 1, config.embed_dim) * 0.02)
        self.pos_dropout = nn.Dropout(config.dropout)
        
        # Transformer encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Classification head
        self.head = nn.Linear(config.embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize patch embedding projection
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize cls token and pos embedding
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Logits [B, num_classes]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer encoder
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification head (use cls token)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits
    
    def get_encoder(self) -> nn.Module:
        """
        Get the encoder part for transfer learning.
        
        Returns:
            Encoder module with encoder weights
        """
        return self
