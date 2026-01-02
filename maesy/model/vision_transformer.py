"""Vision Transformer for object detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import math

from .config import ModelConfig


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        
        # Convolutional layer for patch embedding
        self.projection = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        x = self.projection(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    
    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return self.dropout(x + self.pos_embed)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class MLP(nn.Module):
    """MLP layer."""
    
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadAttention(
            config.embed_dim,
            config.num_heads,
            config.attention_dropout
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(
            config.embed_dim,
            int(config.embed_dim * config.mlp_ratio),
            config.dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DetectionHead(nn.Module):
    """Detection head for object detection."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.num_queries = config.num_queries
        
        # Object queries
        self.query_embed = nn.Embedding(config.num_queries, config.embed_dim)
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(config.embed_dim, config.num_classes + 1)  # +1 for no-object
        self.bbox_embed = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 4)  # [x, y, w, h]
        )
        
    def forward(self, encoder_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            encoder_output: Output from encoder [B, N, D]
            
        Returns:
            Dictionary with 'pred_logits' and 'pred_boxes'
        """
        B = encoder_output.shape[0]
        
        # Get queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, D]
        
        # Decode
        decoder_output = self.decoder(queries, encoder_output)  # [B, num_queries, D]
        
        # Predict classes and boxes
        pred_logits = self.class_embed(decoder_output)  # [B, num_queries, num_classes + 1]
        pred_boxes = self.bbox_embed(decoder_output).sigmoid()  # [B, num_queries, 4]
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }


class VisionTransformerDetector(nn.Module):
    """Vision Transformer for object detection."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.num_patches,
            config.embed_dim,
            config.dropout
        )
        
        # Transformer encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Detection head
        self.detection_head = DetectionHead(config)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dictionary with predictions
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Detection head (use all tokens except cls)
        encoder_output = x[:, 1:, :]  # [B, num_patches, embed_dim]
        
        predictions = self.detection_head(encoder_output)
        
        return predictions
    
    @torch.no_grad()
    def inference(self, x: torch.Tensor, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Inference mode with post-processing.
        
        Args:
            x: Input images [B, C, H, W]
            confidence_threshold: Confidence threshold for filtering
            
        Returns:
            Filtered predictions
        """
        self.eval()
        predictions = self.forward(x)
        
        # Get class predictions (ignore no-object class)
        pred_logits = predictions['pred_logits']
        pred_boxes = predictions['pred_boxes']
        
        # Convert logits to probabilities
        pred_probs = F.softmax(pred_logits, dim=-1)
        
        # Get max probability and class (excluding no-object class)
        scores, labels = pred_probs[:, :, :-1].max(-1)
        
        # Filter by confidence
        keep = scores > confidence_threshold
        
        results = []
        for i in range(x.shape[0]):
            results.append({
                'boxes': pred_boxes[i][keep[i]],
                'labels': labels[i][keep[i]],
                'scores': scores[i][keep[i]]
            })
        
        return results
