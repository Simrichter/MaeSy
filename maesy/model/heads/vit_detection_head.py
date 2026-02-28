"""Detection head for object detection with transformer decoder."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict


@dataclass
class DetectionHeadConfig:
    """Configuration for Detection Head."""
    embed_dim: int = 128
    num_classes: int = 80
    num_queries: int = 100
    num_decoder_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    hidden_dim: int = 256


class ViTDetectionHead(nn.Module):
    """Detection head for object detection using transformer decoder.
    
    This head takes encoded features from a backbone and produces
    bounding box predictions and class logits using a transformer decoder
    with learned object queries.
    """
    
    def __init__(self, config: DetectionHeadConfig):
        super().__init__()
        self.type = "DetectionHead"
        self.config = config
        
        # Object queries (learnable embeddings)
        self.query_embed = nn.Embedding(config.num_queries, config.embed_dim)
        
        # Transformer decoder
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
        
        # Classification head
        self.class_embed = nn.Linear(config.embed_dim, config.num_classes + 1)  # +1 for no-object
        
        # Bounding box regression head
        self.bbox_embed = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 4), # [cx, cy, w, h]
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through detection head.
        
        Args:
            features: Encoded features from backbone [B, N, D]
            
        Returns:
            Dictionary containing:
                - pred_logits: Class predictions [B, num_queries, num_classes + 1]
                - pred_boxes: Bounding box predictions [B, num_queries, 4]
        """
        B = features.shape[0]
        
        # Get object queries [B, num_queries, D]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # Decode queries with encoder features
        decoder_output = self.decoder(queries, features)  # [B, num_queries, D]
        
        # Predict classes and bounding boxes
        pred_logits = self.class_embed(decoder_output)  # [B, num_queries, num_classes + 1]
        pred_boxes = self.bbox_embed(decoder_output).sigmoid()  # [B, num_queries, 4] normalized to [0, 1]
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }
