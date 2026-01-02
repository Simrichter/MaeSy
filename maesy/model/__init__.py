"""Model architecture module for Vision Transformer object detection."""

from .vision_transformer import VisionTransformerDetector
from .config import ModelConfig

__all__ = [
    "VisionTransformerDetector",
    "ModelConfig",
]
