"""MaeSy - Vision Transformer framework for object detection."""

__version__ = "0.1.0"

from .dataset import DatasetManager, ObjectDetectionDataset
from .model import VisionTransformerDetector, ModelConfig
from .training import Trainer, TrainingConfig
from .evaluation import Evaluator, evaluate_model

__all__ = [
    "DatasetManager",
    "ObjectDetectionDataset",
    "VisionTransformerDetector",
    "ModelConfig",
    "Trainer",
    "TrainingConfig",
    "Evaluator",
    "evaluate_model",
]
