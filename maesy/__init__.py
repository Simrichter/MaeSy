"""MaeSy - Vision Transformer framework for object detection."""

__version__ = "0.1.0"

from .dataset import DatasetManager, ObjectDetectionDataset
from .model import ModelConfig
from .evaluation import Evaluator, evaluate_model
# from .debug import testMAEPretraining

__all__ = [
    "DatasetManager",
    "ObjectDetectionDataset",
    "ModelConfig",
    "Evaluator",
    "evaluate_model",
    # "testMAEPretraining"
]
