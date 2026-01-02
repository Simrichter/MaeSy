"""Evaluation module for model evaluation and metrics."""

from .evaluator import Evaluator, evaluate_model
from .metrics import compute_map, compute_precision_recall

__all__ = [
    "Evaluator",
    "evaluate_model",
    "compute_map",
    "compute_precision_recall",
]
