"""Evaluation module for model evaluation and metrics."""

from .evaluator import Evaluator, evaluate_model
from .metrics import (
    compute_detection_metrics,
    compute_map,
    compute_precision_recall,
    decode_detr_predictions,
    prepare_targets_for_detection_metrics,
)
from maesy.evaluation.infer_video import infer_video
from .visualizer import visualize_annotations

__all__ = [
    "Evaluator",
    "evaluate_model",
    "compute_detection_metrics",
    "compute_map",
    "compute_precision_recall",
    "decode_detr_predictions",
    "prepare_targets_for_detection_metrics",
    "infer_video",
    "visualize_annotations",
]
