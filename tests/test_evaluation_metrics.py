import pytest
import torch

from _maesy_core.model.rt_detr import _decode_detr_predictions
from maesy.evaluation.metrics import (
    compute_detection_metrics,
    prepare_targets_for_detection_metrics,
)


def _detection(boxes, labels, scores):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def test_decode_detr_predictions_filters_no_object_queries_and_routes_geometry():
    # classes: bbox 0, line 1, no-object 2
    pred_logits = torch.tensor(
        [[[6.0, -6.0, -8.0], [-6.0, 6.0, -8.0], [-6.0, -6.0, 6.0]]],
        dtype=torch.float32,
    )
    pred_boxes = torch.tensor(
        [[[0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.6, 0.6], [0.15, 0.15, 0.25, 0.25]]],
        dtype=torch.float32,
    )
    pred_lines = torch.tensor(
        [[[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.8, 0.8], [0.2, 0.2, 0.3, 0.3]]],
        dtype=torch.float32,
    )

    decoded = _decode_detr_predictions(
        pred_logits,
        pred_boxes,
        pred_lines=pred_lines,
        line_class_id=1,
        no_object_class=2,
    )

    assert len(decoded) == 1
    assert decoded[0]["boxes"].shape == (1, 4)
    assert decoded[0]["labels"].tolist() == [0]
    assert decoded[0]["line_points"].shape == (1, 4)
    assert decoded[0]["line_labels"].tolist() == [1]


def test_perfect_bbox_detection_has_exact_metric_values():
    predictions = [_detection([[0.4, 0.4, 0.6, 0.6]], [0], [0.9])]
    targets = [{
        "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.long),
    }]

    metrics = compute_detection_metrics(predictions, targets, num_classes=1)

    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["mAP50_95"] == pytest.approx(1.0)
    assert metrics["AP50_class_0"] == pytest.approx(1.0)
    assert metrics["precision50"] == pytest.approx(1.0)
    assert metrics["recall50"] == pytest.approx(1.0)
    assert metrics["f1_50"] == pytest.approx(1.0)
    assert metrics["pred_gt_ratio"] == pytest.approx(1.0)
    assert metrics["curves"]["bbox"]["combined"]["pr"]["recall"].tolist() == [1.0]


def test_high_confidence_false_positive_reduces_ap_and_precision():
    predictions = [
        _detection(
            [[0.0, 0.0, 0.1, 0.1], [0.4, 0.4, 0.6, 0.6]],
            [0, 0],
            [0.99, 0.9],
        )
    ]
    targets = [{
        "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.long),
    }]

    metrics = compute_detection_metrics(predictions, targets, num_classes=1)

    assert metrics["AP50_class_0"] == pytest.approx(0.5)
    assert metrics["precision50"] == pytest.approx(0.5)
    assert metrics["recall50"] == pytest.approx(1.0)
    assert metrics["f1_50"] == pytest.approx(2.0 / 3.0)
    assert metrics["pred_gt_ratio"] == pytest.approx(2.0)


def test_missed_target_has_zero_recall_and_ap():
    predictions = [_detection([], [], [])]
    targets = [{
        "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.long),
    }]

    metrics = compute_detection_metrics(predictions, targets, num_classes=1)

    assert metrics["mAP50"] == 0.0
    assert metrics["precision50"] == 0.0
    assert metrics["recall50"] == 0.0
    assert metrics["pred_gt_ratio"] == 0.0


def test_metrics_handle_no_bbox_targets_and_lines_without_division_by_zero():
    predictions = [{
        **_detection([], [], []),
        "line_points": torch.tensor([[0.1, 0.1, 0.9, 0.9]], dtype=torch.float32),
        "line_scores": torch.tensor([0.9], dtype=torch.float32),
    }]
    raw_targets = [{
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
        "line_points": torch.tensor([[0.1, 0.1, 0.9, 0.9]], dtype=torch.float32),
    }]
    targets = prepare_targets_for_detection_metrics(raw_targets, line_class_id=1)

    metrics = compute_detection_metrics(
        predictions,
        targets,
        num_classes=2,
        line_class_id=1,
        line_distance_thresholds=(0.05,),
    )

    assert metrics["pred_gt_ratio"] == 0.0
    assert metrics["line_pred_gt_ratio"] == pytest.approx(1.0)
    assert metrics["line_AP@0.05"] == pytest.approx(1.0)


def test_prepare_targets_filters_invalid_boxes_and_routes_line_targets():
    line_class_id = 2
    targets = [{
        "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6], [0.8, 0.8, 0.7, 0.9]], dtype=torch.float32),
        "labels": torch.tensor([0, 1, line_class_id], dtype=torch.long),
        "line_points": torch.tensor([[0.1, 0.1, 0.8, 0.8]], dtype=torch.float32),
    }]

    prepared = prepare_targets_for_detection_metrics(targets, line_class_id=line_class_id)

    assert prepared[0]["boxes"].shape == (1, 4)
    assert prepared[0]["labels"].tolist() == [0]
    assert prepared[0]["line_points"].shape == (1, 4)
    assert prepared[0]["line_labels"].tolist() == [line_class_id]


def test_compute_detection_metrics_includes_line_metrics_when_enabled():
    predictions = [{
        **_detection([[0.4, 0.4, 0.6, 0.6]], [0], [0.95]),
        "line_points": torch.tensor([[0.1, 0.1, 0.8, 0.8]], dtype=torch.float32),
        "line_labels": torch.tensor([2], dtype=torch.long),
        "line_scores": torch.tensor([0.9], dtype=torch.float32),
    }]
    raw_targets = [{
        "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
        "labels": torch.tensor([0, 2], dtype=torch.long),
        "line_points": torch.tensor([[0.1, 0.1, 0.8, 0.8]], dtype=torch.float32),
    }]
    targets = prepare_targets_for_detection_metrics(raw_targets, line_class_id=2)

    metrics = compute_detection_metrics(
        predictions,
        targets,
        num_classes=3,
        line_class_id=2,
        line_distance_thresholds=(0.05,),
    )

    assert metrics["line_precision@0.05"] == pytest.approx(1.0)
    assert metrics["line_recall@0.05"] == pytest.approx(1.0)
    assert metrics["line_f1@0.05"] == pytest.approx(1.0)
    assert metrics["line_AP@0.05"] == pytest.approx(1.0)
    assert metrics["line_endpoint_error@0.05"] == pytest.approx(0.0)
    assert metrics["line_mAP"] == pytest.approx(1.0)


def test_compute_detection_metrics_includes_ellipse_metrics_and_curves():
    predictions = [{
        **_detection([[0.4, 0.4, 0.6, 0.6]], [0], [0.95]),
        "ellipses": torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0, 1.0]], dtype=torch.float32),
        "ellipse_labels": torch.tensor([3], dtype=torch.long),
        "ellipse_scores": torch.tensor([0.9], dtype=torch.float32),
    }]
    raw_targets = [{
        "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
        "labels": torch.tensor([0, 3], dtype=torch.long),
        "ellipses": torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0, 1.0]], dtype=torch.float32),
    }]
    targets = prepare_targets_for_detection_metrics(raw_targets, ellipse_class_id=3)

    metrics = compute_detection_metrics(
        predictions,
        targets,
        num_classes=4,
        ellipse_class_id=3,
        ellipse_distance_thresholds=(0.1,),
    )

    assert metrics["ellipse_precision@0.10"] == pytest.approx(1.0)
    assert metrics["ellipse_recall@0.10"] == pytest.approx(1.0)
    assert metrics["ellipse_f1@0.10"] == pytest.approx(1.0)
    assert metrics["ellipse_AP@0.10"] == pytest.approx(1.0)
    assert metrics["ellipse_distance@0.10"] == pytest.approx(0.0)
    assert metrics["ellipse_mAP"] == pytest.approx(1.0)
    assert "ellipse" in metrics["curves"]
