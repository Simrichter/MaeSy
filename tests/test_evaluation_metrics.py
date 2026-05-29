import torch

from maesy.evaluation.metrics import (
    compute_detection_metrics,
    decode_detr_predictions,
    prepare_targets_for_detection_metrics,
)


def test_decode_detr_predictions_filters_no_object_queries():
    # classes: 0,1 plus no-object at index 2
    pred_logits = torch.tensor(
        [[[4.0, 0.1, -1.0], [0.1, 0.2, 3.5], [0.2, 4.0, -0.1]]],
        dtype=torch.float32,
    )
    pred_boxes = torch.tensor(
        [[[0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.6, 0.6], [0.15, 0.15, 0.25, 0.25]]],
        dtype=torch.float32,
    )

    decoded = decode_detr_predictions(pred_logits, pred_boxes, no_object_class=2)

    assert len(decoded) == 1
    assert decoded[0]["boxes"].shape[0] == 2
    assert set(decoded[0]["labels"].tolist()) == {0, 1}


def test_compute_detection_metrics_reports_map_fields():
    predictions = [
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
            "scores": torch.tensor([0.9], dtype=torch.float32),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }
    ]
    prepared_targets = prepare_targets_for_detection_metrics(targets)

    metrics = compute_detection_metrics(predictions, prepared_targets, num_classes=2)

    assert 0.0 <= metrics["mAP50"] <= 1.0
    assert 0.0 <= metrics["mAP50_95"] <= 1.0
    assert "precision50" in metrics
    assert "recall50" in metrics
    assert "f1_50" in metrics
    assert "AP50_class_0" in metrics
    assert "curves" in metrics
    assert "bbox" in metrics["curves"]


def test_prepare_and_decode_support_mixed_bbox_and_line_targets():
    line_class_id = 2
    targets = [
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
            "labels": torch.tensor([0, line_class_id], dtype=torch.long),
            "line_points": torch.tensor([[0.1, 0.1, 0.8, 0.8]], dtype=torch.float32),
        }
    ]
    prepared = prepare_targets_for_detection_metrics(targets, line_class_id=line_class_id)

    assert prepared[0]["boxes"].shape == (1, 4)
    assert prepared[0]["labels"].tolist() == [0]
    assert prepared[0]["line_points"].shape == (1, 4)
    assert prepared[0]["line_labels"].tolist() == [line_class_id]

    pred_logits = torch.tensor(
        [[[6.0, -6.0, -6.0, -8.0], [-6.0, -6.0, 6.0, -8.0]]],
        dtype=torch.float32,
    )
    pred_boxes = torch.tensor(
        [[[0.4, 0.4, 0.6, 0.6], [0.1, 0.1, 0.3, 0.3]]],
        dtype=torch.float32,
    )
    pred_lines = torch.tensor(
        [[[0.2, 0.2, 0.3, 0.3], [0.1, 0.1, 0.8, 0.8]]],
        dtype=torch.float32,
    )

    decoded = decode_detr_predictions(
        pred_logits,
        pred_boxes,
        pred_lines=pred_lines,
        line_class_id=line_class_id,
        no_object_class=3,
    )

    assert decoded[0]["boxes"].shape[0] == 1
    assert decoded[0]["labels"].tolist() == [0]
    assert decoded[0]["line_points"].shape[0] == 1
    assert decoded[0]["line_labels"].tolist() == [line_class_id]


def test_compute_detection_metrics_includes_line_metrics_when_enabled():
    predictions = [
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
            "scores": torch.tensor([0.95], dtype=torch.float32),
            "line_points": torch.tensor([[0.1, 0.1, 0.8, 0.8]], dtype=torch.float32),
            "line_labels": torch.tensor([2], dtype=torch.long),
            "line_scores": torch.tensor([0.9], dtype=torch.float32),
        }
    ]
    raw_targets = [
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
            "labels": torch.tensor([0, 2], dtype=torch.long),
            "line_points": torch.tensor([[0.1, 0.1, 0.8, 0.8]], dtype=torch.float32),
        }
    ]
    targets = prepare_targets_for_detection_metrics(raw_targets, line_class_id=2)

    metrics = compute_detection_metrics(
        predictions,
        targets,
        num_classes=3,
        line_class_id=2,
        line_distance_thresholds=(0.05,),
    )

    assert "line_precision@0.05" in metrics
    assert "line_recall@0.05" in metrics
    assert "line_f1@0.05" in metrics
    assert "line_AP@0.05" in metrics
    assert "line_endpoint_error@0.05" in metrics
    assert "line_mAP" in metrics


def test_compute_detection_metrics_includes_ellipse_metrics_and_curves():
    predictions = [
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
            "scores": torch.tensor([0.95], dtype=torch.float32),
            "ellipses": torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0, 1.0]], dtype=torch.float32),
            "ellipse_labels": torch.tensor([3], dtype=torch.long),
            "ellipse_scores": torch.tensor([0.9], dtype=torch.float32),
        }
    ]
    raw_targets = [
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
            "labels": torch.tensor([0, 3], dtype=torch.long),
            "ellipses": torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0, 1.0]], dtype=torch.float32),
        }
    ]
    targets = prepare_targets_for_detection_metrics(raw_targets, ellipse_class_id=3)

    metrics = compute_detection_metrics(
        predictions,
        targets,
        num_classes=4,
        ellipse_class_id=3,
        ellipse_distance_thresholds=(0.1,),
    )

    assert "ellipse_precision@0.10" in metrics
    assert "ellipse_recall@0.10" in metrics
    assert "ellipse_f1@0.10" in metrics
    assert "ellipse_AP@0.10" in metrics
    assert "ellipse_distance@0.10" in metrics
    assert "ellipse_mAP" in metrics
    assert "curves" in metrics
    assert "ellipse" in metrics["curves"]
