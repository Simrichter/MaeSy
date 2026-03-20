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
        [[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]],
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
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
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

