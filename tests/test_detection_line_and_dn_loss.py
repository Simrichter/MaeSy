import torch

from maesy.training.losses import DetectionLoss


def test_detection_loss_computes_line_and_dn_terms_when_enabled():
    loss_fn = DetectionLoss(
        num_classes=3,
        enable_line_detection=True,
        line_class_id=2,
        line_loss_coef=2.0,
        dn_loss_coef=1.0,
        device=torch.device("cpu"),
    )

    predictions = {
        "pred_logits": torch.tensor(
            [[
                [6.0, -6.0, -6.0, -8.0],
                [-6.0, -6.0, 6.0, -8.0],
            ]],
            dtype=torch.float32,
        ),
        "pred_boxes": torch.tensor(
            [[
                [0.40, 0.40, 0.60, 0.60],
                [0.225, 0.225, 0.375, 0.375],
            ]],
            dtype=torch.float32,
        ),
        "pred_lines": torch.tensor(
            [[
                [0.40, 0.40, 0.60, 0.60],
                [0.10, 0.10, 0.20, 0.20],
            ]],
            dtype=torch.float32,
        ),
        "dn_outputs": {
            "pred_logits": torch.tensor(
                [[[6.0, -6.0, -6.0, -8.0]]],
                dtype=torch.float32,
            ),
            "pred_boxes": torch.tensor(
                [[[0.40, 0.40, 0.60, 0.60]]],
                dtype=torch.float32,
            ),
            "pred_lines": torch.tensor(
                [[[0.40, 0.40, 0.60, 0.60]]],
                dtype=torch.float32,
            ),
            "target_labels": torch.tensor([[2]], dtype=torch.long),
            "target_boxes": torch.tensor([[[0.40, 0.40, 0.60, 0.60]]], dtype=torch.float32),
            "target_lines": torch.tensor([[[0.40, 0.40, 0.60, 0.60]]], dtype=torch.float32),
            "target_valid_mask": torch.tensor([[True]]),
        },
    }
    targets = [{
        "labels": torch.tensor([0, 2], dtype=torch.long),
        "boxes": torch.tensor([[0.40, 0.40, 0.60, 0.60]], dtype=torch.float32),
        "line_points": torch.tensor([[0.40, 0.40, 0.60, 0.60]], dtype=torch.float32),
    }]

    losses = loss_fn(predictions, targets)

    assert "loss_line" in losses
    assert "loss_dn" in losses
    assert losses["loss_line"] >= 0
    assert losses["loss_dn"] >= 0
    assert torch.isclose(
        losses["loss"],
        losses["loss_ce"] + losses["loss_bbox"] + losses["loss_giou"] + losses["loss_line"] + losses["loss_aux"] + losses["loss_dn"],
    )


def test_detection_loss_normalizes_bbox_loss_by_box_matches_only():
    loss_fn = DetectionLoss(
        num_classes=3,
        bbox_loss_coef=1.0,
        class_loss_coef=0.0,
        giou_loss_coef=0.0,
        line_loss_coef=0.0,
        enable_line_detection=True,
        line_class_id=2,
        device=torch.device("cpu"),
    )

    predictions = {
        "pred_logits": torch.tensor(
            [[
                [6.0, -6.0, -6.0, -8.0],
                [-6.0, -6.0, 6.0, -8.0],
            ]],
            dtype=torch.float32,
        ),
        "pred_boxes": torch.tensor(
            [[
                [0.40, 0.40, 0.80, 0.80],
                [0.15, 0.15, 0.25, 0.25],
            ]],
            dtype=torch.float32,
        ),
        "pred_lines": torch.tensor(
            [[
                [0.30, 0.30, 0.70, 0.70],
                [0.40, 0.40, 0.60, 0.60],
            ]],
            dtype=torch.float32,
        ),
    }
    targets = [{
        "labels": torch.tensor([0, 2], dtype=torch.long),
        "boxes": torch.tensor([[0.40, 0.40, 0.60, 0.60]], dtype=torch.float32),
        "line_points": torch.tensor([[0.40, 0.40, 0.60, 0.60]], dtype=torch.float32),
    }]

    losses = loss_fn(predictions, targets)

    # L1 sum over the single bbox pair: |0.40-0.40|+|0.40-0.40|+|0.80-0.60|+|0.80-0.60| = 0.4
    assert torch.isclose(losses["loss_bbox"], torch.tensor(0.4), atol=1e-6)


def test_detection_loss_computes_line_angle_and_log_length_components():
    loss_fn = DetectionLoss(
        num_classes=3,
        bbox_loss_coef=0.0,
        class_loss_coef=0.0,
        giou_loss_coef=0.0,
        line_loss_coef=1.0,
        line_angle_loss_coef=2.0,
        line_length_loss_coef=3.0,
        enable_line_detection=True,
        line_class_id=2,
        device=torch.device("cpu"),
    )

    predictions = {
        "pred_logits": torch.tensor([[[0.0, 0.0, 6.0, -8.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        "pred_lines": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    }
    targets = [{
        "labels": torch.tensor([2], dtype=torch.long),
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "line_points": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    }]

    losses = loss_fn(predictions, targets)

    assert torch.isclose(losses["loss_line"], torch.tensor(4.0), atol=1e-6)
