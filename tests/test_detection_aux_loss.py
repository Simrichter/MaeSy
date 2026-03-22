import torch

from maesy.model.heads.detr_head import DETRHead, DETRHeadConfig
from maesy.training.losses import DetectionLoss


def test_detr_head_emits_aux_outputs_when_enabled():
    config = DETRHeadConfig(
        feature_channels=16,
        embed_dim=8,
        spatial_feature_size=(2, 2),
        num_classes=3,
        num_queries=5,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_heads_encoder=2,
        num_heads_decoder=2,
        enable_auxiliary_losses=True,
    )
    head = DETRHead(config)
    features = torch.randn(2, 16, 2, 2)

    outputs = head(features)

    assert "pred_logits" in outputs
    assert "pred_boxes" in outputs
    assert "aux_outputs" in outputs
    assert len(outputs["aux_outputs"]) == 2
    for aux_output in outputs["aux_outputs"]:
        assert aux_output["pred_logits"].shape == outputs["pred_logits"].shape
        assert aux_output["pred_boxes"].shape == outputs["pred_boxes"].shape


def test_detection_loss_accumulates_auxiliary_outputs():
    loss_fn = DetectionLoss(num_classes=2, aux_loss_coef=0.5, device=torch.device("cpu"))

    predictions = {
        "pred_logits": torch.tensor(
            [[
                [6.0, -6.0, -8.0],
                [-4.0, 4.0, -8.0],
            ]],
            dtype=torch.float32,
        ),
        "pred_boxes": torch.tensor(
            [[
                [0.50, 0.50, 0.20, 0.20],
                [0.10, 0.10, 0.10, 0.10],
            ]],
            dtype=torch.float32,
        ),
        "aux_outputs": [
            {
                "pred_logits": torch.tensor(
                    [[
                        [0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2],
                    ]],
                    dtype=torch.float32,
                ),
                "pred_boxes": torch.tensor(
                    [[
                        [0.20, 0.20, 0.30, 0.30],
                        [0.90, 0.90, 0.05, 0.05],
                    ]],
                    dtype=torch.float32,
                ),
            }
        ],
    }
    targets = [{
        "labels": torch.tensor([0], dtype=torch.long),
        "boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]], dtype=torch.float32),
    }]

    losses = loss_fn(predictions, targets)

    assert losses["loss_aux"] > 0
    assert "loss_ce_aux_0" in losses
    assert "loss_bbox_aux_0" in losses
    assert "loss_giou_aux_0" in losses
    assert torch.isclose(losses["loss_aux_0"], losses["loss_aux"])
    assert torch.isclose(
        losses["loss"],
        losses["loss_ce"] + losses["loss_bbox"] + losses["loss_giou"] + losses["loss_aux"],
    )

    main_only_loss_fn = DetectionLoss(num_classes=2, aux_loss_coef=0.5, device=torch.device("cpu"))
    main_only_losses = main_only_loss_fn(
        {
            "pred_logits": predictions["pred_logits"],
            "pred_boxes": predictions["pred_boxes"],
        },
        targets,
    )
    assert losses["loss"] > main_only_losses["loss"]

