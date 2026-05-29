import torch

from maesy.model import RTDETR, RTDETRConfig


def test_rt_detr_forward_matches_detection_contract():
    config = RTDETRConfig(
        image_size=64,
        backbone_version="resnet18",
        backbone_pretrained=False,
        num_classes=3,
        num_queries=20,
        embed_dim=64,
        num_decoder_layers=3,
        decoder_num_heads=8,
        hidden_dim_out_layers=64,
        enable_auxiliary_losses=True,
    )
    model = RTDETR(config)

    images = torch.randn(2, 3, 64, 64)
    outputs = model(images)

    assert "pred_logits" in outputs
    assert "pred_boxes" in outputs
    assert "enc_outputs" in outputs
    assert outputs["pred_logits"].shape == (2, 20, 4)
    assert outputs["pred_boxes"].shape == (2, 20, 4)
    assert outputs["enc_outputs"]["pred_logits"].shape == (2, 20, 4)
    assert outputs["enc_outputs"]["pred_boxes"].shape == (2, 20, 4)
    assert torch.all(outputs["pred_boxes"] >= 0.0)
    assert torch.all(outputs["pred_boxes"] <= 1.0)

    assert "aux_outputs" in outputs
    assert len(outputs["aux_outputs"]) == 2
    for aux in outputs["aux_outputs"]:
        assert aux["pred_logits"].shape == (2, 20, 4)
        assert aux["pred_boxes"].shape == (2, 20, 4)


def test_rt_detr_optional_line_head_outputs_pred_lines():
    config = RTDETRConfig(
        image_size=64,
        backbone_version="resnet18",
        backbone_pretrained=False,
        num_classes=3,
        num_queries=12,
        embed_dim=64,
        num_decoder_layers=2,
        decoder_num_heads=8,
        hidden_dim_out_layers=64,
        enable_line_detection=True,
        line_class_id=2,
    )
    model = RTDETR(config)

    outputs = model(torch.randn(1, 3, 64, 64))
    assert "pred_lines" in outputs
    assert outputs["pred_lines"].shape == (1, 12, 4)


def test_rt_detr_denoising_outputs_emitted_only_in_training_with_targets():
    config = RTDETRConfig(
        image_size=64,
        backbone_version="resnet18",
        backbone_pretrained=False,
        num_classes=3,
        num_queries=10,
        embed_dim=64,
        num_decoder_layers=2,
        decoder_num_heads=8,
        hidden_dim_out_layers=64,
        enable_denoising=True,
        denoising_num_queries=6,
    )
    model = RTDETR(config)
    model.train()

    images = torch.randn(1, 3, 64, 64)
    targets = [{
        "labels": torch.tensor([1], dtype=torch.long),
        "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
        "line_points": torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=torch.float32),
    }]

    outputs = model(images, targets=targets)
    assert "dn_outputs" in outputs
    assert outputs["dn_outputs"]["pred_logits"].shape[1] == 6


