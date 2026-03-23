import torch

from maesy.model import RTDETR, RTDETRConfig


def test_rt_detr_forward_matches_detection_contract():
    config = RTDETRConfig(
        image_size=64,
        resnet_version="resnet18",
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
    assert outputs["pred_logits"].shape == (2, 20, 4)
    assert outputs["pred_boxes"].shape == (2, 20, 4)
    assert torch.all(outputs["pred_boxes"] >= 0.0)
    assert torch.all(outputs["pred_boxes"] <= 1.0)

    assert "aux_outputs" in outputs
    assert len(outputs["aux_outputs"]) == 2
    for aux in outputs["aux_outputs"]:
        assert aux["pred_logits"].shape == (2, 20, 4)
        assert aux["pred_boxes"].shape == (2, 20, 4)

