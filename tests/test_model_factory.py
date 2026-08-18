import pytest

from _maesy_core.model import DETRConfig, RTDETRConfig
from _maesy_core.model.model_tools.model_factory import create_model


def test_model_factory_creates_detr_and_rt_detr_models():
    detr = create_model(
        "detr",
        DETRConfig(
            image_size=64,
            resnet_version="resnet18",
            embed_dim=64,
            num_classes=3,
            num_queries=10,
            num_encoder_layers=1,
            num_decoder_layers=1,
            encoder_num_heads=4,
            decoder_num_heads=4,
            hidden_dim_out_layers=64,
        ),
    )
    rt_detr = create_model(
        "rt_detr",
        RTDETRConfig(
            image_size=64,
            resnet_version="resnet18",
            backbone_pretrained=False,
            embed_dim=64,
            num_classes=3,
            num_queries=10,
            num_decoder_layers=2,
            decoder_num_heads=8,
            hidden_dim_out_layers=64,
        ),
    )

    assert detr.__class__.__name__ == "DETR"
    assert rt_detr.__class__.__name__ == "RTDETR"


def test_model_factory_rejects_wrong_config_type():
    with pytest.raises(TypeError, match="expects config type"):
        create_model("rt_detr", DETRConfig())

