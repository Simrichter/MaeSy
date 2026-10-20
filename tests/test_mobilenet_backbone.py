import pytest
import torch

from _maesy_core.model.backbones.mobilenet_backbone import (
    MobileNetBackbone,
    MobileNetBackboneConfig,
)


def test_mobilenet_v2_all_scales_match_feature_dimensions():
    backbone = MobileNetBackbone(
        MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c3", "c4", "c5"),
        )
    )

    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 224, 224))

    assert {scale: tuple(feature.shape) for scale, feature in features.items()} == {
        "c3": (1, 32, 28, 28),
        "c4": (1, 96, 14, 14),
        "c5": (1, 1280, 7, 7),
    }
    assert backbone.get_feature_dims() == {
        "c3": torch.Size((32, 28, 28)),
        "c4": torch.Size((96, 14, 14)),
        "c5": torch.Size((1280, 7, 7)),
    }
    assert backbone.get_feature_channels() == (32, 96, 1280)


def test_mobilenet_v2_returns_only_requested_scales():
    backbone = MobileNetBackbone(
        MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c4",),
        )
    )

    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 224, 224))

    assert tuple(features) == ("c4",)
    assert tuple(features["c4"].shape) == (1, 96, 14, 14)
    assert backbone.get_feature_dims() == {"c4": torch.Size((96, 14, 14))}
    assert backbone.get_feature_channels() == (96,)


@pytest.mark.parametrize("version", ["mobilenetv3", "mobilenetv2-extra", "unknown"])
def test_unsupported_mobilenet_version_is_rejected(version):
    config = MobileNetBackboneConfig(version=version, pretrained=False)

    with pytest.raises(ValueError, match="Unsupported MobileNet version"):
        MobileNetBackbone(config)
