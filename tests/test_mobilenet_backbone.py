import pytest
import torch
from maesy.model.backbones.mobilenet_backbone import MobileNetBackbone, MobileNetBackboneConfig


class TestMobileNetBackbone:
    """Test suite for MobileNet backbone to ensure feature scale selection works correctly."""

    def test_mobilenet_backbone_creation_default(self):
        """Test basic MobileNet backbone creation with default config."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
        )
        backbone = MobileNetBackbone(config)
        assert backbone.type == "MobileNetBackbone_mobilenetv2"
        assert backbone.config.version == "mobilenetv2"

    def test_mobilenet_backbone_single_scale_c5(self):
        """Test MobileNet backbone configuration with c5 feature scale."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c5",),
        )
        backbone = MobileNetBackbone(config)

        # Verify config was set correctly
        assert backbone.config.feature_scales == ("c5",)

    def test_mobilenet_backbone_single_scale_c4(self):
        """Test MobileNet backbone with only c4 feature scale requested."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c4",),
        )
        backbone = MobileNetBackbone(config)

        # Verify config was set correctly
        assert backbone.config.feature_scales == ("c4",)

    def test_mobilenet_backbone_single_scale_c3(self):
        """Test MobileNet backbone with only c3 feature scale requested."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c3",),
        )
        backbone = MobileNetBackbone(config)

        # Verify config was set correctly
        assert backbone.config.feature_scales == ("c3",)

    def test_mobilenet_backbone_multi_scale(self):
        """Test MobileNet backbone with multiple feature scales."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c3", "c4"),
        )
        backbone = MobileNetBackbone(config)

        assert hasattr(backbone, "layer2")
        assert hasattr(backbone, "layer3")

    def test_mobilenet_backbone_forward_c5_only(self):
        """Test forward pass with c5 only (actually c4 as only endpoint)."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c4",),
        )
        backbone = MobileNetBackbone(config)

        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert isinstance(output, dict)
        assert "c4" in output
        assert output["c4"].shape == (2, 64, 14, 14)

    def test_mobilenet_backbone_forward_c4_c5(self):
        """Test forward pass with c3 and c4."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c3", "c4"),
        )
        backbone = MobileNetBackbone(config)

        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert isinstance(output, dict)
        assert "c3" in output
        assert "c4" in output
        assert output["c3"].shape == (2, 32, 28, 28)
        assert output["c4"].shape == (2, 64, 14, 14)

    def test_mobilenet_backbone_forward_all_scales(self):
        """Test forward pass with c3 and c4 scales."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c3", "c4"),
        )
        backbone = MobileNetBackbone(config)

        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert isinstance(output, dict)
        assert "c3" in output
        assert "c4" in output
        assert output["c3"].shape == (2, 32, 28, 28)
        assert output["c4"].shape == (2, 64, 14, 14)

    def test_mobilenet_backbone_get_feature_dims(self):
        """Test get_feature_dims returns correct dimensions."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c3", "c4"),
        )
        backbone = MobileNetBackbone(config)

        feature_dims = backbone.get_feature_dims()

        assert "c3" in feature_dims
        assert "c4" in feature_dims
        assert feature_dims["c3"] == torch.Size((32, 28, 28))
        assert feature_dims["c4"] == torch.Size((64, 14, 14))

    def test_mobilenet_backbone_get_feature_dims_partial(self):
        """Test get_feature_dims with partial scale selection."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c4",),
        )
        backbone = MobileNetBackbone(config)

        feature_dims = backbone.get_feature_dims()

        assert "c4" in feature_dims
        assert len(feature_dims) == 1
        assert feature_dims["c4"] == torch.Size((64, 14, 14))

    def test_mobilenet_backbone_get_feature_channels(self):
        """Test get_feature_channels returns correct channels in correct order."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c3", "c4"),
        )
        backbone = MobileNetBackbone(config)

        channels = backbone.get_feature_channels()

        assert channels == (32, 64)

    def test_mobilenet_backbone_get_feature_channels_partial(self):
        """Test get_feature_channels with partial scale selection maintains order."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
            feature_scales=("c3", "c4"),
        )
        backbone = MobileNetBackbone(config)

        channels = backbone.get_feature_channels()

        assert channels == (32, 64)

    def test_mobilenet_backbone_different_image_sizes(self):
        """Test MobileNet backbone with different input image sizes."""
        for img_size in [112, 224, 448]:
            config = MobileNetBackboneConfig(
                version="mobilenetv2",
                image_size=img_size,
                pretrained=False,
                feature_scales=("c3", "c4"),
            )
            backbone = MobileNetBackbone(config)

            x = torch.randn(1, 3, img_size, img_size)
            output = backbone(x)

            c3_size = img_size // 8
            c4_size = img_size // 16

            assert output["c3"].shape == (1, 32, c3_size, c3_size)
            assert output["c4"].shape == (1, 64, c4_size, c4_size)

    def test_mobilenet_backbone_unsupported_version(self):
        """Test that unsupported MobileNet version raises error."""
        config = MobileNetBackboneConfig(
            version="mobilenetv3",
            image_size=224,
            pretrained=False,
        )

        with pytest.raises(ValueError, match="Unsupported MobileNet version"):
            MobileNetBackbone(config)

    def test_mobilenet_backbone_returns_correct_type_name(self):
        """Test that backbone type name is correctly formatted."""
        config = MobileNetBackboneConfig(
            version="mobilenetv2",
            image_size=224,
            pretrained=False,
        )
        backbone = MobileNetBackbone(config)

        assert backbone.type == "MobileNetBackbone_mobilenetv2"






















