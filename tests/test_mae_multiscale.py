import torch
import pytest

from _maesy_core.model import MaeMultiscaleConfig, MaskedAutoencoderMultiscale
from _maesy_core.model.heads import MaeMultiscaleDecoder, MaeMultiscaleDecoderConfig


def test_mae_multiscale_decoder_returns_patch_predictions():
    cfg = MaeMultiscaleDecoderConfig(
        embed_dim=64,
        num_patches=16,
        patch_size=16,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
        num_layers=2,
        in_channels=3,
        feature_dims={"c3": 32, "c4": 64, "c5": 96},
        feature_scales=("c3", "c4", "c5"),
        use_skip_connections=True,
        skip_scales=("c3", "c4"),
        window_size=4,
    )
    decoder = MaeMultiscaleDecoder(cfg)
    features = {
        "c3": torch.randn(2, 32, 8, 8),
        "c4": torch.randn(2, 64, 4, 4),
        "c5": torch.randn(2, 96, 2, 2),
    }

    out = decoder(features)

    assert out.shape == (2, 16, 16 * 16 * 3)


def test_mae_multiscale_model_forward_contract():
    cfg = MaeMultiscaleConfig(
        image_size=64,
        patch_size=16,
        in_channels=3,
        backbone_version="mobilenetv2",
        backbone_pretrained=False,
        feature_scales=("c3", "c4", "c5"),
        decoder_embed_dim=64,
        decoder_num_layers=1,
        decoder_num_heads=4,
        decoder_mlp_ratio=2.0,
        decoder_dropout=0.0,
        decoder_attention_dropout=0.0,
        decoder_window_size=4,
    )
    model = MaskedAutoencoderMultiscale(cfg)
    images = torch.randn(2, 3, 64, 64)

    out = model(images)

    assert out.shape == (2, 16, 16 * 16 * 3)


def test_decoder_rejects_non_divisible_window_grids():
    cfg = MaeMultiscaleDecoderConfig(
        embed_dim=32,
        num_patches=9,
        patch_size=16,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
        num_layers=1,
        in_channels=3,
        feature_dims={"c3": 16, "c4": 16, "c5": 16},
        feature_scales=("c3", "c4", "c5"),
        window_size=2,
    )
    decoder = MaeMultiscaleDecoder(cfg)
    features = {
        "c3": torch.randn(1, 16, 6, 6),
        "c4": torch.randn(1, 16, 4, 4),
        "c5": torch.randn(1, 16, 3, 3),
    }

    with pytest.raises(ValueError):
        decoder(features)

