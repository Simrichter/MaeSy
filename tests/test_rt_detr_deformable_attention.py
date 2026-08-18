import torch

from _maesy_core.model.heads.rt_detr_head import (
    MultiScaleDeformableAttention,
    RTDETRHead,
    RTDETRHeadConfig,
)


def test_multiscale_deformable_attention_forward_shape():
    attention = MultiScaleDeformableAttention(embed_dim=32, num_heads=4, num_levels=3, num_points=2)

    batch_size = 2
    num_queries = 6
    spatial_shapes = [(8, 8), (4, 4), (2, 2)]
    value_len = sum(h * w for h, w in spatial_shapes)

    query = torch.randn(batch_size, num_queries, 32)
    value = torch.randn(batch_size, value_len, 32)
    reference_points = torch.rand(batch_size, num_queries, 4)

    output = attention(
        query=query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
    )

    assert output.shape == (batch_size, num_queries, 32)


def test_rt_detr_head_reuses_positional_encoding_cache_for_same_shapes():
    head = RTDETRHead(
        RTDETRHeadConfig(
            feature_channels=(16, 32, 64),
            num_classes=3,
            num_queries=8,
            embed_dim=32,
            num_decoder_layers=2,
            decoder_num_heads=4,
            num_deformable_points=2,
        )
    )

    features = {
        "c3": torch.randn(1, 16, 8, 8),
        "c4": torch.randn(1, 32, 4, 4),
        "c5": torch.randn(1, 64, 2, 2),
    }

    _ = head(features)
    cache_size_after_first = len(head._pos_encoding_cache)
    _ = head(features)
    cache_size_after_second = len(head._pos_encoding_cache)

    assert cache_size_after_first > 0
    assert cache_size_after_first == cache_size_after_second

