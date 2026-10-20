from _maesy_core.model.model_tools.model_factory import (
    architecture_paths,
    create_model_from_dict,
    resolve_architecture_path,
)


def test_discovered_architecture_ids_resolve_to_existing_yaml_files():
    architectures = architecture_paths()

    assert architectures
    for architecture_id, path in architectures.items():
        assert path.is_file()
        assert path.suffix == ".yaml"
        assert resolve_architecture_path(architecture_id) == path


def test_unique_legacy_basename_resolves_and_unknown_id_does_not():
    architectures = architecture_paths()
    identifier, path = next(iter(architectures.items()))
    basename = path.stem

    assert resolve_architecture_path(basename) == path
    assert resolve_architecture_path("not-a-real-architecture") is None


def test_checkpoint_model_factory_reconstructs_vit_detector():
    from _maesy_core.model import ViTDetector, ViTDetectorConfig

    config = ViTDetectorConfig(
        image_size=32,
        patch_size=8,
        embed_dim=16,
        num_layers=1,
        num_heads=4,
        num_decoder_layers=1,
        decoder_num_heads=4,
        num_classes=3,
        num_queries=4,
        hidden_dim=32,
    )
    model = create_model_from_dict(config.__dict__)

    assert config.type == "ViTDetector"
    assert isinstance(model, ViTDetector)
    assert model.config.num_classes == 3
    assert model.head.class_embed.out_features == 4
    assert model.config.num_patches == 16


def test_vit_detector_component_config_types_are_checkpoint_serializable():
    from dataclasses import fields

    from _maesy_core.model import ViTDetectorConfig
    from _maesy_core.model.backbones.transformer_backbone import TransformerBackboneConfig
    from _maesy_core.model.heads.vit_detection_head import ViTDetectionHeadConfig

    assert "type" in {field.name for field in fields(ViTDetectorConfig)}
    assert "type" in {field.name for field in fields(TransformerBackboneConfig)}
    assert "type" in {field.name for field in fields(ViTDetectionHeadConfig)}
    assert TransformerBackboneConfig().type == "TransformerBackbone"
    assert ViTDetectionHeadConfig().type == "ViTDetectionHead"


def test_vit_component_configs_preserve_positional_parameter_order():
    from _maesy_core.model.backbones.transformer_backbone import TransformerBackboneConfig
    from _maesy_core.model.heads.vit_detection_head import ViTDetectionHeadConfig

    backbone = TransformerBackboneConfig(224, 16, 3)
    head = ViTDetectionHeadConfig(128, 3)

    assert (backbone.image_size, backbone.patch_size, backbone.in_channels) == (224, 16, 3)
    assert (head.embed_dim, head.num_classes) == (128, 3)
