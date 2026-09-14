from _maesy_core.model.model_tools.model_factory import (
    architecture_paths,
    resolve_architecture_path,
)


def test_recursive_architecture_resolution_supports_canonical_and_legacy_ids():
    architectures = architecture_paths()

    assert "rt-detr/rt-detr6" in architectures
    canonical = resolve_architecture_path("rt-detr/rt-detr6")
    legacy = resolve_architecture_path("rt-detr6")
    assert canonical == legacy
    assert canonical is not None
    assert canonical.as_posix().endswith("cfg/rt-detr/rt-detr6.yaml")
