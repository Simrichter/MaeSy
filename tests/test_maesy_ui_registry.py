from maesy_ui.registry import (
    get_target,
    model_architecture_tree,
    model_architectures,
    registered_targets,
)


def test_default_ui_targets_expose_existing_parsers_and_dispatchers():
    targets = {target.name: target for target in registered_targets()}

    assert {"clusterdevil", "maesy"}.issubset(targets)
    assert get_target("clusterdevil").parser().prog
    clusterdevil = get_target("clusterdevil")
    assert callable(clusterdevil.dispatcher)
    assert clusterdevil.dispatch_module == "clusterdevil.command_line"
    assert clusterdevil.field_hint((), "model") is not None
    assert "train" in get_target("maesy").parser()._subparsers._group_actions[0].choices


def test_model_architecture_tree_reflects_available_configuration_ids():
    tree = model_architecture_tree()
    architectures = model_architectures()

    assert architectures
    assert set(architectures) == set(__import__("_maesy_core.model.model_tools.model_factory", fromlist=["architecture_paths"]).architecture_paths())
    assert all(identifier in architectures for identifier in _flatten_tree(tree))


def _flatten_tree(tree, prefix=""):
    identifiers = []
    for name, value in tree.items():
        identifier = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            identifiers.extend(_flatten_tree(value, identifier))
        else:
            identifiers.append(identifier)
    return identifiers


def test_maesy_registers_model_selectors_for_supported_arguments():
    maesy = get_target("maesy")

    assert maesy.field_hint(("train", "od"), "model") is not None
    assert maesy.field_hint(("train", "mae"), "model") is not None
    assert maesy.field_hint(("export",), "model") is not None
    assert maesy.field_hint(("evaluate", "infer"), "checkpoint") is None
