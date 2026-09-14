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


def test_model_architecture_tree_preserves_cfg_subdirectories():
    tree = model_architecture_tree()

    assert "rt-detr" in tree
    assert isinstance(tree["rt-detr"], dict)
    assert "rt-detr6" in tree["rt-detr"]
    assert "rt-detr/rt-detr6" in model_architectures()


def test_maesy_registers_model_selectors_for_all_architecture_arguments():
    maesy = get_target("maesy")

    assert maesy.field_hint(("train", "od"), "model") is not None
    assert maesy.field_hint(("train", "mae"), "model") is not None
    assert maesy.field_hint(("export",), "model") is not None
    assert maesy.field_hint(("evaluate", "infer"), "checkpoint") is None
