"""Model utilities with lazy exports to keep model imports acyclic."""

from importlib import import_module

_EXPORTS = {
    "CheckpointHandler": ".checkpoint_handler",
    "check_finite": ".debug_helpers",
    "log_print": ".debug_helpers",
    "replace_bn_with_frozenbn": ".layer_manipulations",
    "architecture_paths": ".model_factory",
    "create_model": ".model_factory",
    "create_model_from_checkpoint": ".model_factory",
    "create_model_from_config": ".model_factory",
    "create_model_from_dict": ".model_factory",
    "known_architectures": ".model_factory",
    "read_yaml": ".model_factory",
    "resolve_architecture_path": ".model_factory",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
