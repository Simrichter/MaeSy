"""MaeSy - Vision Transformer framework for object detection."""

__version__ = "0.1.0"

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "DatasetManager",
    "ModelConfig",
]

_LAZY_ATTRS = { # Lazy imports to increase "-h" response time in cli
    "DatasetManager": ("maesy.dataset", "DatasetManager"),
    "ObjectDetectionDataset": ("maesy.dataset", "ObjectDetectionDataset"),
    "ModelConfig": ("maesy.model", "ModelConfig"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value  # Cache for future lookups.
        return value
    raise AttributeError(f"module 'maesy' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))


if TYPE_CHECKING:
    from _maesy_core.dataset import DatasetManager
    from _maesy_core.model import ModelConfig
