"""Registration of argparse-based CLI applications exposed by MaeSy UI."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .argparse_adapter import ArgumentField


ArchitectureTree: TypeAlias = dict[str, "ArchitectureTree | None"]
FieldHintKey: TypeAlias = tuple[tuple[str, ...], str]


@dataclass(frozen=True)
class FieldHint:
    kind: Literal["model-selector"]
    choices_provider: Callable[[], ArchitectureTree]


@dataclass(frozen=True)
class CliTarget:
    name: str
    display_name: str
    parser_provider: Callable[[], argparse.ArgumentParser]
    dispatcher: Callable[[list[str]], int | None]
    dispatch_module: str
    field_hints: Mapping[FieldHintKey | str, FieldHint] = field(default_factory=dict)

    def parser(self) -> argparse.ArgumentParser:
        return self.parser_provider()

    def field_hint(
        self,
        field_or_path: "ArgumentField | tuple[str, ...]",
        destination: str | None = None,
    ) -> FieldHint | None:
        """Find a hint by field, or by its command path and argparse destination.

        A destination-only lookup remains supported for root-level third-party CLIs.
        """
        if hasattr(field_or_path, "path") and hasattr(field_or_path, "action"):
            path = field_or_path.path
            dest = field_or_path.action.dest
        elif isinstance(field_or_path, str) and destination is None:
            path = ()
            dest = field_or_path
        else:
            path = field_or_path
            if destination is None:
                raise TypeError("destination is required when looking up a command path")
            dest = destination
        return self.field_hints.get((path, dest)) or self.field_hints.get(dest)


def model_architecture_tree() -> ArchitectureTree:
    """Return cfg YAML architectures as a nested directory hierarchy."""
    cfg_dir = Path(__file__).resolve().parents[1] / "cfg"
    tree: ArchitectureTree = {}
    for yaml_path in sorted(cfg_dir.rglob("*.yaml")):
        relative = yaml_path.relative_to(cfg_dir).with_suffix("")
        node = tree
        for part in relative.parts[:-1]:
            child = node.setdefault(part, {})
            assert isinstance(child, dict)
            node = child
        node[relative.name] = None
    return tree


def model_architectures() -> tuple[str, ...]:
    """Return canonical slash-separated IDs for all discovered architectures."""
    def flatten(tree: ArchitectureTree, prefix: tuple[str, ...] = ()) -> list[str]:
        ids: list[str] = []
        for name, child in tree.items():
            path = (*prefix, name)
            if child is None:
                ids.append("/".join(path))
            else:
                ids.extend(flatten(child, path))
        return ids

    return tuple(flatten(model_architecture_tree()))


_targets: dict[str, CliTarget] = {}
_defaults_registered = False


def register_cli(
    name: str,
    display_name: str,
    parser_provider: Callable[[], argparse.ArgumentParser],
    dispatch_module: str,
    field_hints: Mapping[str, FieldHint] | None = None,
) -> Callable[[Callable[[list[str]], int | None]], Callable[[list[str]], int | None]]:
    """Decorate an existing argv dispatcher and make it available to the UI."""

    def decorator(
        dispatcher: Callable[[list[str]], int | None],
    ) -> Callable[[list[str]], int | None]:
        if name in _targets:
            raise ValueError(f"A UI target named '{name}' is already registered.")
        _targets[name] = CliTarget(
            name,
            display_name,
            parser_provider,
            dispatcher,
            dispatch_module,
            field_hints or {},
        )
        return dispatcher

    return decorator


def registered_targets() -> tuple[CliTarget, ...]:
    register_default_targets()
    return tuple(_targets.values())


def get_target(name: str) -> CliTarget:
    register_default_targets()
    try:
        return _targets[name]
    except KeyError as exc:
        raise KeyError(f"No UI target named '{name}' is registered.") from exc


def register_default_targets() -> None:
    """Register the existing standalone CLIs without changing their parsers."""
    global _defaults_registered
    if _defaults_registered:
        return

    from clusterdevil.command_line import dispatch_command as clusterdevil_dispatch
    from clusterdevil.command_line import parser as clusterdevil_parser
    from maesy.command_line import dispatch_command as maesy_dispatch
    from maesy.command_line import parser as maesy_parser

    register_cli(
        "clusterdevil",
        "ClusterDevil",
        lambda: clusterdevil_parser,
        "clusterdevil.command_line",
        {((), "model"): FieldHint("model-selector", model_architecture_tree)},
    )(clusterdevil_dispatch)
    register_cli(
        "maesy",
        "MaeSy",
        lambda: maesy_parser,
        "maesy.command_line",
        {
            (("train", "od"), "model"): FieldHint("model-selector", model_architecture_tree),
            (("train", "mae"), "model"): FieldHint("model-selector", model_architecture_tree),
            (("export",), "model"): FieldHint("model-selector", model_architecture_tree),
        },
    )(maesy_dispatch)
    _defaults_registered = True
