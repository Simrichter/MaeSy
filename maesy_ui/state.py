"""Persistent UI state so the window remembers previously entered inputs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .argparse_adapter import command_choices


def state_file() -> Path:
    """Return the path of the UI state file."""
    return Path(os.environ.get("MAESY_UI_STATE_FILE", "~/.maesy_ui_state.json")).expanduser()


def load_state(path: Path | None = None) -> dict[str, Any]:
    """Load the saved UI state, tolerating missing or corrupt files."""
    path = path or state_file()
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: Mapping[str, Any], path: Path | None = None) -> None:
    """Persist the UI state atomically. Write failures are ignored."""
    path = path or state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(dict(state), file, indent=2, default=str)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    except (OSError, TypeError, ValueError):
        pass


def encode_field_key(path: tuple[str, ...], dest: str) -> str:
    return "/".join((*path, dest))


def decode_field_key(encoded: str) -> tuple[tuple[str, ...], str] | None:
    parts = [part for part in encoded.split("/") if part]
    if not parts:
        return None
    return tuple(parts[:-1]), parts[-1]


def encode_subcommand_key(prefix: tuple[str, ...]) -> str:
    return "/".join(prefix)


def decode_subcommand_key(encoded: str) -> tuple[str, ...]:
    return tuple(part for part in encoded.split("/") if part)


def target_state(
    subcommand_values: Mapping[tuple[str, ...], str],
    form_values: Mapping[tuple[tuple[str, ...], str], Any],
) -> dict[str, Any]:
    """Serialize the in-memory UI selections for one CLI target."""
    return {
        "subcommands": {
            encode_subcommand_key(prefix): value for prefix, value in subcommand_values.items()
        },
        "values": {
            encode_field_key(path, dest): value for (path, dest), value in form_values.items()
        },
    }


def decode_target_state(
    raw: Any, parser: argparse.ArgumentParser
) -> tuple[dict[tuple[str, ...], str], dict[tuple[tuple[str, ...], str], Any]]:
    """Decode a stored target state, validating subcommand selections against *parser*."""
    subcommands: dict[tuple[str, ...], str] = {}
    values: dict[tuple[tuple[str, ...], str], Any] = {}
    if not isinstance(raw, dict):
        return subcommands, values

    saved_subcommands = raw.get("subcommands")
    if isinstance(saved_subcommands, dict):
        cursor = parser
        path: list[str] = []
        while choices := command_choices(cursor):
            prefix = tuple(path)
            saved = saved_subcommands.get(encode_subcommand_key(prefix))
            selection = saved if saved in choices else next(iter(choices))
            subcommands[prefix] = selection
            path.append(selection)
            cursor = choices[selection]

    saved_values = raw.get("values")
    if isinstance(saved_values, dict):
        for encoded, value in saved_values.items():
            if not isinstance(encoded, str):
                continue
            key = decode_field_key(encoded)
            if key is not None:
                values[key] = value

    return subcommands, values
