"""Subprocess entry point that invokes an existing CLI argv dispatcher."""

from __future__ import annotations

import importlib
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m maesy_ui.runner <dispatch-module> [arguments...]")

    module = importlib.import_module(sys.argv[1])
    dispatch_command = getattr(module, "dispatch_command")
    result = dispatch_command(sys.argv[2:])
    raise SystemExit(result or 0)


if __name__ == "__main__":
    main()
