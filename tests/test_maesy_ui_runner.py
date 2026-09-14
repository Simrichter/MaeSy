import sys
from types import SimpleNamespace

import pytest

from maesy_ui import runner


def test_runner_invokes_registered_module_dispatcher(monkeypatch):
    captured = {}

    def dispatch_command(argv):
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: SimpleNamespace(dispatch_command=dispatch_command),
    )
    monkeypatch.setattr(sys, "argv", ["runner", "example.cli", "arg one", "--flag"])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 7
    assert captured["argv"] == ["arg one", "--flag"]
