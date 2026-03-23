import sys

import pytest

import maesy.command_line as command_line
import maesy.evaluation.cli_evaluate as cli_evaluate
import maesy.training.cli_train as cli_train


def test_command_line_dispatches_train_module(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["module"] = args.module
        captured["mode"] = args.mode
        captured["dataset"] = args.dataset

    monkeypatch.setattr(cli_train, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["maesy", "train", "mae", "--dataset", "/tmp/dataset"],
    )

    command_line.main()

    assert captured == {
        "module": "train",
        "mode": "mae",
        "dataset": "/tmp/dataset",
    }


def test_command_line_unknown_module_prints_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["maesy", "unknown"])

    with pytest.raises(SystemExit):
        command_line.main()

    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_command_line_train_od_default_detector_is_rt_detr(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["mode"] = args.mode
        captured["detector"] = args.detector

    monkeypatch.setattr(cli_train, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["maesy", "train", "od", "--dataset", "/tmp/dataset"],
    )

    command_line.main()

    assert captured == {"mode": "od", "detector": "rt_detr"}


def test_command_line_evaluate_infer_accepts_detector_override(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["command"] = args.command
        captured["detector"] = args.detector

    monkeypatch.setattr(cli_evaluate, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["maesy", "evaluate", "infer", "/tmp/images", "/tmp/checkpoint.pth", "--detector", "detr"],
    )

    command_line.main()

    assert captured == {"command": "infer", "detector": "detr"}


