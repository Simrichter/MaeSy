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


def test_command_line_train_od_dispatches_supported_arguments(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["mode"] = args.mode
        captured["model"] = args.model
        captured["dataset"] = args.dataset
        captured["enable_denoising"] = args.enable_denoising
        captured["dn_queries"] = args.dn_queries
        captured["enable_line_detection"] = args.enable_line_detection

    monkeypatch.setattr(cli_train, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "maesy", "train", "od", "rt-detr", "--dataset", "/tmp/dataset",
            "--enable-denoising", "--dn-queries", "7", "--enable-line-detection",
        ],
    )

    command_line.main()

    assert captured == {
        "mode": "od",
        "model": "rt-detr",
        "dataset": ["/tmp/dataset"],
        "enable_denoising": True,
        "dn_queries": 7,
        "enable_line_detection": True,
    }


def test_command_line_evaluate_infer_dispatches_supported_arguments(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["command"] = args.command
        captured["imgpath"] = args.imgpath
        captured["checkpoint"] = args.checkpoint
        captured["split"] = args.split
        captured["confidence"] = args.confidence
        captured["device"] = args.device

    monkeypatch.setattr(cli_evaluate, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "maesy", "evaluate", "infer", "/tmp/images", "/tmp/checkpoint.pth",
            "--split", "val", "--confidence", "0.4", "--device", "cpu",
        ],
    )

    command_line.main()

    assert captured == {
        "command": "infer",
        "imgpath": "/tmp/images",
        "checkpoint": "/tmp/checkpoint.pth",
        "split": "val",
        "confidence": pytest.approx(0.4),
        "device": "cpu",
    }

def test_command_line_infer_confidence_is_optional_for_model_default(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["confidence"] = args.confidence

    monkeypatch.setattr(cli_evaluate, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["maesy", "evaluate", "infer", "/tmp/images", "/tmp/checkpoint.pth"],
    )

    command_line.main()

    assert captured == {"confidence": None}


def test_command_line_rejects_removed_detector_option(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["maesy", "evaluate", "infer", "/tmp/images", "/tmp/checkpoint.pth", "--detector", "detr"],
    )

    with pytest.raises(SystemExit):
        command_line.main()

    assert "unrecognized arguments" in capsys.readouterr().err


