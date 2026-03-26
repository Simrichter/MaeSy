import torch
from types import SimpleNamespace

import maesy.evaluation.cli_evaluate as cli_evaluate
import maesy.training.cli_train as cli_train


def test_cli_train_forwards_detector_arch(monkeypatch):
    captured = {}

    def fake_train_vit_detector(checkpoint, dataset, output, no_freeze, enable_wandb, continue_from_checkpoint, detector_arch, **kwargs):
        captured["checkpoint"] = checkpoint
        captured["dataset"] = dataset
        captured["output"] = output
        captured["no_freeze"] = no_freeze
        captured["enable_wandb"] = enable_wandb
        captured["continue"] = continue_from_checkpoint
        captured["detector"] = detector_arch
        captured["enable_denoising"] = kwargs.get("enable_denoising")
        captured["enable_line_detection"] = kwargs.get("enable_line_detection")

    monkeypatch.setattr(cli_train, "train_vit_detector", fake_train_vit_detector)

    args = SimpleNamespace(
        mode="od",
        checkpoint="/tmp/ckpt.pth",
        dataset="/tmp/data",
        output="/tmp/out",
        no_freeze=True,
        wandb=False,
        resume=True,
        detector="rt_detr",
    )

    cli_train.main(args)

    assert captured["detector"] == "rt_detr"
    assert captured["continue"] is True
    assert captured["enable_denoising"] is False
    assert captured["enable_line_detection"] is False


def test_cli_evaluate_maps_auto_detector_to_none(monkeypatch):
    captured = {}

    def fake_infer_vit_detector(checkpoint, imgpath, out, visualize, device, detector_arch=None):
        captured["checkpoint"] = checkpoint
        captured["imgpath"] = imgpath
        captured["out"] = out
        captured["visualize"] = visualize
        captured["device"] = device
        captured["detector"] = detector_arch

    monkeypatch.setattr(cli_evaluate, "infer_vit_detector", fake_infer_vit_detector)

    args = SimpleNamespace(
        command="infer",
        checkpoint="/tmp/ckpt.pth",
        imgpath="/tmp/images",
        out="/tmp/out",
        visualize=False,
        device="cpu",
        detector="auto",
    )

    cli_evaluate.main(args)

    assert captured["detector"] is None
    assert captured["device"] == torch.device("cpu")

