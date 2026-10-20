from types import SimpleNamespace

import torch

import maesy.evaluation.cli_evaluate as cli_evaluate
import maesy.training.cli_train as cli_train
import maesy.training.train_setups as train_setups


def test_cli_train_forwards_current_od_options(monkeypatch):
    captured = {}

    def fake_train_vit_detector(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(train_setups, "train_vit_detector", fake_train_vit_detector)
    args = SimpleNamespace(
        mode="od",
        model="resnet50",
        dataset=["/tmp/data"],
        output="/tmp/out",
        finetune=True,
        wandb=False,
        name=None,
        resume=True,
        backbone="/tmp/backbone.pth",
        enable_denoising=True,
        dn_queries=7,
        dn_label_noise=0.3,
        dn_box_noise=0.5,
        enable_line_detection=True,
        enable_ellipse_detection=False,
        learning_rate=1e-4,
        backbone_learning_rate=-1.0,
        batch_size=8,
        eos_coef=-1.0,
        seed=123,
        device="cpu",
        fast_mode=False,
        debug=False,
    )

    cli_train.main(args)

    assert captured["model_info"] == "resnet50"
    assert captured["dataset_paths"] == ["/tmp/data"]
    assert captured["continue_training_from_checkpoint"] is True
    assert captured["pretrained_backbone"] == "/tmp/backbone.pth"
    assert captured["enable_denoising"] is True
    assert captured["denoising_num_queries"] == 7
    assert captured["denoising_label_noise_ratio"] == 0.3
    assert captured["denoising_box_noise_scale"] == 0.5
    assert captured["enable_line_detection"] is True
    assert captured["enable_ellipse_detection"] is False
    assert captured["override_params"] == {"learning_rate": 1e-4, "batch_size": 8}
    assert captured["device"] == "cpu"


def test_cli_evaluate_maps_device_and_forwards_inference_options(monkeypatch):
    captured = {}

    def fake_infer_vit_detector(*args):
        captured["args"] = args

    monkeypatch.setattr(train_setups, "infer_vit_detector", fake_infer_vit_detector)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(
        command="infer",
        checkpoint="/tmp/checkpoint.pth",
        imgpath="/tmp/images",
        out="/tmp/out",
        visualize=True,
        device="cpu",
        split="val",
        confidence=0.4,
    )

    cli_evaluate.main(args)

    checkpoint, image_path, output, visualize, device, split, confidence = captured["args"]
    assert checkpoint == "/tmp/checkpoint.pth"
    assert image_path == "/tmp/images"
    assert output == "/tmp/out"
    assert visualize is True
    assert device == torch.device("cpu")
    assert split == "val"
    assert confidence == 0.4
