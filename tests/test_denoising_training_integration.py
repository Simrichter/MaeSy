"""Denoising configuration behavior in object-detection training setup."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import maesy.training.train_setups.train_object_detection as train_object_detection


def _patch_training_dependencies(monkeypatch, trainer_type):
    dataset = MagicMock()
    dataset.get_num_classes.return_value = 3
    dataset.get_special_classes.return_value = {
        "line_class_id": -1,
        "ellipse_class_id": -1,
    }

    monkeypatch.setattr(train_object_detection, "ODValTransforms", lambda **kwargs: object())
    monkeypatch.setattr(train_object_detection, "MaesyDataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_object_detection, "MultiDataset", lambda datasets: dataset)
    monkeypatch.setattr(train_object_detection, "DataLoader", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_object_detection, "DetectionTrainer", trainer_type)
    monkeypatch.setattr(train_object_detection, "replace_bn_with_frozenbn", lambda backbone: None)
    return dataset


def test_denoising_settings_are_applied_to_new_training_model(monkeypatch):
    captured = {}

    class CapturingTrainer:
        def __init__(self, model, **kwargs):
            self.model = model
            captured["model"] = model

        def train(self):
            captured["trained"] = True

    _patch_training_dependencies(monkeypatch, CapturingTrainer)

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_classes=3,
            softmax_activated=True,
            line_class_id=-1,
            ellipse_class_id=-1,
            enable_denoising=False,
            denoising_num_queries=0,
            denoising_label_noise_ratio=0.2,
            denoising_box_noise_scale=0.4,
        ),
        head=SimpleNamespace(
            config=SimpleNamespace(
                enable_denoising=False,
                denoising_num_queries=0,
                denoising_label_noise_ratio=0.2,
                denoising_box_noise_scale=0.4,
                embed_dim=8,
            ),
            dn_query_embedding=None,
        ),
        backbone=object(),
    )

    def create_model(model_info, overwrite_func):
        captured["model_info"] = model_info
        captured["config"] = overwrite_func({
            "num_classes": 3,
            "line_class_id": -1,
            "ellipse_class_id": -1,
        })
        return model

    monkeypatch.setattr(train_object_detection, "create_model", create_model)

    train_object_detection.train_vit_detector(
        model_info="rt-detr",
        dataset_paths=["/tmp/dummy_dataset"],
        output_dir="/tmp/checkpoints",
        finetune=False,
        continue_training_from_checkpoint=False,
        pretrained_backbone="",
        wandb_name=None,
        enable_wandb=False,
        enable_denoising=True,
        denoising_num_queries=6,
        denoising_label_noise_ratio=0.3,
        denoising_box_noise_scale=0.5,
        enable_line_detection=False,
        enable_ellipse_detection=False,
    )

    model = captured["model"]
    assert captured["trained"] is True
    assert captured["model_info"] == "rt-detr"
    assert captured["config"]["enable_denoising"] is True
    assert captured["config"]["denoising_num_queries"] == 6
    assert captured["config"]["denoising_label_noise_ratio"] == 0.3
    assert captured["config"]["denoising_box_noise_scale"] == 0.5
    assert model.config.enable_denoising is True
    assert model.config.denoising_num_queries == 6
    assert model.config.denoising_label_noise_ratio == 0.3
    assert model.config.denoising_box_noise_scale == 0.5
    assert model.head.config.enable_denoising is True
    assert model.head.config.denoising_num_queries == 6
    assert model.head.config.denoising_label_noise_ratio == 0.3
    assert model.head.config.denoising_box_noise_scale == 0.5
    assert isinstance(model.head.dn_query_embedding, torch.nn.Embedding)
    assert model.head.dn_query_embedding.weight.shape == (6, 8)


def test_resuming_training_preserves_checkpoint_denoising_settings(monkeypatch):
    checkpoint_embedding = object()
    checkpoint_settings = {
        "enable_denoising": True,
        "denoising_num_queries": 4,
        "denoising_label_noise_ratio": 0.15,
        "denoising_box_noise_scale": 0.25,
    }
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_classes=3,
            softmax_activated=False,
            **checkpoint_settings,
        ),
        head=SimpleNamespace(
            config=SimpleNamespace(**checkpoint_settings, embed_dim=8),
            dn_query_embedding=checkpoint_embedding,
        ),
        backbone=object(),
    )
    captured = {}

    class CapturingTrainer:
        def __init__(self, model, **kwargs):
            self.model = model
            captured["model"] = model

        def resume(self, checkpoint_path):
            captured["resumed_from"] = checkpoint_path

        def train(self):
            raise AssertionError("resume setup must not start a new training run")

    _patch_training_dependencies(monkeypatch, CapturingTrainer)

    def load_checkpoint_model(model_info, overwrite_func):
        captured["model_info"] = model_info
        captured["overwrite_func"] = overwrite_func
        return model

    monkeypatch.setattr(train_object_detection, "create_model", load_checkpoint_model)
    checkpoint_path = "/tmp/checkpoint.pth"

    train_object_detection.train_vit_detector(
        model_info=checkpoint_path,
        dataset_paths=["/tmp/dummy_dataset"],
        output_dir="/tmp/checkpoints",
        finetune=False,
        continue_training_from_checkpoint=True,
        pretrained_backbone="",
        wandb_name=None,
        enable_wandb=False,
        # Deliberately differ from the settings stored in the checkpoint.
        enable_denoising=False,
        denoising_num_queries=99,
        denoising_label_noise_ratio=0.9,
        denoising_box_noise_scale=0.8,
        enable_line_detection=False,
        enable_ellipse_detection=False,
    )

    assert captured["model"] is model
    assert captured["model_info"] == checkpoint_path
    assert captured["resumed_from"] == checkpoint_path
    assert model.config.enable_denoising is True
    assert model.config.denoising_num_queries == 4
    assert model.config.denoising_label_noise_ratio == 0.15
    assert model.config.denoising_box_noise_scale == 0.25
    assert model.head.config.enable_denoising is True
    assert model.head.config.denoising_num_queries == 4
    assert model.head.config.denoising_label_noise_ratio == 0.15
    assert model.head.config.denoising_box_noise_scale == 0.25
    assert model.head.dn_query_embedding is checkpoint_embedding
