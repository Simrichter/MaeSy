from PIL import Image
import pytest
import torch
import yaml

from _maesy_core.dataset import MaesyDataset
from maesy.evaluation.metrics import compute_classification_metrics
from maesy.training.collate_functions import collate_classification_fn
from maesy.training.train_setups import pretrain_classification


def _make_classification_split(root, split, labels):
    images_dir = root / split / "images"
    images_dir.mkdir(parents=True)
    for index, label in enumerate(labels):
        Image.new("RGB", (12, 8), color=(index * 30, 80, 120)).save(
            images_dir / f"sample_{index}_{label}.png"
        )

    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(
            yaml.safe_dump({"path": str(root), "train": "train", "val": "val"}),
            encoding="utf-8",
        )


def test_classification_dataset_collates_filename_labels_for_both_splits(tmp_path):
    _make_classification_split(tmp_path, "train", [0, 2])
    _make_classification_split(tmp_path, "val", [1, 2])

    train = MaesyDataset(tmp_path, split="train", annotation_type="classification")
    val = MaesyDataset(tmp_path, split="val", annotation_type="classification")
    _, train_labels = collate_classification_fn([train[0], train[1]])
    _, val_labels = collate_classification_fn([val[0], val[1]])

    assert train_labels.tolist() == [0, 2]
    assert val_labels.tolist() == [1, 2]
    assert train_labels.dtype == torch.long
    assert val_labels.dtype == torch.long


def test_classification_setup_uses_labeled_splits_and_collator(monkeypatch, tmp_path):
    _make_classification_split(tmp_path, "train", [0, 2])
    _make_classification_split(tmp_path, "val", [1, 2])
    captured = {}

    class FakeModel:
        def __init__(self, config):
            self.head = type("Head", (), {"config": type("HeadConfig", (), {"num_classes": config.num_classes})()})()

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def train(self):
            captured["trained"] = True

    monkeypatch.setattr(pretrain_classification, "ClassificationCNN", FakeModel)
    monkeypatch.setattr(pretrain_classification, "ClassificationTrainer", FakeTrainer)
    monkeypatch.setattr(pretrain_classification, "DataLoader", torch.utils.data.DataLoader)

    pretrain_classification.train_classification(
        str(tmp_path), enable_wandb=False, batch_size=2, num_epochs=1, num_classes=3
    )

    train_batch = next(iter(captured["train_loader"]))
    val_batch = next(iter(captured["val_loader"]))
    assert train_batch[0].shape == (2, 3, 224, 224)
    assert train_batch[1].tolist() == [0, 2]
    assert val_batch[1].tolist() == [1, 2]
    assert captured["train_loader"].collate_fn is collate_classification_fn
    assert captured["val_loader"].collate_fn is collate_classification_fn
    assert captured["trained"] is True


@pytest.mark.parametrize(
    ("resnet_version", "feature_channels"),
    [
        ("resnet18", 512),
        ("resnet34", 512),
        ("resnet50", 2048),
        ("resnet101", 2048),
        ("resnet152", 2048),
    ],
)
def test_classification_model_returns_logits_from_configured_resnet_feature(
    monkeypatch, resnet_version, feature_channels
):
    class FakeResNetBackbone(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.feature_channels = feature_channels
            self.projection = torch.nn.Linear(3, feature_channels)

        def get_feature_channels(self):
            return (self.feature_channels,)

        def forward(self, images, *args, **kwargs):
            pooled = images.mean(dim=(-2, -1))
            feature = self.projection(pooled).unsqueeze(-1).unsqueeze(-1)
            return {self.config.feature_scales[0]: feature}

    monkeypatch.setattr(
        "_maesy_core.model.classification_CNN_model.ResNetBackbone",
        FakeResNetBackbone,
    )
    config = pretrain_classification.ClassificationCNNConfig(
        resnet_version=resnet_version,
        backbone_pretrained=False,
        num_classes=4,
    )
    model = pretrain_classification.ClassificationCNN(config)

    assert model.backbone.config.version == resnet_version
    assert model.backbone.config.pretrained is False
    assert model.backbone.config.feature_scales == ("c6",)
    assert model.head.config.feature_level == "c6"
    assert model.head.config.input_dim == feature_channels

    logits = model(torch.randn(2, 3, 8, 8))
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert model.backbone.projection.weight.grad is not None
    assert model.head.linear.weight.grad is not None


def test_classification_setup_rejects_out_of_range_validation_label_without_decoding(monkeypatch, tmp_path):
    _make_classification_split(tmp_path, "train", [0, 1])
    _make_classification_split(tmp_path, "val", [3])

    class FakeModel:
        def __init__(self, config):
            self.head = type("Head", (), {"config": type("HeadConfig", (), {"num_classes": config.num_classes})()})()

    monkeypatch.setattr(pretrain_classification, "ClassificationCNN", FakeModel)
    monkeypatch.setattr(pretrain_classification, "ClassificationTrainer", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="outside the model class range"):
        pretrain_classification.train_classification(
            str(tmp_path), enable_wandb=False, batch_size=2, num_epochs=1, num_classes=3
        )


def test_classification_model_factory_dispatches_serialized_config(monkeypatch):
    from _maesy_core.model.model_tools.model_factory import create_model_from_dict

    created_configs = []

    class FakeClassifier:
        def __init__(self, config):
            created_configs.append(config)

    monkeypatch.setattr("_maesy_core.model.ClassificationCNN", FakeClassifier)

    model = create_model_from_dict({
        "type": "classification_cnn",
        "resnet_version": "resnet18",
        "embed_dim": 32,
        "num_classes": 5,
    })

    assert isinstance(model, FakeClassifier)
    assert created_configs[0].type == "classification_cnn"
    assert created_configs[0].resnet_version == "resnet18"
    assert created_configs[0].num_classes == 5


def test_patch_classifier_factory_matches_normalized_config_type(monkeypatch):
    from _maesy_core.model.model_tools.model_factory import create_model_from_dict

    created_configs = []

    class FakePatchClassifier:
        def __init__(self, config):
            created_configs.append(config)

    monkeypatch.setattr("_maesy_core.model.PatchClassificator", FakePatchClassifier)

    model = create_model_from_dict({
        "type": "PatchClassificator",
        "resnet_version": "resnet18",
        "backbone_pretrained": False,
        "num_classes": 2,
    })

    assert isinstance(model, FakePatchClassifier)
    assert created_configs[0].type == "PatchClassificator"
    assert created_configs[0].backbone_pretrained is False


def test_classifier_checkpoint_reconstruction_disables_pretrained_weights(monkeypatch):
    from _maesy_core.model.model_tools.checkpoint_handler import CheckpointHandler

    received = {}

    class FakeClassifier:
        def __init__(self, config):
            self.config = config

    def build_model(config):
        received.update(config)
        return FakeClassifier(type("Config", (), config)())

    monkeypatch.setattr(
        "_maesy_core.model.model_tools.model_factory.create_model_from_dict",
        build_model,
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "modelconfig": {
                "type": "classification_cnn",
                "backbone_pretrained": True,
                "resnet_version": "resnet18",
            },
            "backbone": {},
            "head": {},
            "backbonetype": "unused",
            "headtype": "unused",
        },
    )
    # Skip component loading; this test is about reconstruction policy.
    monkeypatch.setattr(CheckpointHandler, "_autoload_model", lambda *_args: None)

    model = CheckpointHandler(device=torch.device("cpu")).load_model("local.pth")

    assert isinstance(model, FakeClassifier)
    assert received["backbone_pretrained"] is False
    assert received["resnet_version"] == "resnet18"


def test_model_tools_lazy_exports_preserve_public_api_without_importing_factory_eagerly():
    import _maesy_core.model as models
    from _maesy_core.model.model_tools import CheckpointHandler, create_model_from_dict

    assert models.RTDETR is not None
    assert CheckpointHandler.__name__ == "CheckpointHandler"
    assert callable(create_model_from_dict)


def test_model_tools_exports_are_available_in_fresh_process():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import _maesy_core.model as m; "
            "from _maesy_core.model.model_tools import CheckpointHandler, create_model_from_dict; "
            "assert m.RTDETR and CheckpointHandler and callable(create_model_from_dict)",
        ],
        check=True,
    )


def test_classification_metrics_include_all_classes_and_guard_zero_denominators():
    metrics = compute_classification_metrics(
        [torch.tensor([0, 2, 2])],
        [torch.tensor([0, 1, 2])],
    )

    assert metrics["Accuracy"] == pytest.approx(2 / 3)
    assert metrics["Precision"] == pytest.approx(1 / 2)
    assert metrics["Recall"] == pytest.approx(2 / 3)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())

    no_positive_predictions = compute_classification_metrics(
        [torch.tensor([0, 0])],
        [torch.tensor([0, 1])],
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in no_positive_predictions.values())
    assert no_positive_predictions["Accuracy"] == pytest.approx(0.5)
