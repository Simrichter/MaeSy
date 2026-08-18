from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from _maesy_core.model.base_model import BaseModel
from maesy.training.base_trainer import BaseTrainer
from maesy.training import BaseTrainingConfig


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.embedding = nn.Embedding(16, 8)

    def forward(self, x, **kwargs):
        return self.norm(self.linear(x))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 3)
        self.norm = nn.LayerNorm(3)

    def forward(self, x, **kwargs):
        return self.norm(self.linear(x))


class TinyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = TinyHead()
        self.config = SimpleNamespace(num_classes=3)


class DummyTrainer(BaseTrainer):
    pass


def test_adamw_applies_weight_decay_only_to_standard_weights():
    model = TinyModel()
    dataset = TensorDataset(torch.zeros(2, 8), torch.zeros(2, dtype=torch.long))
    train_loader = DataLoader(dataset, batch_size=1)

    config = BaseTrainingConfig(
        optimizer="adamw",
        lr_scheduler="step",
        criterion="ClassificationLoss",
        learning_rate=1e-3,
        backbone_learning_rate=1e-4,
        weight_decay=1e-2,
        use_amp=False,
        device=torch.device("cpu"),
    )

    trainer = DummyTrainer(
        model=model,
        train_loader=train_loader,
        project_name="test-project",
        config=config,
        enable_wandb=False,
    )

    name_by_id = {id(param): name for name, param in model.named_parameters()}
    decay_names = set()
    no_decay_names = set()
    lr_by_name = {}

    for group in trainer.optimizer.param_groups:
        names = {name_by_id[id(param)] for param in group["params"]}
        for name in names:
            lr_by_name[name] = group["lr"]
        if group["weight_decay"] == 0.0:
            no_decay_names.update(names)
        else:
            decay_names.update(names)

    expected_decay = {
        "backbone.linear.weight",
        "head.linear.weight",
    }
    expected_no_decay = {
        "backbone.linear.bias",
        "backbone.norm.weight",
        "backbone.norm.bias",
        "backbone.embedding.weight",
        "head.linear.bias",
        "head.norm.weight",
        "head.norm.bias",
    }

    assert decay_names == expected_decay
    assert no_decay_names == expected_no_decay
    assert decay_names.isdisjoint(no_decay_names)

    assert lr_by_name["backbone.linear.weight"] == config.backbone_learning_rate
    assert lr_by_name["backbone.embedding.weight"] == config.backbone_learning_rate
    assert lr_by_name["head.linear.weight"] == config.learning_rate
    assert lr_by_name["head.norm.weight"] == config.learning_rate


