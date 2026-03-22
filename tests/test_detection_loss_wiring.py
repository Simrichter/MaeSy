from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from maesy.training.base_trainer import BaseTrainer
from maesy.training.config import TrainingConfig
from maesy.training.losses import DetectionLoss


class DummyDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.projection = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)
        self.config = SimpleNamespace(
            num_classes=3,
            bbox_loss_coef=7.0,
            class_loss_coef=3.0,
            giou_loss_coef=4.0,
            aux_loss_coef=0.6,
            eos_coef=0.25,
        )

    def forward(self, x):
        return x


class DummyTrainer(BaseTrainer):
    pass


def test_detection_loss_uses_model_config_coefficients():
    model = DummyDetectionModel()
    dataset = TensorDataset(torch.zeros(2, 4), torch.zeros(2, 4))
    train_loader = DataLoader(dataset, batch_size=1)

    trainer = DummyTrainer(
        model=model,
        train_loader=train_loader,
        project_name="test-project",
        config=TrainingConfig(criterion="DetectionLoss", device=torch.device("cpu")),
        enable_wandb=False,
    )

    assert isinstance(trainer.loss, DetectionLoss)
    assert trainer.loss.num_classes == 3
    assert trainer.loss.bbox_loss_coef == 7.0
    assert trainer.loss.class_loss_coef == 3.0
    assert trainer.loss.giou_loss_coef == 4.0
    assert trainer.loss.aux_loss_coef == 0.6
    assert trainer.loss.empty_weight[-1].item() == 0.25

