from types import SimpleNamespace

import pytest
import torch
from torch import nn

from _maesy_core.model.model_tools.checkpoint_handler import CheckpointHandler


class DummyPart(nn.Module):
    def __init__(self, part_type: str, width: int):
        super().__init__()
        self.type = part_type
        self.config = SimpleNamespace(width=width)
        self.layer = nn.Linear(4, width)

    def forward(self, x):
        return self.layer(x)


class DummyModel(nn.Module):
    def __init__(self, backbone_type: str = "bb", head_type: str = "head", head_width: int = 2):
        super().__init__()
        self.backbone = DummyPart(backbone_type, 4)
        self.head = DummyPart(head_type, head_width)


class DummyConfig:
    pass


def test_checkpoint_save_and_load_round_trip(tmp_path):
    device = torch.device("cpu")
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    handler = CheckpointHandler(device=device, save_dir=tmp_path)
    handler.save_checkpoint(
        current_epoch=3,
        global_step=12,
        model=model,
        optimizer=optimizer,
        best_val_loss=1.23,
        config=DummyConfig(),
        filename="ckpt.pth",
        scheduler=scheduler,
    )

    reloaded_model = DummyModel()
    reloaded_optimizer = torch.optim.SGD(reloaded_model.parameters(), lr=0.1)
    reloaded_scheduler = torch.optim.lr_scheduler.StepLR(reloaded_optimizer, step_size=1)

    epoch, step, best_loss = handler.load_training_state(
        str(tmp_path / "ckpt.pth"),
        optimizer=reloaded_optimizer,
        scheduler=reloaded_scheduler,
    )

    reloaded_model = handler.load_model(str(tmp_path / "ckpt.pth"))

    assert epoch == 3
    assert step == 12
    assert best_loss == 1.23

    for original, restored in zip(model.backbone.parameters(), reloaded_model.backbone.parameters()):
        assert torch.allclose(original, restored)
    for original, restored in zip(model.head.parameters(), reloaded_model.head.parameters()):
        assert torch.allclose(original, restored)


def test_checkpoint_load_fails_on_incompatible_head_config(tmp_path):
    device = torch.device("cpu")
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    handler = CheckpointHandler(device=device, save_dir=tmp_path)
    handler.save_checkpoint(
        current_epoch=0,
        global_step=0,
        model=model,
        optimizer=optimizer,
        best_val_loss=0.0,
        config=DummyConfig(),
        filename="ckpt.pth",
    )

    incompatible_model = DummyModel()
    incompatible_model.head.config = SimpleNamespace(width=999)

    with pytest.raises(ValueError, match="incompatible configuration"):
        handler.load_model(str(tmp_path / "ckpt.pth"), model=incompatible_model)

