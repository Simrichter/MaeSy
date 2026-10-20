from types import SimpleNamespace

import pytest
import torch
from torch import nn

from _maesy_core.model import ViTDetector, ViTDetectorConfig
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
        self.config = SimpleNamespace(type="dummy")
        self.backbone = DummyPart(backbone_type, 4)
        self.head = DummyPart(head_type, head_width)


class DummyConfig:
    pass


def _tiny_vit_config():
    return ViTDetectorConfig(
        image_size=32,
        patch_size=8,
        embed_dim=16,
        num_layers=1,
        num_heads=4,
        num_decoder_layers=1,
        decoder_num_heads=4,
        num_classes=3,
        num_queries=4,
        hidden_dim=32,
        dropout=0.0,
        attention_dropout=0.0,
        decoder_dropout=0.0,
    )


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

    reloaded_model = handler.load_model(str(tmp_path / "ckpt.pth"), model=DummyModel())

    assert epoch == 3
    assert step == 12
    assert best_loss == 1.23
    assert reloaded_model.config.type == "dummy"

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


def test_vit_detector_checkpoint_reconstructs_model_and_restores_component_weights(tmp_path):
    torch.manual_seed(41)
    model = ViTDetector(_tiny_vit_config())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)

    handler = CheckpointHandler(device=torch.device("cpu"), save_dir=tmp_path)
    handler.save_checkpoint(
        current_epoch=2,
        global_step=7,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        best_val_loss=0.5,
        config=DummyConfig(),
        filename="vit.pth",
    )

    restored = handler.load_model(str(tmp_path / "vit.pth"))

    assert isinstance(restored, ViTDetector)
    assert restored.config.num_classes == 3
    for component_name in ("backbone", "head"):
        original_state = getattr(model, component_name).state_dict()
        restored_state = getattr(restored, component_name).state_dict()
        assert original_state.keys() == restored_state.keys()
        for key, original_value in original_state.items():
            torch.testing.assert_close(restored_state[key], original_value)
