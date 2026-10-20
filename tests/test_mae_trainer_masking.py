from types import SimpleNamespace

import pytest
import torch

from maesy.training.mae_trainer import MaeTrainer


def _build_minimal_trainer(mask_ratio: float, mask_patch_size: int) -> MaeTrainer:
    trainer = MaeTrainer.__new__(MaeTrainer)
    trainer.model = SimpleNamespace(config=SimpleNamespace(patch_size=16, image_size=64))
    trainer.config = SimpleNamespace(mask_ratio=mask_ratio, mask_patch_size=mask_patch_size)
    return trainer


def test_create_patch_mask_supports_larger_mask_blocks_at_pixel_resolution():
    trainer = _build_minimal_trainer(mask_ratio=0.5, mask_patch_size=32)
    images = torch.randn(2, 3, 64, 64)

    mask = trainer._create_patch_mask(images)

    assert mask.shape == (2, 64, 64)
    mask_2d = mask[:, ::16, ::16]
    assert torch.all((mask_2d == 0) | (mask_2d == 1))
    for bi in range(mask_2d.shape[0]):
        for y in range(0, 4, 2):
            for x in range(0, 4, 2):
                block = mask_2d[bi, y:y + 2, x:x + 2]
                assert torch.all(block == block[0, 0])


def test_create_patch_mask_honors_configured_mask_ratio(monkeypatch):
    trainer = _build_minimal_trainer(mask_ratio=0.25, mask_patch_size=16)
    sampled = {}

    def sample_random_mask(batch_size, num_tokens, mask_ratio, device):
        sampled["mask_ratio"] = mask_ratio
        return torch.zeros((batch_size, num_tokens), device=device)

    monkeypatch.setattr(trainer, "_sample_random_mask", sample_random_mask)
    mask = trainer._create_patch_mask(torch.randn(1, 3, 64, 64))

    assert sampled["mask_ratio"] == 0.25
    assert torch.count_nonzero(mask) == 0
    assert mask.shape == (1, 64, 64)


def test_create_patch_mask_rejects_incompatible_mask_block_size():
    trainer = _build_minimal_trainer(mask_ratio=0.5, mask_patch_size=24)
    with pytest.raises(ValueError, match="multiple of model patch_size"):
        trainer._create_patch_mask(torch.randn(1, 3, 64, 64))


def test_create_patch_mask_rejects_non_divisible_image_grid():
    trainer = _build_minimal_trainer(mask_ratio=0.5, mask_patch_size=32)
    with pytest.raises(ValueError, match="divisible by mask group size"):
        trainer._create_patch_mask(torch.randn(1, 3, 48, 48))


@pytest.mark.parametrize("mask_ratio", [-0.1, 0.0, 1.1, float("nan"), float("inf")])
def test_create_patch_mask_rejects_ratios_without_a_finite_positive_mask(mask_ratio):
    trainer = _build_minimal_trainer(mask_ratio=mask_ratio, mask_patch_size=16)
    with pytest.raises(ValueError, match="mask_ratio"):
        trainer._create_patch_mask(torch.randn(1, 3, 64, 64))


def test_apply_mask_to_images_zeroes_masked_patch_regions():
    trainer = _build_minimal_trainer(mask_ratio=0.0, mask_patch_size=16)
    images = torch.ones(1, 3, 64, 64)
    patch_mask = torch.zeros(1, 16)
    patch_mask[0, 0] = 1

    masked = trainer._apply_mask_to_images(images, patch_mask)

    assert masked.shape == images.shape
    assert torch.all(masked[:, :, :16, :16] == 0)
    assert torch.all(masked[:, :, 16:, 16:] == 1)

