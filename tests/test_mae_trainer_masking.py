from types import SimpleNamespace

import torch

from maesy.training.mae_trainer import MaeTrainer


def _build_minimal_trainer(mask_ratio: float, mask_patch_size: int) -> MaeTrainer:
    trainer = MaeTrainer.__new__(MaeTrainer)
    trainer.model = SimpleNamespace(config=SimpleNamespace(patch_size=16, image_size=64))
    trainer.config = SimpleNamespace(mask_ratio=mask_ratio, mask_patch_size=mask_patch_size)
    return trainer


def test_create_patch_mask_supports_larger_mask_blocks():
    trainer = _build_minimal_trainer(mask_ratio=0.5, mask_patch_size=32)
    images = torch.randn(2, 3, 64, 64)

    mask = trainer._create_patch_mask(images)

    assert mask.shape == (2, 16)
    mask_2d = mask.view(2, 4, 4)
    for bi in range(mask_2d.shape[0]):
        for y in range(0, 4, 2):
            for x in range(0, 4, 2):
                block = mask_2d[bi, y:y + 2, x:x + 2]
                assert torch.all(block == block[0, 0])


def test_apply_mask_to_images_zeroes_masked_patch_regions():
    trainer = _build_minimal_trainer(mask_ratio=0.0, mask_patch_size=16)
    images = torch.ones(1, 3, 64, 64)
    patch_mask = torch.zeros(1, 16)
    patch_mask[0, 0] = 1

    masked = trainer._apply_mask_to_images(images, patch_mask)

    assert masked.shape == images.shape
    assert torch.all(masked[:, :, :16, :16] == 0)
    assert torch.all(masked[:, :, 16:, 16:] == 1)

