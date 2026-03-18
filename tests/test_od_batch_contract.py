import torch

from maesy.training.utils import collate_detection_fn, handle_raw_batch


def test_collate_detection_fn_and_handle_raw_batch_contract():
    batch = [
        (
            torch.rand(3, 8, 8),
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            },
        ),
        (
            torch.rand(3, 8, 8),
            {
                "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.1]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
            },
        ),
    ]

    images, targets = collate_detection_fn(batch)

    assert images.shape == (2, 3, 8, 8)
    assert isinstance(targets, list)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)

    images_dev, targets_dev = handle_raw_batch((images, targets), torch.device("cpu"))

    assert images_dev.device.type == "cpu"
    assert isinstance(targets_dev, list)
    assert targets_dev[0]["boxes"].device.type == "cpu"
    assert targets_dev[0]["labels"].dtype == torch.long

