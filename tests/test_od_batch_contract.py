import pytest
import torch

from _maesy_core.dataset.utils import handle_raw_batch
from maesy.training.collate_functions import collate_detection_fn


def test_collate_detection_fn_preserves_variable_length_targets():
    batch = [
        (
            torch.rand(3, 8, 8),
            {
                "boxes": torch.tensor([[0.375, 0.375, 0.625, 0.625]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
                "line_points": torch.tensor([[0.1, 0.2, 0.8, 0.2]], dtype=torch.float32),
            },
        ),
        (
            torch.rand(3, 8, 8),
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.long),
                "line_points": torch.empty((0, 4), dtype=torch.float32),
            },
        ),
    ]

    images, targets = collate_detection_fn(batch)

    assert images.shape == (2, 3, 8, 8)
    assert isinstance(targets, list)
    assert [target["boxes"].shape for target in targets] == [(1, 4), (0, 4)]
    assert targets[0]["labels"].dtype == torch.long
    assert targets[1]["line_points"].shape == (0, 4)


def test_handle_raw_batch_moves_all_target_fields_to_device():
    images = torch.rand(2, 3, 8, 8)
    targets = [{
        "boxes": torch.ones((1, 4)),
        "labels": torch.ones((1,), dtype=torch.long),
        "line_points": torch.ones((1, 4)),
        "ellipses": torch.ones((1, 6)),
    }]

    images_dev, targets_dev = handle_raw_batch((images, targets), torch.device("cpu"))

    assert images_dev.device.type == "cpu"
    assert targets_dev[0]["boxes"].device.type == "cpu"
    assert targets_dev[0]["line_points"].shape == (1, 4)
    assert targets_dev[0]["ellipses"].shape == (1, 6)
    assert targets_dev[0]["labels"].dtype == torch.long
    assert torch.equal(targets_dev[0]["boxes"], targets[0]["boxes"])


def test_collate_detection_fn_rejects_different_image_shapes():
    batch = [
        (torch.rand(3, 8, 8), {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.long)}),
        (torch.rand(3, 6, 8), {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.long)}),
    ]

    with pytest.raises(RuntimeError):
        collate_detection_fn(batch)


def test_collate_classification_fn_stacks_images_and_filename_labels():
    from maesy.training.collate_functions import collate_classification_fn

    batch = [
        (torch.zeros(3, 8, 8), [{"label": 2}]),
        (torch.ones(3, 8, 8), [{"label": 0}]),
    ]

    images, labels = collate_classification_fn(batch)

    assert images.shape == (2, 3, 8, 8)
    assert labels.tolist() == [2, 0]
    assert labels.dtype == torch.long

