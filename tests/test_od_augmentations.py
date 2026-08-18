import math

import torch

from _maesy_core.dataset import (
    _clip_line_to_bounds,
    _crop_targets,
    _resize_targets,
    apply_affine_to_target,
    apply_hflip,
    ODTrainTransforms,
)


def test_clip_line_to_bounds_intersects():
    p0 = torch.tensor([-1e-17, 0.25])
    p1 = torch.tensor([1.00000000000001, 0.25])
    clipped = _clip_line_to_bounds(p0, p1, width=1.0, height=1.0)
    assert clipped is not None
    c0, c1 = clipped
    assert torch.allclose(c0, torch.tensor([0.0, 0.25]))
    assert torch.allclose(c1, torch.tensor([1.0, 0.25]))


def test_clip_line_to_bounds_outside():
    p0 = torch.tensor([-0.2, -0.2])
    p1 = torch.tensor([-0.1, -0.1])
    clipped = _clip_line_to_bounds(p0, p1, width=1.0, height=1.0)
    assert clipped is None


def test_hflip_updates_lines_and_ellipses():
    image = torch.zeros((3, 50, 100), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[0.10, 0.10, 0.30, 0.40]], dtype=torch.float32),
        "line_points": torch.tensor([[0.05, 0.20, 0.20, 0.20]], dtype=torch.float32),
        "ellipses": torch.tensor([[0.25, 0.30, math.log(0.05), math.log(0.03), 0.6, 0.8]], dtype=torch.float32),
    }
    _, flipped = apply_hflip(image, target)
    assert torch.allclose(flipped["boxes"], torch.tensor([[0.70, 0.10, 0.90, 0.40]]))
    assert torch.allclose(flipped["line_points"], torch.tensor([[0.95, 0.20, 0.80, 0.20]]))
    assert torch.allclose(flipped["ellipses"][0, :2], torch.tensor([0.75, 0.30]))
    assert torch.isclose(flipped["ellipses"][0, 4], torch.tensor(0.6))
    assert torch.isclose(flipped["ellipses"][0, 5], torch.tensor(-0.8))


def test_resize_targets_noop_for_normalized_values():
    target = {
        "boxes": torch.tensor([[0.10, 0.05, 0.30, 0.20]], dtype=torch.float32),
        "line_points": torch.tensor([[0.05, 0.10, 0.20, 0.10]], dtype=torch.float32),
        "ellipses": torch.tensor([[0.25, 0.15, math.log(0.05), math.log(0.03), 0.6, 0.8]], dtype=torch.float32),
    }
    resized = _resize_targets({k: v.clone() for k, v in target.items()}, (50, 100), (224, 224))
    assert torch.allclose(resized["boxes"], target["boxes"])
    assert torch.allclose(resized["line_points"], target["line_points"])
    assert torch.allclose(resized["ellipses"], target["ellipses"])


def test_crop_targets_scales_ellipse_axes():
    target = {
        "boxes": torch.tensor([[0.40, 0.20, 0.60, 0.80]], dtype=torch.float32),
        "line_points": torch.tensor([[0.40, 0.20, 0.60, 0.80]], dtype=torch.float32),
        "ellipses": torch.tensor([[0.50, 0.50, math.log(0.10), math.log(0.12), 0.0, 1.0]], dtype=torch.float32),
    }
    cropped = _crop_targets(target, top=0, left=10, height=50, width=80, img_height=50, img_width=100)
    assert torch.allclose(cropped["boxes"], torch.tensor([[0.375, 0.20, 0.625, 0.80]]), atol=1e-6)
    assert torch.allclose(cropped["line_points"], torch.tensor([[0.375, 0.20, 0.625, 0.80]]), atol=1e-6)
    assert torch.allclose(cropped["ellipses"][0, :2], torch.tensor([0.50, 0.50]))
    assert torch.isclose(cropped["ellipses"][0, 2], torch.tensor(math.log(0.125)))
    assert torch.isclose(cropped["ellipses"][0, 3], torch.tensor(math.log(0.12)))


def test_affine_translation_works_in_normalized_coordinates():
    image = torch.zeros((3, 50, 100), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[0.20, 0.20, 0.40, 0.40]], dtype=torch.float32),
        "line_points": torch.tensor([[0.20, 0.20, 0.40, 0.20]], dtype=torch.float32),
        "ellipses": torch.tensor([[0.30, 0.35, math.log(0.10), math.log(0.08), 0.6, 0.8]], dtype=torch.float32),
    }
    _, warped = apply_affine_to_target(
        image,
        target,
        angle=0.0,
        translate=(10, 5),
        scale=1.0,
        shear=(0.0, 0.0),
    )
    expected_box = torch.tensor([[0.30, 0.30, 0.50, 0.50]], dtype=torch.float32)
    expected_line = torch.tensor([[0.30, 0.30, 0.50, 0.30]], dtype=torch.float32)
    expected_center = torch.tensor([0.40, 0.45], dtype=torch.float32)
    assert torch.allclose(warped["boxes"], expected_box, atol=1e-6)
    assert torch.allclose(warped["line_points"], expected_line, atol=1e-6)
    assert torch.allclose(warped["ellipses"][0, :2], expected_center, atol=1e-6)
    assert torch.isclose(warped["ellipses"][0, 2], target["ellipses"][0, 2], atol=1e-6)
    assert torch.isclose(warped["ellipses"][0, 3], target["ellipses"][0, 3], atol=1e-6)


def test_affine_rotation_uses_normalized_forward_matrix():
    image = torch.zeros((3, 100, 100), dtype=torch.float32)
    target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "line_points": torch.tensor([[0.75, 0.50, 0.75, 0.75]], dtype=torch.float32),
        "ellipses": torch.empty((0, 6), dtype=torch.float32),
    }

    _, warped = apply_affine_to_target(
        image,
        target,
        angle=90.0,
        translate=(0, 0),
        scale=1.0,
        shear=(0.0, 0.0),
    )

    assert torch.allclose(warped["line_points"], torch.tensor([[0.50, 0.75, 0.25, 0.75]]), atol=1e-6)


def test_train_transforms_without_targets_returns_image_tensor():
    transforms = ODTrainTransforms(image_size=32, p_affine=0.0, p_hflip=0.0, p_crop=0.0)
    transforms.color_jitter = lambda image: image
    transforms.random_autocontrast = lambda image: image
    transforms.random_grayscale = lambda image: image
    transforms.random_blur = lambda image: image
    transforms.random_sharpness = lambda image: image
    transforms.random_erasing = lambda image: image

    image = torch.zeros((3, 16, 16), dtype=torch.float32)
    output = transforms(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 32, 32)
    assert torch.isfinite(output).all()

