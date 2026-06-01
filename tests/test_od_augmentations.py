import math

import torch

from maesy.dataset.od_augmentations import apply_affine_to_target, apply_hflip_to_target, _clip_line_to_bounds


def test_clip_line_to_bounds_intersects():
    p0 = torch.tensor([-10.0, 25.0])
    p1 = torch.tensor([110.0, 25.0])
    clipped = _clip_line_to_bounds(p0, p1, width=100.0, height=50.0)
    assert clipped is not None
    c0, c1 = clipped
    assert torch.allclose(c0, torch.tensor([0.0, 25.0]))
    assert torch.allclose(c1, torch.tensor([100.0, 25.0]))


def test_clip_line_to_bounds_outside():
    p0 = torch.tensor([-10.0, -10.0])
    p1 = torch.tensor([-5.0, -5.0])
    clipped = _clip_line_to_bounds(p0, p1, width=100.0, height=50.0)
    assert clipped is None


def test_hflip_updates_lines_and_ellipses():
    image = torch.zeros((3, 50, 100), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[10.0, 5.0, 30.0, 20.0]], dtype=torch.float32),
        "line_points": torch.tensor([[5.0, 10.0, 20.0, 10.0]], dtype=torch.float32),
        "ellipses": torch.tensor([[25.0, 15.0, math.log(5.0), math.log(3.0), 0.6, 0.8]], dtype=torch.float32),
    }
    _, flipped = apply_hflip_to_target(image, target)
    assert torch.allclose(flipped["boxes"], torch.tensor([[70.0, 5.0, 90.0, 20.0]]))
    assert torch.allclose(flipped["line_points"], torch.tensor([[95.0, 10.0, 80.0, 10.0]]))
    assert torch.allclose(flipped["ellipses"][0, :2], torch.tensor([75.0, 15.0]))
    assert torch.isclose(flipped["ellipses"][0, 4], torch.tensor(0.6))
    assert torch.isclose(flipped["ellipses"][0, 5], torch.tensor(-0.8))


def test_affine_identity_keeps_targets():
    image = torch.zeros((3, 50, 100), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[10.0, 5.0, 30.0, 20.0]], dtype=torch.float32),
        "line_points": torch.tensor([[5.0, 10.0, 20.0, 10.0]], dtype=torch.float32),
        "ellipses": torch.tensor([[25.0, 15.0, math.log(5.0), math.log(3.0), 0.6, 0.8]], dtype=torch.float32),
    }
    _, warped = apply_affine_to_target(
        image,
        target,
        angle=0.0,
        translate=(0, 0),
        scale=1.0,
        shear=(0.0, 0.0),
    )
    assert torch.allclose(warped["boxes"], target["boxes"])
    assert torch.allclose(warped["line_points"], target["line_points"])
    assert torch.allclose(warped["ellipses"], target["ellipses"])

