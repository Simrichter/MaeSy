import pytest

from maesy.dataset.bounding_box import BoundingBox


def test_scale_to_size_preserves_subpixel_precision():
    box = BoundingBox.from_str("0 0.1234 0.5678 0.1111 0.2222", xyxy=False)

    box.scale_to_size(image_width=640, image_height=480)
    cx, cy, w, h = box.as_cxcywh()

    assert box.normalized is False
    assert cx == pytest.approx(0.1234 * 640)
    assert cy == pytest.approx(0.5678 * 480)
    assert w == pytest.approx(0.1111 * 640)
    assert h == pytest.approx(0.2222 * 480)


