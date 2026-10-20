from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from _maesy_core.dataset import MaesyDataset


def _write_dataset(root: Path, box_format: str, box_row: str) -> Path:
    split = root / "train"
    images_dir = split / "images"
    labels_dir = split / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10), color=(32, 64, 96)).save(images_dir / "frame.png")
    (labels_dir / "frame.txt").write_text(
        f"{box_row}\n"
        "1 0.1 0.2 0.8 0.2\n"
        "2 0.5 0.5 -2.302585 -2.995732 1.0 0.0\n",
        encoding="utf-8",
    )
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({
            "path": str(root),
            "train": "train",
            "nc": 3,
            "names": ["ball", "lines", "ellipses"],
            "box_format": box_format,
            "lines": "lines",
            "ellipses": "ellipses",
        }),
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize(
    "box_format,box_row",
    [
        ("xyxy", "0 0.2 0.3 0.6 0.7"),
        ("cxcywh", "0 0.4 0.5 0.4 0.4"),
    ],
)
def test_maesy_dataset_parses_normalized_boxes_lines_and_ellipses(tmp_path, box_format, box_row):
    root = _write_dataset(tmp_path / box_format, box_format, box_row)
    dataset = MaesyDataset(root, split="train", annotation_type="detection", enable_lines=True, enable_ellipses=True)

    image, target = dataset[0]

    assert image.shape == (3, 10, 20)
    assert target["boxes"].shape == (1, 4)
    assert torch.allclose(target["boxes"][0], torch.tensor([0.2, 0.3, 0.6, 0.7]), atol=1e-6)
    assert target["labels"].tolist() == [0, 1, 2]
    assert target["labels"].dtype == torch.long
    assert target["line_points"].shape == (1, 4)
    assert target["line_points"][0].tolist() == pytest.approx([0.1, 0.2, 0.8, 0.2])
    assert target["ellipses"].shape == (1, 6)
    assert dataset.get_num_classes() == 3
    assert dataset.get_special_classes() == {"line_class_id": 1, "ellipse_class_id": 2}


def test_maesy_dataset_can_disable_special_annotations_without_losing_bbox_contract(tmp_path):
    root = _write_dataset(tmp_path / "no_special", "xyxy", "0 0.2 0.3 0.6 0.7")
    dataset = MaesyDataset(root, split="train", annotation_type="detection")

    _image, target = dataset[0]

    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [0]
    assert target["line_points"].shape == (0, 4)
    assert target["ellipses"].shape == (0, 6)
    assert dataset.get_num_classes() == 1


def test_auto_annotation_type_dispatches_as_detection_for_nested_images(tmp_path):
    root = tmp_path / "auto"
    nested_images = root / "train" / "images" / "camera" / "session"
    nested_labels = root / "train" / "labels" / "camera" / "session"
    nested_images.mkdir(parents=True)
    nested_labels.mkdir(parents=True)
    Image.new("RGB", (20, 10), color=(32, 64, 96)).save(nested_images / "frame.png")
    (nested_labels / "frame.txt").write_text("0 0.2 0.3 0.6 0.7\n", encoding="utf-8")
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({
            "path": str(root), "train": "train", "nc": 1,
            "names": ["ball"], "box_format": "xyxy",
        }),
        encoding="utf-8",
    )

    dataset = MaesyDataset(root, split="train", annotation_type="auto")
    _image, target = dataset[0]

    assert dataset.annotation_type == "detection"
    assert dataset.get_image_path(0) == nested_images / "frame.png"
    assert target["labels"].tolist() == [0]
    assert target["boxes"][0].tolist() == pytest.approx([0.2, 0.3, 0.6, 0.7])


def test_dataset_yaml_path_is_resolved_relative_to_yaml_file(tmp_path, monkeypatch):
    root = _write_dataset(tmp_path / "portable", "xyxy", "0 0.2 0.3 0.6 0.7")
    yaml_path = root / "dataset.yaml"
    monkeypatch.chdir(tmp_path)

    dataset = MaesyDataset(yaml_path, split="train", annotation_type="detection")

    assert len(dataset) == 1
    assert dataset.get_image_path(0) == root / "train" / "images" / "frame.png"
