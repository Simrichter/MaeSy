from types import SimpleNamespace

import pytest
import torch

import maesy.training.train_setups.train_object_detection as train_object_detection


def test_infer_writes_xyxy_boxes_and_optional_geometry(tmp_path, monkeypatch):
    images_dir = tmp_path / "input"
    images_dir.mkdir()
    image_names = ["frame.png", "empty.png"]
    for name in image_names:
        (images_dir / name).write_bytes(f"synthetic {name}".encode())

    dataset = SimpleNamespace(images=image_names, images_dir=images_dir)
    transform = object()
    monkeypatch.setattr(train_object_detection, "create_model_from_checkpoint", lambda _path: SimpleNamespace(
        eval=lambda: None,
        head=SimpleNamespace(config=SimpleNamespace(line_class_id=7, ellipse_class_id=8)),
    ))
    monkeypatch.setattr(train_object_detection, "ODValTransforms", lambda **kwargs: (kwargs, transform))
    dataset_calls = []

    def make_dataset(*args, **kwargs):
        dataset_calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr(train_object_detection, "MaesyDataset", make_dataset)
    loader_calls = []

    def make_loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(train_object_detection, "DataLoader", make_loader)
    raw_box = torch.tensor([[0.1, 0.2, 0.7, 0.8]])
    line = torch.tensor([[0.125, 0.25, 0.875, 0.5]])
    ellipse = torch.tensor([[0.125, 0.25, 0.5, 0.625, 0.75, 0.875]])
    predictions = [
        {
            "boxes": raw_box,
            "labels": torch.tensor([2]),
            "scores": torch.tensor([0.875]),
            "line_points": line,
            "ellipses": ellipse,
        },
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.long),
            "scores": torch.empty((0,)),
        },
    ]
    inferer_calls = []

    class FakeInferer:
        def __init__(self, model, data_loader, device):
            inferer_calls.append((model, data_loader, device))

        def infer(self, **kwargs):
            inferer_calls.append(kwargs)
            return predictions, []

    monkeypatch.setattr(train_object_detection, "Inferer", FakeInferer)
    monkeypatch.setattr(train_object_detection, "tqdm", lambda iterable: iterable)
    out_dir = tmp_path / "results"

    train_object_detection.infer_vit_detector(
        checkpoint_path="mock-checkpoint.pth",
        dataset_path=str(images_dir),
        out_path=str(out_dir),
        visualize=False,
        device=torch.device("cpu"),
        split="val",
        confidence=0.42,
    )

    assert dataset_calls == [
        ((str(images_dir), "val", "None"), {"transforms": ({"image_size": 224}, transform)})
    ]
    assert loader_calls == [((dataset,), {"batch_size": 1, "shuffle": False, "drop_last": False})]
    assert len(inferer_calls) == 2
    model_arg, loader_arg, device_arg = inferer_calls[0]
    assert model_arg.head.config.line_class_id == 7
    assert loader_arg is dataset
    assert device_arg == torch.device("cpu")
    assert inferer_calls[1] == {"score_threshold": 0.42}

    assert (out_dir / "frame.png").read_bytes() == b"synthetic frame.png"
    rows = (out_dir / "frame.txt").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert [int(row.split()[0]) for row in rows] == [2, 7, 8]
    assert [float(value) for value in rows[0].split()[1:]] == pytest.approx([0.1, 0.2, 0.7, 0.8])
    assert [float(value) for value in rows[1].split()[1:]] == pytest.approx([0.125, 0.25, 0.875, 0.5])
    assert [float(value) for value in rows[2].split()[1:]] == pytest.approx([0.125, 0.25, 0.5, 0.625, 0.75, 0.875])
    assert (out_dir / "empty.png").read_bytes() == b"synthetic empty.png"
    assert (out_dir / "empty.txt").read_text(encoding="utf-8") == ""


def test_visualization_uses_core_visualizer_and_complete_special_class_mapping(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame.png").write_bytes(b"image")
    dataset = SimpleNamespace(images=["frame.png"], images_dir=images_dir)
    model = SimpleNamespace(
        eval=lambda: None,
        head=SimpleNamespace(config=SimpleNamespace(line_class_id=None, ellipse_class_id=8)),
    )
    monkeypatch.setattr(train_object_detection, "create_model_from_checkpoint", lambda _path: model)
    monkeypatch.setattr(train_object_detection, "ODValTransforms", lambda **_kwargs: object())
    monkeypatch.setattr(train_object_detection, "MaesyDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(train_object_detection, "DataLoader", lambda *args, **_kwargs: object())
    monkeypatch.setattr(
        train_object_detection,
        "Inferer",
        lambda **_kwargs: SimpleNamespace(
            infer=lambda **_options: ([{
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.long),
                "scores": torch.empty((0,)),
            }], [])
        ),
    )
    monkeypatch.setattr(train_object_detection, "tqdm", lambda iterable: iterable)
    visualization_calls = []

    from _maesy_core.dataset import visualizer

    monkeypatch.setattr(visualizer, "visualize_data", lambda *args, **kwargs: visualization_calls.append((args, kwargs)))

    out_dir = tmp_path / "output"
    train_object_detection.infer_vit_detector(
        checkpoint_path="mock-checkpoint.pth",
        dataset_path=str(images_dir),
        out_path=str(out_dir),
        visualize=True,
        device=torch.device("cpu"),
        split="",
        confidence=0.0,
    )

    assert visualization_calls == [(
        (str(out_dir), ""),
        {
            "special_classes": {"lines": -1, "ellipses": 8},
            "enable_lines": False,
            "enable_ellipses": True,
        },
    )]


def test_inference_omitted_confidence_uses_model_default(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame.jpg").write_bytes(b"image")
    dataset = SimpleNamespace(images=["frame.jpg"], images_dir=images_dir)
    model = SimpleNamespace(eval=lambda: None, head=SimpleNamespace(config=SimpleNamespace()))
    monkeypatch.setattr(train_object_detection, "create_model_from_checkpoint", lambda _path: model)
    monkeypatch.setattr(train_object_detection, "ODValTransforms", lambda **_kwargs: object())
    monkeypatch.setattr(train_object_detection, "MaesyDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(train_object_detection, "DataLoader", lambda *args, **_kwargs: object())
    infer_calls = []

    class FakeInferer:
        def __init__(self, **_kwargs):
            pass

        def infer(self, **kwargs):
            infer_calls.append(kwargs)
            return ([{
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.long),
                "scores": torch.empty((0,)),
            }], [])

    monkeypatch.setattr(train_object_detection, "Inferer", FakeInferer)
    monkeypatch.setattr(train_object_detection, "tqdm", lambda iterable: iterable)

    train_object_detection.infer_vit_detector(
        checkpoint_path="mock-checkpoint.pth",
        dataset_path=str(images_dir),
        out_path=str(tmp_path / "output"),
        visualize=False,
        device=torch.device("cpu"),
        split="",
        confidence=None,
    )

    assert infer_calls == [{}]


def test_image_folder_inference_converts_images_dir_to_path(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame.jpg").write_bytes(b"image")
    dataset = SimpleNamespace(images=["frame.jpg"], images_dir=str(images_dir))
    model = SimpleNamespace(eval=lambda: None, head=SimpleNamespace(config=SimpleNamespace()))
    monkeypatch.setattr(train_object_detection, "create_model_from_checkpoint", lambda _path: model)
    monkeypatch.setattr(train_object_detection, "ODValTransforms", lambda **_kwargs: object())
    monkeypatch.setattr(train_object_detection, "MaesyDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(train_object_detection, "DataLoader", lambda *args, **_kwargs: object())
    monkeypatch.setattr(
        train_object_detection,
        "Inferer",
        lambda **_kwargs: SimpleNamespace(
            infer=lambda **_options: ([{
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.long),
                "scores": torch.empty((0,)),
            }], [])
        ),
    )
    monkeypatch.setattr(train_object_detection, "tqdm", lambda iterable: iterable)

    out_dir = tmp_path / "output"
    train_object_detection.infer_vit_detector(
        checkpoint_path="mock-checkpoint.pth",
        dataset_path=str(images_dir),
        out_path=str(out_dir),
        visualize=False,
        device=torch.device("cpu"),
        split="",
        confidence=0.0,
    )

    assert (out_dir / "frame.jpg").read_bytes() == b"image"
    assert (out_dir / "frame.txt").read_text(encoding="utf-8") == ""
    assert isinstance(dataset.images_dir, type(images_dir))
