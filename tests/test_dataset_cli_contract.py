import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image


def _install_package_stubs(monkeypatch):
    dataset_dir = Path(__file__).parents[1] / "_maesy_core" / "dataset"
    core_package = types.ModuleType("_maesy_core")
    core_package.__path__ = [str(dataset_dir.parent)]
    dataset_package = types.ModuleType("_maesy_core.dataset")
    dataset_package.__path__ = [str(dataset_dir)]
    dataset_manager = types.ModuleType("_maesy_core.dataset.dataset_manager")
    dataset_manager.DatasetManager = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "_maesy_core", core_package)
    monkeypatch.setitem(sys.modules, "_maesy_core.dataset", dataset_package)
    monkeypatch.setitem(sys.modules, "_maesy_core.dataset.dataset_manager", dataset_manager)


def _load_cli_module(monkeypatch):
    import importlib.util

    _install_package_stubs(monkeypatch)
    source = Path(__file__).parents[1] / "_maesy_core" / "dataset" / "cli_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "_maesy_core.dataset.cli_dataset", source
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _load_converter_module(monkeypatch):
    import importlib.util

    _install_package_stubs(monkeypatch)

    # The test data exercises boxes only, so lightweight stand-ins avoid requiring
    # the optional geometry libraries in this isolated CLI contract test.
    skimage = types.ModuleType("skimage")
    skimage.__path__ = []
    measure = types.ModuleType("skimage.measure")

    class EllipseModel:
        params = (0.5, 0.5, 0.1, 0.1, 0.0)

        def estimate(self, _points):
            return False

    measure.EllipseModel = EllipseModel
    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []
    decomposition = types.ModuleType("sklearn.decomposition")

    class PCA:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _points):
            self.components_ = [[1.0, 0.0], [0.0, 1.0]]

    decomposition.PCA = PCA
    monkeypatch.setitem(sys.modules, "skimage", skimage)
    monkeypatch.setitem(sys.modules, "skimage.measure", measure)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", decomposition)

    source = Path(__file__).parents[1] / "_maesy_core" / "dataset" / "converter.py"
    spec = importlib.util.spec_from_file_location(
        "_maesy_core.dataset.converter", source
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _datumaro_tree(root: Path) -> Path:
    dataset = root / "datumaro"
    annotations = dataset / "annotations"
    images = dataset / "images" / "default" / "nested"
    annotations.mkdir(parents=True)
    images.mkdir(parents=True)
    Image.new("RGB", (40, 20), color=(30, 60, 90)).save(images / "frame.jpg")
    payload = {
        "categories": {"label": {"labels": [{"name": "ball"}]}},
        "items": [
            {
                "id": "nested/frame.jpg",
                "image": {"size": [20, 40]},
                "annotations": [
                    {"type": "bbox", "label_id": 0, "bbox": [4, 2, 20, 10]}
                ],
            }
        ],
    }
    (annotations / "default.json").write_text(json.dumps(payload), encoding="utf-8")
    return dataset


def _robert_tree(root: Path) -> Path:
    dataset = root / "robert"
    labels = dataset / "train" / "labels" / "nested"
    labels.mkdir(parents=True)
    images = dataset / "train" / "images" / "nested"
    images.mkdir(parents=True)
    Image.new("RGB", (40, 20), color=(30, 60, 90)).save(images / "frame.jpg")
    (labels / "frame.txt").write_text(
        "BoundingBoxes:\nK1(Clone): 0.5 0.5 0.2 0.4\n", encoding="utf-8"
    )
    return dataset


def _args(input_format: str, input_path: Path, output_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        command="convert",
        path=str(input_path),
        input_format=input_format,
        output_path=str(output_path),
        format="xyxy" if input_format == "obb" else "cxcywh",
        convert_id_blacklist=[],
        convert_merge_ids=[],
        convert_permute_ids=[],
    )


def test_datumaro_converter_copies_nested_images_and_labels_with_consistent_metadata(
    tmp_path, monkeypatch
):
    cli_dataset = _load_cli_module(monkeypatch)
    converter = _load_converter_module(monkeypatch)
    source = _datumaro_tree(tmp_path)
    output = tmp_path / "generated" / "datumaro-output"

    cli_dataset.main(_args("datumaro", source, output))

    image = output / "default" / "images" / "nested" / "frame.jpg"
    label = output / "default" / "labels" / "nested" / "frame.txt"
    assert image.is_file()
    assert [float(value) for value in label.read_text(encoding="utf-8").split()] == pytest.approx(
        [0, 0.35, 0.35, 0.5, 0.5]
    )
    metadata = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert metadata["names"] == ["ball"]
    assert metadata["nc"] == 1
    assert metadata["box_format"] == "cxcywh"
    assert metadata["default"] == "default"
    assert Path(metadata["path"]).resolve() == output.resolve()
    assert (source / "images" / "default" / "nested" / "frame.jpg").is_file()
    assert not (source / "images" / "default" / "nested" / "frame.txt").exists()
    assert converter._datumaro_item_path(output / "default" / "labels", "nested/frame.jpg") == label


def test_robert_converter_copies_images_and_labels_into_dataset_tree(tmp_path, monkeypatch):
    cli_dataset = _load_cli_module(monkeypatch)
    _load_converter_module(monkeypatch)
    source = _robert_tree(tmp_path)
    output = tmp_path / "created" / "robert-output"
    output.parent.mkdir()

    cli_dataset.main(_args("robert", source, output))

    image = output / "train" / "images" / "nested" / "frame.jpg"
    label = output / "train" / "labels" / "nested" / "frame.txt"
    assert image.is_file()
    assert label.read_text(encoding="utf-8").startswith("1 ")
    metadata = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert metadata["train"] == "train"
    assert metadata["names"][1] == "K1(Clone)"
    assert metadata["box_format"] == "cxcywh"
    assert (source / "train" / "labels" / "nested" / "frame.txt").is_file()
    assert (source / "train" / "images" / "nested" / "frame.jpg").is_file()


def test_dataset_create_parser_forwards_destination_and_options(tmp_path, monkeypatch):
    cli_dataset = _load_cli_module(monkeypatch)
    captured = {}

    class Manager:
        def __init__(self, data_root):
            captured["data_root"] = data_root

        def create_dataset(self, **kwargs):
            captured["options"] = kwargs

    monkeypatch.setattr(
        sys.modules["_maesy_core.dataset.dataset_manager"], "DatasetManager", Manager
    )
    command_line = __import__("runpy").run_path(
        str(Path(__file__).parents[1] / "maesy" / "command_line.py")
    )
    args = command_line["parser"].parse_args([
        "dataset", "--path", str(tmp_path / "default-root"), "create",
        str(tmp_path / "source"), "created", "--output-path", str(tmp_path / "chosen-root"),
        "--no-labels", "--format", "cxcywh", "--step", "2",
    ])

    cli_dataset.main(args)

    assert captured["data_root"] == str(tmp_path / "chosen-root")
    assert captured["options"]["folder_names"] == [str(tmp_path / "source")]
    assert captured["options"]["dataset_name"] == "created"
    assert captured["options"]["with_labels"] is False
    assert captured["options"]["box_format"] == "cxcywh"
    assert captured["options"]["step"] == 2
    assert captured["options"]["del_folders"] is False


def test_dataset_create_delete_option_fails_before_manager_instantiation(tmp_path, monkeypatch):
    cli_dataset = _load_cli_module(monkeypatch)

    def unexpected_manager(**_kwargs):
        raise AssertionError("manager must not be instantiated for unsupported --delete")

    monkeypatch.setattr(
        sys.modules["_maesy_core.dataset.dataset_manager"], "DatasetManager", unexpected_manager
    )
    command_line = __import__("runpy").run_path(
        str(Path(__file__).parents[1] / "maesy" / "command_line.py")
    )
    args = command_line["parser"].parse_args([
        "dataset", "create", str(tmp_path / "source"), "created", "--delete",
    ])

    with pytest.raises(NotImplementedError, match="--delete is unsupported"):
        cli_dataset.main(args)


@pytest.mark.parametrize(
    ("input_format", "function_name", "input_keyword"),
    [
        ("datumaro", "datumaro_to_devils_yolo", "datumaro_dir"),
        ("robert", "robert_to_devils_yolo", "dataset_dir"),
        ("obb", "datumaro_to_ultralyticsOBB", "datumaro_dir"),
    ],
)
def test_convert_forwards_options_by_keyword(monkeypatch, tmp_path, input_format, function_name, input_keyword):
    cli_dataset = _load_cli_module(monkeypatch)
    converter_module = types.ModuleType("_maesy_core.dataset.converter")
    captured = {}

    def fake_converter(**kwargs):
        captured.update(kwargs)

    setattr(converter_module, function_name, fake_converter)
    monkeypatch.setitem(sys.modules, "_maesy_core.dataset.converter", converter_module)
    args = _args(input_format, tmp_path / "input", tmp_path / "unused")
    args.format = "xyxy"
    args.convert_id_blacklist = [7]
    args.convert_merge_ids = [1, 2, 3, 4]
    args.convert_permute_ids = [1, 0]

    cli_dataset.main(args)

    assert captured == {
        input_keyword: str(args.path),
        "box_format": "xyxy",
        "class_id_blacklist": [7],
        "merge_ids": {1: 2, 3: 4},
        "permute_ids": [1, 0],
        "output_path": str(args.output_path),
    }


def test_standalone_convert_requires_explicit_output_path():
    import runpy

    command_line = runpy.run_path(str(Path(__file__).parents[1] / "maesy" / "command_line.py"))
    with pytest.raises(SystemExit) as error:
        command_line["parser"].parse_args(["dataset", "convert", "input"])
    assert error.value.code == 2


def test_obb_converter_writes_parseable_four_corner_labels(tmp_path, monkeypatch):
    converter = _load_converter_module(monkeypatch)
    source = tmp_path / "obb-source"
    images = source / "images" / "default"
    annotations = source / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (40, 20), color=(20, 40, 60)).save(images / "frame.jpg")
    (annotations / "default.json").write_text(
        json.dumps({
            "categories": {"label": {"labels": [{"name": "goal"}]}},
            "items": [{
                "id": "frame.jpg", "image": {"size": [20, 40]},
                "annotations": [{
                    "type": "polygon", "label_id": 0,
                    "points": [10, 2, 30, 4, 28, 14, 8, 12],
                }],
            }],
        }),
        encoding="utf-8",
    )
    output = tmp_path / "obb-output"

    converter.datumaro_to_ultralyticsOBB(source, output_path=output)

    metadata = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert metadata["names"] == ["goal"]
    assert metadata["nc"] == 1
    assert metadata["path"] == str(output.resolve())
    assert (output / "images" / "default" / "frame.jpg").is_file()
    row = (output / "labels" / "default" / "frame.txt").read_text(encoding="utf-8").split()
    assert len(row) == 9
    assert row[0] == "0"
    assert [float(value) for value in row[1:]] == pytest.approx(
        [0.25, 0.1, 0.75, 0.2, 0.7, 0.7, 0.2, 0.6]
    )


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        ({"type": "polygon", "label_id": 0, "points": [0, 0, float("nan"), 2, 3, 4, 5, 6]}, "polygon coordinates must be finite"),
        ({"type": "bbox", "label_id": 0, "bbox": [0, 0, float("inf"), 2]}, "four finite numbers"),
    ],
)
def test_obb_converter_rejects_non_finite_annotation_coordinates_before_output(
    tmp_path, monkeypatch, annotation, message
):
    converter = _load_converter_module(monkeypatch)
    source = tmp_path / "obb-source"
    images = source / "images" / "default"
    annotations = source / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (40, 20)).save(images / "frame.jpg")
    (annotations / "default.json").write_text(
        json.dumps({
            "categories": {"label": {"labels": [{"name": "goal"}]}},
            "items": [{
                "id": "frame.jpg", "image": {"size": [20, 40]},
                "annotations": [annotation],
            }],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        converter.datumaro_to_ultralyticsOBB(source, output_path=tmp_path / "out")

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "item_id",
    ["../outside.jpg", "/tmp/outside.jpg", "nested/../../outside.jpg"],
)
def test_datumaro_converter_rejects_unsafe_item_paths_before_creating_output(
    tmp_path, monkeypatch, item_id
):
    converter = _load_converter_module(monkeypatch)
    source = tmp_path / "unsafe-datumaro"
    images = source / "images" / "default"
    annotations = source / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (40, 20)).save(images / "frame.jpg")
    (annotations / "default.json").write_text(
        json.dumps({
            "categories": {"label": {"labels": [{"name": "ball"}]}},
            "items": [{
                "id": item_id, "image": {"size": [20, 40]},
                "annotations": [],
            }],
        }),
        encoding="utf-8",
    )

    output = tmp_path / "out"
    with pytest.raises(ValueError, match="safe relative path"):
        converter.datumaro_to_devils_yolo(source, output_path=output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        ({"type": "bbox", "label_id": 0, "bbox": [0, 0, float("nan"), 2]}, "Invalid Datumaro bounding box"),
        ({"type": "polyline", "label_id": 0, "points": [0, 0, float("inf"), 2]}, "Invalid Datumaro polyline"),
        ({"type": "polygon", "label_id": 0, "points": [0, 0, 1, 0, 2, 1, 1, 2, float("nan"), 1]}, "Invalid Datumaro polygon"),
    ],
)
def test_datumaro_converter_rejects_non_finite_coordinates_before_output(
    tmp_path, monkeypatch, annotation, message
):
    converter = _load_converter_module(monkeypatch)
    source = tmp_path / "nonfinite-datumaro"
    images = source / "images" / "default"
    annotations = source / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (40, 20)).save(images / "frame.jpg")
    (annotations / "default.json").write_text(
        json.dumps({
            "categories": {"label": {"labels": [{"name": "ellipse" if annotation["type"] == "polygon" else "ball"}]}},
            "items": [{
                "id": "frame.jpg", "image": {"size": [20, 40]},
                "annotations": [annotation],
            }],
        }),
        encoding="utf-8",
    )

    output = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        converter.datumaro_to_devils_yolo(source, output_path=output)
    assert not output.exists()


def test_obb_converter_rejects_non_quadrilateral_polygon_without_truncation(tmp_path, monkeypatch):
    converter = _load_converter_module(monkeypatch)
    source = tmp_path / "obb-source"
    images = source / "images" / "default"
    annotations = source / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (40, 20)).save(images / "frame.jpg")
    (annotations / "default.json").write_text(
        json.dumps({
            "categories": {"label": {"labels": [{"name": "goal"}]}},
            "items": [{
                "id": "frame.jpg", "image": {"size": [20, 40]},
                "annotations": [{
                    "type": "polygon", "label_id": 0,
                    "points": [1, 1, 10, 1, 20, 8, 10, 18, 1, 18],
                }],
            }],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corners"):
        converter.datumaro_to_ultralyticsOBB(source, output_path=tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_robert_malformed_annotation_fails_before_creating_output(tmp_path, monkeypatch):
    converter = _load_converter_module(monkeypatch)
    source = _robert_tree(tmp_path)
    label = source / "train" / "labels" / "nested" / "frame.txt"
    label.write_text(
        "BoundingBoxes:\nK1(Clone): 0.5 0.5 nan 0.4\n", encoding="utf-8"
    )
    output = tmp_path / "robert-invalid-output"

    with pytest.raises(ValueError, match="Invalid bounding box"):
        converter.robert_to_devils_yolo(source, output_path=output)

    assert not output.exists()
    assert (source / "train" / "images" / "nested" / "frame.jpg").is_file()


def test_robert_write_failure_removes_partial_conversion_output(tmp_path, monkeypatch):
    converter = _load_converter_module(monkeypatch)
    source = _robert_tree(tmp_path)
    output = tmp_path / "robert-partial-output"
    original_copy = converter.shutil.copy2

    def fail_copy(*_args, **_kwargs):
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(converter.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="synthetic copy failure"):
        converter.robert_to_devils_yolo(source, output_path=output)
    assert not output.exists()
    assert list(tmp_path.glob(".robert-partial-output.*")) == []

    monkeypatch.setattr(converter.shutil, "copy2", original_copy)
    converted, *_ = converter.robert_to_devils_yolo(source, output_path=output)
    assert (converted / "train" / "labels" / "nested" / "frame.txt").is_file()


def test_convert_rejects_unpaired_merge_ids_before_writing(tmp_path, monkeypatch):
    cli_dataset = _load_cli_module(monkeypatch)
    source = _datumaro_tree(tmp_path)
    args = _args("datumaro", source, tmp_path / "should-not-exist")
    args.convert_merge_ids = [1]

    with pytest.raises(ValueError, match="provided in pairs"):
        cli_dataset.main(args)

    assert not (tmp_path / "should-not-exist").exists()


def test_converter_requires_separate_output_and_preserves_source(tmp_path, monkeypatch):
    converter = _load_converter_module(monkeypatch)
    source = _datumaro_tree(tmp_path)
    original_image = source / "images" / "default" / "nested" / "frame.jpg"
    original_bytes = original_image.read_bytes()

    with pytest.raises(ValueError, match="output_path is required"):
        converter.datumaro_to_devils_yolo(source)

    assert original_image.read_bytes() == original_bytes
    assert not (source / "images" / "default" / "nested" / "frame.txt").exists()
