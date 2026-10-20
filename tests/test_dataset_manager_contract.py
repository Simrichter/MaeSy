import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from PIL import Image


def _load_dataset_manager(monkeypatch):
    """Load the manager without importing unrelated optional dataset/model APIs."""
    package = types.ModuleType("_maesy_core")
    package.__path__ = [str(Path(__file__).parents[1] / "_maesy_core")]
    dataset_package = types.ModuleType("_maesy_core.dataset")
    dataset_package.__path__ = [str(Path(__file__).parents[1] / "_maesy_core" / "dataset")]
    package.dataset = dataset_package
    monkeypatch.setitem(sys.modules, "_maesy_core", package)
    monkeypatch.setitem(sys.modules, "_maesy_core.dataset", dataset_package)

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
    decomposition.PCA = type("PCA", (), {"__init__": lambda self, **_kwargs: None})
    monkeypatch.setitem(sys.modules, "skimage", skimage)
    monkeypatch.setitem(sys.modules, "skimage.measure", measure)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", decomposition)

    converter_source = Path(__file__).parents[1] / "_maesy_core" / "dataset" / "converter.py"
    converter_spec = importlib.util.spec_from_file_location(
        "_maesy_core.dataset.converter", converter_source
    )
    converter = importlib.util.module_from_spec(converter_spec)
    monkeypatch.setitem(sys.modules, converter_spec.name, converter)
    converter_spec.loader.exec_module(converter)
    dataset_package.datumaro_to_devils_yolo = converter.datumaro_to_devils_yolo
    dataset_package.robert_to_devils_yolo = converter.robert_to_devils_yolo

    source = Path(__file__).parents[1] / "_maesy_core" / "dataset" / "dataset_manager.py"
    spec = importlib.util.spec_from_file_location("_maesy_core.dataset.dataset_manager", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    dataset_package.dataset_manager = module
    return module.DatasetManager


def _write_datumaro(root: Path, items: list[dict]) -> Path:
    dataset = root / "datumaro"
    images = dataset / "images" / "default"
    annotations = dataset / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for item in items:
        image = images / item["id"]
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 20), color=(30, 60, 90)).save(image)
    (annotations / "default.json").write_text(
        json.dumps({
            "categories": {"label": {"labels": [{"name": "ball"}]}},
            "items": items,
        }),
        encoding="utf-8",
    )
    return dataset


def _item(image_id: str) -> dict:
    return {
        "id": image_id,
        "image": {"size": [20, 40]},
        "annotations": [{"type": "bbox", "label_id": 0, "bbox": [4, 2, 20, 10]}],
    }


def _create_manager_dataset(manager, source, name, **overrides):
    options = {
        "folder_names": [str(source)],
        "chosen_paths": [],
        "dataset_name": name,
        "split_percentages": [1.0, 0.0, 0.0],
        "resize": None,
        "with_labels": True,
        "step": 1,
        "start_index": 0,
        "del_folders": False,
        "cluster_method": None,
        "left_right": False,
        "convert": "datumaro",
        "nc": None,
        "class_names": None,
        "special_classes": None,
    }
    options.update(overrides)
    return manager.create_dataset(**options)


def test_create_dataset_conversion_keeps_nested_image_label_pairs_and_loadable_yaml(
    tmp_path, monkeypatch
):
    DatasetManager = _load_dataset_manager(monkeypatch)
    conversion = pytest.importorskip("_maesy_core.dataset.converter")
    monkeypatch.setattr(conversion, "EllipseModel", type("NoEllipse", (), {"estimate": lambda self, _points: False}))
    source = _write_datumaro(tmp_path, [_item("nested/one/frame.jpg")])
    manager = DatasetManager(data_root=str(tmp_path / "output"))

    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)
    result = _create_manager_dataset(manager, source, "converted")

    image = result / "train" / "images" / "nested" / "one" / "frame.jpg"
    label = result / "train" / "labels" / "nested" / "one" / "frame.txt"
    assert image.is_file()
    assert label.read_text(encoding="utf-8").startswith("0 ")
    metadata = yaml.safe_load((result / "dataset.yaml").read_text(encoding="utf-8"))
    assert metadata["names"] == ["ball"]
    assert metadata["nc"] == 1
    assert (source / "images" / "default" / "nested" / "one" / "frame.jpg").is_file()


def test_create_dataset_renames_colliding_images_and_labels_as_pairs(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    conversion = pytest.importorskip("_maesy_core.dataset.converter")
    monkeypatch.setattr(conversion, "EllipseModel", type("NoEllipse", (), {"estimate": lambda self, _points: False}))
    first = _write_datumaro(tmp_path / "src-a", [_item("frame.jpg")])
    second_item = _item("frame.jpg")
    second_item["annotations"][0]["bbox"] = [2, 2, 8, 6]
    second = _write_datumaro(tmp_path / "src-b", [second_item])
    manager = DatasetManager(data_root=str(tmp_path / "output"))

    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)
    result = _create_manager_dataset(
        manager,
        first,
        "converted",
        folder_names=[str(first), str(second)],
    )

    train_images = result / "train" / "images"
    train_labels = result / "train" / "labels"
    images = sorted(path.name for path in train_images.rglob("*.jpg"))
    labels = sorted(path.name for path in train_labels.rglob("*.txt"))
    assert images == ["frame.jpg", "frame_copy.jpg"]
    assert labels == ["frame.txt", "frame_copy.txt"]
    assert (train_labels / "frame.txt").read_text(encoding="utf-8") == "0 0.1 0.1 0.6 0.6\n"
    assert (train_labels / "frame_copy.txt").read_text(encoding="utf-8") == "0 0.05 0.1 0.25 0.4\n"


def test_robert_converter_keeps_unlabeled_background_images(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    conversion = pytest.importorskip("_maesy_core.dataset.converter")
    monkeypatch.setattr(conversion, "EllipseModel", type("NoEllipse", (), {"estimate": lambda self, _points: False}))

    source = tmp_path / "robert"
    images = source / "train" / "images"
    labels = source / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (32, 16)).save(images / "with_box.JPG")
    Image.new("RGB", (32, 16)).save(images / "background.png")
    (labels / "with_box.txt").write_text(
        "BoundingBoxes:\nK1(Clone): 0.5 0.5 0.2 0.4\n", encoding="utf-8"
    )

    output = tmp_path / "robert-output"
    _root, count, names, _special = conversion.robert_to_devils_yolo(
        source, output_path=output
    )

    assert count == 5
    assert "K1(Clone)" in names
    assert (output / "train" / "images" / "with_box.JPG").is_file()
    assert (output / "train" / "images" / "background.png").is_file()
    assert (output / "train" / "labels" / "with_box.txt").read_text().startswith("1 ")
    assert (output / "train" / "labels" / "background.txt").read_text() == ""


def test_create_dataset_converted_clustering_selection_preserves_label_pair(
    tmp_path, monkeypatch
):
    DatasetManager = _load_dataset_manager(monkeypatch)
    conversion = pytest.importorskip("_maesy_core.dataset.converter")
    monkeypatch.setattr(conversion, "EllipseModel", type("NoEllipse", (), {"estimate": lambda self, _points: False}))
    source = _write_datumaro(tmp_path, [_item("one.jpg"), _item("two.jpg")])
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    selected = []

    def choose_one(paths, _chosen_paths, **_kwargs):
        roots = [Path(path) for path in paths]
        available = sorted(roots[0].rglob("*.jpg"))
        selected.append(available[1])
        return [available[1]]

    monkeypatch.setattr(manager, "cluster_data", choose_one)
    result = _create_manager_dataset(
        manager,
        source,
        "clustered",
        cluster_method="resnet_kmeans",
        split_percentages=[1, 0, 0],
    )

    images = list((result / "train" / "images").rglob("*.jpg"))
    labels = list((result / "train" / "labels").rglob("*.txt"))
    assert len(images) == len(labels) == 1
    assert images[0].stem == labels[0].stem == selected[0].stem
    assert labels[0].read_text(encoding="utf-8").startswith("0 ")


@pytest.mark.parametrize(
    ("num_images", "expected_counts"),
    [(1, [1, 0, 0]), (2, [1, 1, 0]), (4, [2, 1, 1])],
)
def test_small_datasets_keep_nonzero_splits_populated_when_possible(
    tmp_path, monkeypatch, num_images, expected_counts
):
    DatasetManager = _load_dataset_manager(monkeypatch)
    source = tmp_path / "images"
    source.mkdir()
    for index in range(num_images):
        Image.new("RGB", (8, 8)).save(source / f"frame_{index}.png")
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)

    result = manager.create_dataset(
        folder_names=[str(source)], chosen_paths=[], dataset_name="small-split",
        split_percentages=[0.8, 0.1, 0.1], resize=None, with_labels=False,
        step=1, start_index=0, del_folders=False, cluster_method=None,
        left_right=False, nc=0, class_names=[],
    )

    actual_counts = [
        len(list((result / split / "images").iterdir()))
        for split in ("train", "val", "test")
    ]
    assert actual_counts == expected_counts


def test_split_rounding_does_not_lose_test_examples(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    source = tmp_path / "images"
    source.mkdir()
    for index in range(11):
        Image.new("RGB", (8, 8)).save(source / f"frame_{index}.png")
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)

    result = manager.create_dataset(
        folder_names=[str(source)], chosen_paths=[], dataset_name="split",
        split_percentages=[0.8, 0.1, 0.1], resize=None, with_labels=False,
        step=1, start_index=0, del_folders=False, cluster_method=None,
        left_right=False, nc=0, class_names=[],
    )

    assert len(list((result / "train" / "images").iterdir())) == 9
    assert len(list((result / "val" / "images").iterdir())) == 1
    assert len(list((result / "test" / "images").iterdir())) == 1


def test_zero_percentage_split_stays_empty_after_rounding(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    source = tmp_path / "images"
    source.mkdir()
    for index in range(5):
        Image.new("RGB", (8, 8)).save(source / f"frame_{index}.png")
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)

    result = manager.create_dataset(
        folder_names=[str(source)], chosen_paths=[], dataset_name="zero-test",
        split_percentages=[0.9, 0.1, 0], resize=None, with_labels=False,
        step=1, start_index=0, del_folders=False, cluster_method=None,
        left_right=False, nc=0, class_names=[],
    )

    assert len(list((result / "train" / "images").iterdir())) == 4
    assert len(list((result / "val" / "images").iterdir())) == 1
    assert list((result / "test" / "images").iterdir()) == []


def test_create_dataset_copies_standard_sibling_labels_with_matching_images(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    source = tmp_path / "dataset" / "train" / "images"
    labels = source.parent / "labels"
    source.mkdir(parents=True)
    labels.mkdir()
    for name, label_text in (("one.png", "0 0.1 0.2 0.3 0.4\n"), ("two.png", "1 0.5 0.6 0.7 0.8\n")):
        Image.new("RGB", (8, 8)).save(source / name)
        (labels / f"{Path(name).stem}.txt").write_text(label_text, encoding="utf-8")
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)

    result = manager.create_dataset(
        folder_names=[str(source)], chosen_paths=[], dataset_name="labeled",
        split_percentages=[1, 0, 0], resize=None, with_labels=True, step=1,
        start_index=0, del_folders=False, cluster_method=None, left_right=False,
        nc=2, class_names=["first", "second"],
    )

    assert (result / "train" / "labels" / "one.txt").read_text(encoding="utf-8") == "0 0.1 0.2 0.3 0.4\n"
    assert (result / "train" / "labels" / "two.txt").read_text(encoding="utf-8") == "1 0.5 0.6 0.7 0.8\n"


def test_create_dataset_omits_labels_when_disabled(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (8, 8)).save(source / "frame.png")
    (source.parent / "labels").mkdir()
    (source.parent / "labels" / "frame.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    manager = DatasetManager(data_root=str(tmp_path / "output"))

    result = manager.create_dataset(
        folder_names=[str(source)], chosen_paths=[], dataset_name="image-only",
        split_percentages=[1, 0, 0], resize=None, with_labels=False, step=1,
        start_index=0, del_folders=False, cluster_method=None, left_right=False,
        nc=0, class_names=[],
    )

    assert (result / "train" / "images" / "frame.png").is_file()
    assert not (result / "train" / "labels").exists()


def test_create_dataset_left_right_keeps_views_and_labels_in_same_split(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    root = tmp_path / "stereo"
    right = root / "Right"
    left = root / "Left"
    right.mkdir(parents=True)
    left.mkdir()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(right / "frame_right.png")
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(left / "frame_left.png")
    (right / "frame_right.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (left / "frame_left.txt").write_text("1 0.5 0.6 0.7 0.8\n", encoding="utf-8")
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)

    result = manager.create_dataset(
        folder_names=[str(right)], chosen_paths=[], dataset_name="stereo-output",
        split_percentages=[1, 0, 0], resize=None, with_labels=True, step=1,
        start_index=0, del_folders=False, cluster_method=None, left_right=True,
        nc=2, class_names=["right-class", "left-class"],
    )

    split = result / "train"
    assert (split / "images" / "frame_right.png").is_file()
    assert (split / "images" / "frame_left.png").is_file()
    assert (split / "labels" / "frame_right.txt").read_text(encoding="utf-8").startswith("0 ")
    assert (split / "labels" / "frame_left.txt").read_text(encoding="utf-8").startswith("1 ")
    assert not list((result / "val" / "images").iterdir())


def test_create_dataset_left_right_fails_clearly_when_counterpart_is_missing(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    root = tmp_path / "stereo"
    right = root / "Right"
    right.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(right / "frame_right.png")
    manager = DatasetManager(data_root=str(tmp_path / "output"))

    with pytest.raises(FileNotFoundError, match="corresponding left-camera image"):
        manager.create_dataset(
            folder_names=[str(right)], chosen_paths=[], dataset_name="stereo-output",
            split_percentages=[1, 0, 0], resize=None, with_labels=False, step=1,
            start_index=0, del_folders=False, cluster_method=None, left_right=True,
            nc=0, class_names=[],
        )

    assert not (tmp_path / "output" / "stereo-output").exists()


def test_converted_sampling_applies_per_source_root(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    conversion = pytest.importorskip("_maesy_core.dataset.converter")
    monkeypatch.setattr(conversion, "EllipseModel", type("NoEllipse", (), {"estimate": lambda self, _points: False}))
    first = _write_datumaro(tmp_path / "src-a", [_item(f"{i}.jpg") for i in range(3)])
    second = _write_datumaro(tmp_path / "src-b", [_item(f"{i}.jpg") for i in range(3)])
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)

    result = _create_manager_dataset(
        manager, first, "sampled", folder_names=[str(first), str(second)],
        step=2, start_index=0,
    )

    assert len(list((result / "train" / "images").glob("*.jpg"))) == 4
    assert len(list((result / "train" / "labels").glob("*.txt"))) == 4


def test_robert_conversion_preserves_all_input_splits_through_manager(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    source = tmp_path / "robert"
    for split in ("train", "val", "test"):
        images = source / split / "images"
        labels = source / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (32, 16)).save(images / f"{split}.png")
        (labels / f"{split}.txt").write_text(
            "BoundingBoxes:\nK1(Clone): 0.5 0.5 0.2 0.4\n", encoding="utf-8"
        )
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)

    result = manager.create_dataset(
        folder_names=[str(source)], chosen_paths=[], dataset_name="robert-combined",
        split_percentages=[1, 0, 0], resize=None, with_labels=True, step=1,
        start_index=0, del_folders=False, cluster_method=None, left_right=False,
        convert="robert",
    )

    train_images = list((result / "train" / "images").glob("*.png"))
    train_labels = list((result / "train" / "labels").glob("*.txt"))
    assert len(train_images) == len(train_labels) == 3
    assert {path.stem for path in train_images} == {"train", "val", "test"}
    assert all(label.read_text(encoding="utf-8").startswith("1 ") for label in train_labels)


@pytest.mark.parametrize("dataset_name", ["../escaped", "/tmp/escaped-dataset"])
def test_create_dataset_rejects_destination_outside_data_root(tmp_path, monkeypatch, dataset_name):
    DatasetManager = _load_dataset_manager(monkeypatch)
    manager = DatasetManager(data_root=str(tmp_path / "output"))

    with pytest.raises(ValueError, match="inside data_root"):
        manager.create_dataset(
            folder_names=[], chosen_paths=[], dataset_name=dataset_name,
            split_percentages=[1, 0, 0], resize=None, with_labels=False,
            step=1, start_index=0, del_folders=False, cluster_method=None,
            left_right=False, nc=0, class_names=[],
        )

    assert not (tmp_path / "escaped").exists()
    if dataset_name == "/tmp/escaped-dataset":
        assert not Path(dataset_name).exists()


@pytest.mark.parametrize("resize", [[], [0], [-1, 10], [10, 10, 10], [2.5], [True]])
def test_invalid_resize_is_rejected_before_output_creation(tmp_path, monkeypatch, resize):
    DatasetManager = _load_dataset_manager(monkeypatch)
    manager = DatasetManager(data_root=str(tmp_path / "output"))

    with pytest.raises(ValueError, match="resize must be None or"):
        manager.create_dataset(
            folder_names=[], chosen_paths=[], dataset_name="bad-resize",
            split_percentages=[1, 0, 0], resize=resize, with_labels=False,
            step=1, start_index=0, del_folders=False, cluster_method=None,
            left_right=False, nc=0, class_names=[],
        )

    assert not (tmp_path / "output" / "bad-resize").exists()
    assert not list((tmp_path / "output").glob(".bad-resize.staging-*"))


def test_failed_dataset_copy_cleans_staging_and_allows_retry(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (8, 8)).save(source / "frame.png")
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    monkeypatch.setattr("_maesy_core.dataset.dataset_manager.random.shuffle", lambda items: None)
    original_copy_resize = DatasetManager._copy_resize

    def fail_copy(*_args, **_kwargs):
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(DatasetManager, "_copy_resize", staticmethod(fail_copy))
    with pytest.raises(OSError, match="synthetic copy failure"):
        manager.create_dataset(
            folder_names=[str(source)], chosen_paths=[], dataset_name="retryable",
            split_percentages=[1, 0, 0], resize=None, with_labels=False,
            step=1, start_index=0, del_folders=False, cluster_method=None,
            left_right=False, nc=0, class_names=[],
        )
    assert not (tmp_path / "output" / "retryable").exists()
    assert not list((tmp_path / "output").glob(".retryable.staging-*"))

    monkeypatch.setattr(DatasetManager, "_copy_resize", staticmethod(original_copy_resize))
    result = manager.create_dataset(
        folder_names=[str(source)], chosen_paths=[], dataset_name="retryable",
        split_percentages=[1, 0, 0], resize=None, with_labels=False,
        step=1, start_index=0, del_folders=False, cluster_method=None,
        left_right=False, nc=0, class_names=[],
    )
    assert (result / "train" / "images" / "frame.png").is_file()
    assert (result / "dataset.yaml").is_file()


def test_create_dataset_rejects_obb_before_creating_output(tmp_path, monkeypatch):
    DatasetManager = _load_dataset_manager(monkeypatch)
    manager = DatasetManager(data_root=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="OBB annotations are not supported"):
        _create_manager_dataset(manager, tmp_path / "unused", "obb", convert="obb")
    assert not (tmp_path / "output" / "obb").exists()
