import torch
from PIL import Image

from _maesy_core.inference.inferer import Inferer
from _maesy_core.model import ResnetFeatureExtractor
from _maesy_core.model.resnet_featureextractor import ResnetFeatureExtractorConfig
from clusterdevil.feature_extraction import _find_missing_image_paths, extract_features
from clusterdevil.utils import find_and_load_features, store_features


def _make_model() -> ResnetFeatureExtractor:
    return ResnetFeatureExtractor(
        ResnetFeatureExtractorConfig(resnet_model="resnet18", image_size=64, pretrained=False)
    )


def _make_image(path, seed: int):
    torch.manual_seed(seed)
    arr = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8).numpy()
    Image.fromarray(arr).save(path)


def test_find_missing_image_paths_compares_filenames_only(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"x")
    (folder / "b.png").write_bytes(b"x")
    (folder / "notes.txt").write_bytes(b"x")  # non-image file must be ignored
    subfolder = folder / "nested"
    subfolder.mkdir()
    (subfolder / "c.jpg").write_bytes(b"x")

    stored = {str(folder / "a.jpg"): torch.zeros(4)}
    missing = _find_missing_image_paths(
        [str(folder / "features_abc.feat")], stored, search_roots=[str(folder)]
    )

    assert missing == [str(folder / "b.png"), str(subfolder / "c.jpg")]


def test_find_missing_image_paths_empty_when_up_to_date(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"x")

    stored = {str(folder / "a.jpg"): torch.zeros(4)}
    missing = _find_missing_image_paths([str(folder / "features_abc.feat")], stored)

    assert missing == []


def test_nested_feature_caches_are_found_and_updated_in_their_own_folders(tmp_path):
    root = tmp_path / "dataset"
    folder = root / "session"
    nested_folder = folder / "nested"
    sibling_folder = root / "session_backup"
    nested_folder.mkdir(parents=True)
    sibling_folder.mkdir()
    image = folder / "a.jpg"
    nested_image = nested_folder / "same.jpg"
    sibling_image = sibling_folder / "b.jpg"
    new_image = folder / "same.jpg"
    nested_new_image = nested_folder / "new.jpg"
    for path, seed in ((image, 1), (nested_image, 2), (sibling_image, 3)):
        _make_image(path, seed)

    model_hash = "nested"
    features = {
        str(image): torch.tensor([1.0]),
        str(nested_image): torch.tensor([2.0]),
        str(sibling_image): torch.tensor([3.0]),
    }
    store_features(features, model_hash)

    loaded, feature_files = find_and_load_features([str(root)], model_hash)
    assert set(loaded) == set(features)
    assert set(feature_files) == {
        str(folder / f"features_{model_hash}.feat"),
        str(nested_folder / f"features_{model_hash}.feat"),
        str(sibling_folder / f"features_{model_hash}.feat"),
    }
    assert set(torch.load(folder / f"features_{model_hash}.feat")) == {str(image)}

    _make_image(new_image, 4)
    _make_image(nested_new_image, 5)
    missing = _find_missing_image_paths(feature_files, loaded)
    assert missing == [str(new_image), str(nested_new_image)]

    updated = {
        **loaded,
        str(new_image): torch.tensor([4.0]),
        str(nested_new_image): torch.tensor([5.0]),
    }
    store_features(updated, model_hash)
    folder_cache = torch.load(folder / f"features_{model_hash}.feat")
    nested_cache = torch.load(nested_folder / f"features_{model_hash}.feat")
    sibling_cache = torch.load(sibling_folder / f"features_{model_hash}.feat")
    assert set(folder_cache) == {str(image), str(new_image)}
    assert set(nested_cache) == {str(nested_image), str(nested_new_image)}
    assert set(sibling_cache) == {str(sibling_image)}


def test_extract_features_prunes_deleted_images_and_accepts_explicit_cache(tmp_path):
    model = _make_model()
    folder = tmp_path / "images"
    folder.mkdir()
    kept = folder / "kept.jpg"
    deleted = folder / "deleted.jpg"
    _make_image(kept, 1)
    _make_image(deleted, 2)
    model_hash = model.get_model_hash()
    store_features(
        {str(kept): torch.tensor([1.0]), str(deleted): torch.tensor([2.0])},
        model_hash,
    )
    deleted.unlink()

    cache_path = folder / f"features_{model_hash}.feat"
    loaded = extract_features(model, [str(cache_path)], device="cpu")

    assert set(loaded) == {str(kept)}
    assert set(torch.load(cache_path)) == {str(kept)}


def test_extract_features_removes_cache_when_every_image_is_deleted(tmp_path):
    model = _make_model()
    folder = tmp_path / "images"
    folder.mkdir()
    first = folder / "first.jpg"
    second = folder / "second.jpg"
    _make_image(first, 1)
    _make_image(second, 2)
    model_hash = model.get_model_hash()
    cache_path = folder / f"features_{model_hash}.feat"
    store_features(
        {str(first): torch.tensor([1.0]), str(second): torch.tensor([2.0])},
        model_hash,
    )

    first.unlink()
    second.unlink()
    loaded = extract_features(model, [str(cache_path)], device="cpu")

    assert loaded == {}
    assert not cache_path.exists()
    assert find_and_load_features([str(folder)], model_hash) == ({}, [])


def test_extract_features_rejects_explicit_cache_for_another_model(tmp_path):
    model = _make_model()
    cache_path = tmp_path / "features_other.feat"
    cache_path.write_bytes(b"not a valid cache")

    import pytest

    with pytest.raises(ValueError, match="model hash does not match"):
        extract_features(model, [str(cache_path)], device="cpu")


def test_find_missing_paths_do_not_confuse_prefix_sibling_with_search_root(tmp_path):
    root = tmp_path / "session"
    sibling = tmp_path / "session_backup"
    root.mkdir()
    sibling.mkdir()
    (root / "known.jpg").write_bytes(b"x")
    (sibling / "outside.jpg").write_bytes(b"x")

    missing = _find_missing_image_paths(
        [], {}, search_roots=[str(root)]
    )

    assert missing == [str(root / "known.jpg")]


def test_extract_features_updates_stale_feature_dump(tmp_path, monkeypatch):
    model = _make_model()
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(50):
        _make_image(folder / f"{i}.jpg", i)

    # Count how many images pass through the model on each infer call
    processed_counts = []
    original_infer = Inferer.infer

    def counting_infer(self, **kwargs):
        processed_counts.append(len(self.data_loader.dataset))
        return original_infer(self, **kwargs)

    monkeypatch.setattr(Inferer, "infer", counting_infer)

    # Initial run: all 50 images get features, feature dump is created
    features = extract_features(model, [str(folder)], device="cpu", batch_size=8)
    assert len(features) == 50
    feat_files = [p for p in folder.iterdir() if p.name.endswith(".feat")]
    assert len(feat_files) == 1

    # Add 4 new images to the folder
    for i in range(50, 54):
        _make_image(folder / f"{i}.jpg", i)

    # Second run: only the 4 new images must be processed, dump is replaced in place
    updated = extract_features(model, [str(folder)], device="cpu", batch_size=8)

    assert processed_counts == [50, 4]
    assert len(updated) == 54
    feat_files = [p for p in folder.iterdir() if p.name.endswith(".feat")]
    assert len(feat_files) == 1  # dump replaced in place, not duplicated
    # Preexisting features are reused, not recomputed
    for path, feature in features.items():
        assert torch.equal(updated[path], feature)

    # Third run: folder unchanged -> nothing is inferred or rewritten
    updated2 = extract_features(model, [str(folder)], device="cpu", batch_size=8)
    assert processed_counts == [50, 4]
    assert set(updated2.keys()) == set(updated.keys())
    for path, feature in updated.items():
        assert torch.equal(updated2[path], feature)
