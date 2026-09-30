import torch
from PIL import Image

from _maesy_core.inference.inferer import Inferer
from _maesy_core.model import ResnetFeatureExtractor
from _maesy_core.model.resnet_featureextractor import ResnetFeatureExtractorConfig
from clusterdevil.feature_extraction import _find_missing_image_paths, extract_features


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
    (subfolder / "c.jpg").write_bytes(b"x")  # nested folders must be ignored

    stored = {str(folder / "a.jpg"): torch.zeros(4)}
    missing = _find_missing_image_paths([str(folder / "features_abc.feat")], stored)

    assert missing == [str(folder / "b.png")]


def test_find_missing_image_paths_empty_when_up_to_date(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"x")

    stored = {str(folder / "a.jpg"): torch.zeros(4)}
    missing = _find_missing_image_paths([str(folder / "features_abc.feat")], stored)

    assert missing == []


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
