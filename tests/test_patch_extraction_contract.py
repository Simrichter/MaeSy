import importlib.util
import sys
import types
from pathlib import Path

from PIL import Image


def _load_patch_extractor(monkeypatch):
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.__path__ = []
    torch_data = types.ModuleType("torch.utils.data")
    torch_data.DataLoader = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data)

    maesy = types.ModuleType("maesy")
    maesy.__path__ = []
    training = types.ModuleType("maesy.training")
    training.__path__ = []
    collate = types.ModuleType("maesy.training.collate_functions")
    collate.collate_detection_fn = object()
    monkeypatch.setitem(sys.modules, "maesy", maesy)
    monkeypatch.setitem(sys.modules, "maesy.training", training)
    monkeypatch.setitem(sys.modules, "maesy.training.collate_functions", collate)

    core = types.ModuleType("_maesy_core")
    core.__path__ = []
    dataset = types.ModuleType("_maesy_core.dataset")
    dataset.__path__ = []
    dataset.MaesyDataset = type("MaesyDataset", (), {})
    monkeypatch.setitem(sys.modules, "_maesy_core", core)
    monkeypatch.setitem(sys.modules, "_maesy_core.dataset", dataset)

    source = Path(__file__).parents[1] / "_maesy_core" / "dataset" / "extract_patches_from_dataset.py"
    spec = importlib.util.spec_from_file_location("_maesy_core.dataset.extract_patches_from_dataset", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_saving_same_basename_patches_from_distinct_sources_does_not_overwrite(tmp_path, monkeypatch):
    extractor = _load_patch_extractor(monkeypatch)
    first = tmp_path / "camera-a" / "frame.png"
    second = tmp_path / "camera-b" / "frame.png"
    first.parent.mkdir()
    second.parent.mkdir()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(first)
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(second)
    patches = [
        extractor.Patch((0, 0, 1, 1), first),
        extractor.Patch((0, 0, 1, 1), second),
    ]

    extractor._save_patches([patches[0]], str(tmp_path / "patches"), 1)
    extractor._save_patches([patches[1]], str(tmp_path / "patches"), 1)

    outputs = sorted((tmp_path / "patches").glob("*.png"))
    assert len(outputs) == 2
    assert len({path.name for path in outputs}) == 2
    assert {Image.open(path).getpixel((0, 0)) for path in outputs} == {
        (255, 0, 0), (0, 255, 0)
    }
