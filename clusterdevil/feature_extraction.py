from typing import Optional, List, Callable
import os

import torch
from torch.utils.data import Dataset

from _maesy_core.model import BaseModel
from clusterdevil.utils import store_features, find_and_load_features


def _list_image_files(folder: str) -> List[str]:
    """List image paths directly inside one feature-cache folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, entry.name)
        for entry in os.scandir(folder)
        if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _current_image_paths(search_roots: List[str]) -> set[str]:
    image_paths = set()
    for root in search_roots:
        for parent, _directories, files in os.walk(root):
            image_paths.update(
                os.path.normpath(os.path.abspath(os.path.join(parent, filename)))
                for filename in files
                if filename.lower().endswith((".jpg", ".jpeg", ".png"))
            )
    return image_paths


def _find_missing_image_paths(
    feature_files: List[str],
    stored_features: dict,
    search_roots: Optional[List[str]] = None,
    model_hash: Optional[str] = None,
) -> List[str]:
    """
    Check whether the image folders are still in sync with the stored features.
    For every feature file, compares the image filenames stored in it with the
    image files that currently exist in the folder the feature file belongs to.
    This is a filename-only check (no feature vectors are recomputed here).

    :param feature_files: List of paths to the loaded feature files.
    :param stored_features: The features loaded from those files ({image_path: feature} pairs).
    :return: List of image paths that exist in their folder but have no stored feature yet.
    """
    missing = []
    checked_images = set()
    folders = {
        os.path.normpath(os.path.abspath(os.path.dirname(feature_file)))
        for feature_file in feature_files
    }
    normalized_roots = [
        os.path.normpath(os.path.abspath(root)) for root in (search_roots or [])
    ]
    current_paths = _current_image_paths(normalized_roots) if normalized_roots else None
    if normalized_roots:
        for root in normalized_roots:
            folders.update(
                os.path.normpath(os.path.abspath(parent))
                for parent, _directories, files in os.walk(root)
                if any(filename.lower().endswith((".jpg", ".jpeg", ".png")) for filename in files)
            )
            if model_hash:
                folders.update(
                    os.path.normpath(os.path.abspath(parent))
                    for parent, _directories, files in os.walk(root)
                    if any(filename == f"features_{model_hash}.feat" for filename in files)
                )
            for path in stored_features:
                normalized_path = os.path.normpath(os.path.abspath(path))
                if _is_within(normalized_path, root):
                    folders.add(os.path.dirname(normalized_path))
    folders = sorted(folders)
    stored_paths = {
        os.path.normpath(os.path.abspath(path)) for path in stored_features
    }
    for folder in folders:
        if normalized_roots and not any(_is_within(folder, root) for root in normalized_roots):
            continue
        for image_file in _list_image_files(folder):
            image_path = os.path.normpath(os.path.abspath(image_file))
            if current_paths is not None and image_path not in current_paths:
                continue
            if image_path not in checked_images and image_path not in stored_paths:
                missing.append(image_path)
            checked_images.add(image_path)
    return sorted(set(missing))


class _ImageFilesDataset(Dataset):
    """Minimal dataset that loads a fixed list of image file paths (no labels)."""

    def __init__(self, paths: List[str], transforms: Optional[Callable] = None):
        self.paths = [str(path) for path in paths]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image
        import torchvision
        with Image.open(self.paths[idx]).convert("RGB") as image:
            image = torchvision.tv_tensors.Image(image) / 255.0
            if self.transforms is not None:
                image = self.transforms(image)
        return image

    def get_image_path(self, idx: int) -> str:
        return self.paths[idx]


def _infer_features(model: BaseModel, dataset: Dataset, device: Optional[str], batch_size: int) -> dict:
    """
    Run the model over the given dataset and return {image_path: feature} for all samples.
    """
    from torch.utils.data import DataLoader
    from _maesy_core.inference.inferer import Inferer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory="cuda" in str(device), drop_last=False, in_order=True)
    inferer = Inferer(model, dataloader, device=device)
    preds, _ = inferer.infer()
    return {str(dataset.get_image_path(i)): feature
            for i, feature in zip(range(len(dataset)),
                                  torch.cat([p[list(model.get_output_dims().keys())[0]] for p in preds], dim=0))}


def extract_features(model: BaseModel, paths: List[str], device: Optional[str] = None, batch_size=1, intermediate_layer: Optional[str] = None):
    """
    Extract features from a model given a dataloader and device.
    If features already exist on disk, they are loaded. Additionally, the stored
    feature filenames are checked against the image files in the folder: if new
    images have been added, features are extracted for those images only and the
    feature file is updated in place.
    :param model: The model to extract features from
    :param paths: List of paths to the datasets to extract features from
    :param device: The device to run the model on (e.g., 'cuda' or 'cpu')
    :param batch_size: The batch size for the dataloader
    :param intermediate_layer: The name of the intermediate layer to use for feature extraction (only for ONNX models)
    :return: A list of extracted features
    """
    from _maesy_core.dataset import MaesyDataset, MultiDataset
    from _maesy_core.dataset.augmentations import ClusterTransforms

    if not paths or len(paths) == 0:
        return {}

    # Check if features already exist for the given paths and model hash
    model_hash = model.get_model_hash()
    explicit_feature_files = [
        os.path.realpath(os.fspath(path))
        for path in paths
        if os.path.isfile(os.fspath(path))
        and os.fspath(path).lower().endswith(".feat")
    ]
    mismatched_feature_files = [
        path for path in explicit_feature_files
        if os.path.basename(path) != f"features_{model_hash}.feat"
    ]
    if mismatched_feature_files:
        raise ValueError(
            "Feature cache model hash does not match the requested model: "
            + ", ".join(mismatched_feature_files)
        )

    cache_search_paths = [
        os.path.realpath(os.fspath(path))
        if os.path.isfile(os.fspath(path)) and os.fspath(path).lower().endswith(".feat")
        else os.fspath(path)
        for path in paths
    ]
    final_features, loaded_paths = find_and_load_features(cache_search_paths, model_hash)
    if explicit_feature_files:
        loaded_set = {os.path.realpath(path) for path in loaded_paths}
        missing_caches = [path for path in explicit_feature_files if path not in loaded_set]
        if missing_caches:
            raise ValueError(f"Could not load feature cache(s): {missing_caches}")
    if len(loaded_paths) > 0:
        print(f"Found existing features in the following paths:")
        for loaded_path in loaded_paths:
            print(f" - {loaded_path}")

    model.eval()
    in_dims = model.get_input_dims()
    img_transforms = ClusterTransforms(image_height=in_dims[-2], image_width=in_dims[-1])

    # Extract features for all folders that don't have a feature file yet
    loaded_folders = {
        os.path.normcase(os.path.realpath(os.path.dirname(loaded_path)))
        for loaded_path in loaded_paths
    }
    input_folders = []
    for path in paths:
        resolved_path = os.path.realpath(os.fspath(path))
        if os.path.isfile(resolved_path) and resolved_path.lower().endswith(".feat"):
            if os.path.basename(resolved_path) != f"features_{model_hash}.feat":
                continue
            resolved_path = os.path.dirname(resolved_path)
        if os.path.isdir(resolved_path):
            normalized = os.path.normcase(resolved_path)
            if normalized not in input_folders:
                input_folders.append(normalized)
    for path in input_folders:
        for feature_file in loaded_paths:
            folder = os.path.normcase(os.path.realpath(os.path.dirname(feature_file)))
            if _is_within(folder, path):
                loaded_folders.add(path)
    new_folders = [path for path in input_folders if path not in loaded_folders]
    if len(new_folders) > 0:
        print(f"Extracting features for {len(new_folders)} paths...")
        datasets = [
            MaesyDataset(
                dataset_dir=path, annotation_type="image_folder",
                transforms=img_transforms, use_first_n=None,
            )
            for path in new_folders
        ]
        internal_dataset = MultiDataset(datasets)
        final_features.update(_infer_features(model, internal_dataset, device, batch_size))

    # Check if the stored features are still up to date (filename-only comparison)
    # and extract features for images that were added to the folder since.
    missing_paths = _find_missing_image_paths(
        loaded_paths, final_features, input_folders, model_hash
    )
    if len(missing_paths) > 0:
        print(f"Found {len(missing_paths)} image(s) in the folder(s) without stored features. Extracting them...")
        subset_dataset = _ImageFilesDataset(missing_paths, transforms=img_transforms)
        final_features.update(_infer_features(model, subset_dataset, device, batch_size))

    stale_paths = set()
    if input_folders:
        current_paths = _current_image_paths(input_folders)
        stale_paths = {
            path for path in final_features
            if not any(
                _is_within(os.path.normpath(os.path.abspath(path)), root)
                for root in input_folders
            )
            or os.path.normpath(os.path.abspath(path)) not in current_paths
        }
        for path in stale_paths:
            final_features.pop(path, None)

    if len(new_folders) > 0 or len(missing_paths) > 0 or stale_paths:
        store_features(final_features, model_hash)
        populated_folders = {
            os.path.normpath(os.path.abspath(os.path.dirname(path)))
            for path in final_features
        }
        for feature_file in loaded_paths:
            cache_folder = os.path.normpath(
                os.path.abspath(os.path.dirname(feature_file))
            )
            if (
                cache_folder not in populated_folders
                and any(_is_within(cache_folder, root) for root in input_folders)
            ):
                os.remove(feature_file)
    return final_features
