from typing import Optional, List, Callable
import os

import torch
from torch.utils.data import Dataset

from _maesy_core.model import BaseModel
from clusterdevil.utils import store_features, find_and_load_features


def _list_image_files(folder: str) -> List[str]:
    """List image filenames (sorted) contained directly in the given folder."""
    if not os.path.isdir(folder):
        return []
    # Same extension filter (and case sensitivity) as MaesyDataset's image folder loading
    return sorted(entry.name for entry in os.scandir(folder)
                  if entry.is_file() and entry.name.endswith((".jpg", ".jpeg", ".png")))


def _find_missing_image_paths(feature_files: List[str], stored_features: dict) -> List[str]:
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
    for feature_file in feature_files:
        folder = os.path.dirname(feature_file)
        stored_names = {os.path.basename(path) for path in stored_features if path.startswith(folder)}
        for name in _list_image_files(folder):
            if name not in stored_names:
                missing.append(os.path.join(folder, name))
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
    final_features, loaded_paths = find_and_load_features(paths, model_hash)
    if len(loaded_paths) > 0:
        print(f"Found existing features in the following paths:")
        for loaded_path in loaded_paths:
            print(f" - {loaded_path}")

    model.eval()
    in_dims = model.get_input_dims()
    img_transforms = ClusterTransforms(image_height=in_dims[-2], image_width=in_dims[-1])

    # Extract features for all folders that don't have a feature file yet
    loaded_folders = {os.path.normpath(os.path.dirname(loaded_path)) for loaded_path in loaded_paths}
    new_folders = [path for path in paths if os.path.normpath(path) not in loaded_folders]
    if len(new_folders) > 0:
        print(f"Extracting features for {len(new_folders)} paths...")
        internal_dataset = MultiDataset([MaesyDataset(dataset_dir=path, annotation_type="image_folder", transforms=img_transforms, use_first_n=None) for path in new_folders])
        final_features.update(_infer_features(model, internal_dataset, device, batch_size))

    # Check if the stored features are still up to date (filename-only comparison)
    # and extract features for images that were added to the folder since.
    missing_paths = _find_missing_image_paths(loaded_paths, final_features)
    if len(missing_paths) > 0:
        print(f"Found {len(missing_paths)} image(s) in the folder(s) without stored features. Extracting them...")
        subset_dataset = _ImageFilesDataset(missing_paths, transforms=img_transforms)
        final_features.update(_infer_features(model, subset_dataset, device, batch_size))

    if len(new_folders) > 0 or len(missing_paths) > 0:
        store_features(final_features, model_hash)
    return final_features
