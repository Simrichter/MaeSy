from typing import Dict, List
import torch

def store_features(features: Dict[str, torch.Tensor], model_hash: str):
    """
    Store extracted features to a file.

    Args:
        features (dict): A dictionary containing the extracted features. Expects {path: feature_tensor} pairs. For every parent folder, a new save file is created
        model_hash (str): A unique hash representing the model configuration and parameters.
    """
    print("Storing features to disk...")
    import os
    import torch
    distinct_paths = {
        os.path.normpath(os.path.dirname(os.path.abspath(path)))
        for path in features
    }
    print(f"Found {len(distinct_paths)} distinct paths.")
    for path in sorted(distinct_paths):
        selected_features = {
            image_path: feature
            for image_path, feature in features.items()
            if os.path.normpath(os.path.dirname(os.path.abspath(image_path))) == path
        }
        save_path = os.path.join(path, f"features_{model_hash}.feat")
        print(f"Storing {len(selected_features)} features at: '{save_path}'")
        torch.save(selected_features, save_path)
    print("Success")

def find_and_load_features (paths: List[str], model_hash: str) -> tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Find and load extracted features from given paths if they exist.
    If none are found, an empty dictionary and an empty list are returned.

    Args:
        :param paths: A list of paths that contain feature files
        :param model_hash: A unique hash representing the model configuration and parameters.
    Returns:
        A tuple containing the extracted features and the paths of the loaded feature files. The features dictionary expects {path: feature_tensor} pairs.
    """
    paths = _find_feature_files(paths, model_hash)
    import torch
    features = {}
    for path in paths:
        # print(f"Loading features from {path}...")
        loaded_features = torch.load(path)
        if not isinstance(loaded_features, dict):
            raise ValueError(f"Expected a dictionary of features in file {path}, but got {type(loaded_features).__name__}")
        features.update(loaded_features)
    # print(f"Success")
    return features, paths

def _find_feature_files(paths: List[str], model_hash: str) -> List[str]:
    """
    Find feature files matching the model hash in the given directories.
    If paths to feature files are provided, they will be accepted, if the model hash matches

    Args:
        :param paths: A list of paths to search for feature files.
        :param model_hash: A unique hash representing the model configuration and parameters.
    Returns:
        A list of paths to the found feature files.
    """
    import os
    feature_files = []
    expected_name = f"features_{model_hash}.feat"
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path '{path}' does not exist. Skipping.")
            continue
        if os.path.isdir(path):
            for root, _directories, files in os.walk(path):
                if expected_name in files:
                    feature_files.append(os.path.join(root, expected_name))
        elif os.path.isfile(path) and path.lower().endswith(".feat"):
            if os.path.basename(path) != expected_name:
                raise ValueError(
                    f"Feature cache model hash does not match requested model: {path}"
                )
            feature_files.append(path)
    return sorted(set(feature_files))