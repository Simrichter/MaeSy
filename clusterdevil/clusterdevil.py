from typing import List

from clusterdevil.feature_extraction import extract_features


def cluster(model_info: str, dataset_paths: List[str]):
    """
    Cluster images in a dataset using a specified model.

    Args:
        model_info (str): The path to the model to use for feature extraction.
        dataset_paths (List[str]): The paths to the datasets to cluster.
    """
    from _maesy_core.model.model_tools.model_factory import create_model
    from _maesy_core.inference.inferer import Inferer

    model = create_model(model_info)
    extract_features(model, dataset_paths)