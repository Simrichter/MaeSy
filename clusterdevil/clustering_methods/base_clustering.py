from abc import ABC, abstractmethod
from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader
from _maesy_core.dataset import MultiDataset, MaesyDataset
from _maesy_core.dataset.augmentations import ClusterTransforms
from _maesy_core.model import BaseModel
import importlib

def get_available_clustering_methods() -> list[str]:
    """
    Get a list of available clustering methods from the clustering_methods submodule.
    Returns:
        A list of available clustering method names.
    """
    import clusterdevil.clustering_methods
    available_methods = [method for method in dir(clusterdevil.clustering_methods) if not method.startswith("_")]
    return available_methods

def run(new_features: Dict[str, torch.Tensor], clustering_method: str, preexisting_features: Optional[Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
    """
        Execute clustering on feature vectors
        Args:
            :param new_features: A dictionary containing path:feature vector pairs to cluster
            :param clustering_method: The name of the clustering method. This should match the name of a class that inherits from BaseClustering
            :param preexisting_features: A dictionary containing path:feature vector pairs that are already selected and should be used as reference

        Returns:
            A dictionary containing the clustered results.
    """
    print("=" * 60)
    print("Starting clustering process...")

    available_methods = get_available_clustering_methods()
    # print("available methods: ", available_methods)
    if clustering_method not in available_methods:
        raise ValueError(f"Clustering method '{clustering_method}' is not available. Available methods: {available_methods}")
    print(f"Selected clustering method: {clustering_method}")
    print(f"Number of feature vectors: {len(new_features)}")
    if preexisting_features:
        print(f"Found {len(preexisting_features)} preexisting feature vectors to use as reference.")

    # import the function passed in clustering_method
    sel_method = getattr(importlib.import_module("clusterdevil.clustering_methods." + clustering_method), "cluster")
    return sel_method(new_features=new_features, preexisting_features=preexisting_features, **kwargs)

