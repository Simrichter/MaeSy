from typing import List, Optional


def cluster(model_info: str, dataset_paths: List[str], intermediate_layer: Optional[str] = None):
    """
    Cluster images in a dataset using a specified model.

    Args:
        model_info (str): The path to the model to use for feature extraction.
        dataset_paths (List[str]): The paths to the datasets to cluster.
        intermediate_layer (Optional[str]): The name of the intermediate layer to use for feature extraction (only for ONNX models).
    """
    from _maesy_core.model.model_tools.model_factory import create_model
    from _maesy_core.inference.inferer import Inferer
    from clusterdevil.feature_extraction import extract_features
    from clusterdevil.clustering_methods.base_clustering import run

    model = create_model(model_info)
    feature_dict = extract_features(model, dataset_paths, intermediate_layer = intermediate_layer)
    run(new_features=feature_dict, clustering_method="FAISS", preexisting_features=None, similarity_threshold=0.25)