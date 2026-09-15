from typing import List, Optional


def cluster(model_info: str, dataset_paths: List[str], intermediate_layer: Optional[str] = None, faiss_threshold: float = 0.75):
    """
    Cluster images in a dataset using a specified model.

    Args:
        model_info (str): The path to the model to use for feature extraction.
        dataset_paths (List[str]): The paths to the datasets to cluster.
        intermediate_layer (Optional[str]): The name of the intermediate layer to use for feature extraction (only for ONNX models).
        faiss_threshold (float): The largest acceptable similarity threshold for FAISS clustering. Lower values result in less images.
    """
    from _maesy_core.model.model_tools.model_factory import create_model
    from _maesy_core.inference.inferer import Inferer
    from clusterdevil.feature_extraction import extract_features
    from clusterdevil.clustering_methods.base_clustering import run

    model = create_model(model_info)
    feature_dict = extract_features(model, dataset_paths, intermediate_layer = intermediate_layer)
    selected = run(new_features=feature_dict, clustering_method="FAISS", preexisting_features=None, similarity_threshold=faiss_threshold)

    _copy_selected_images(list(selected.keys()), dataset_paths[0]+"/clustered_output")

def _copy_selected_images(selected_paths: List[str], output_dir: str):
    """
    Copy selected images to the output directory.

    Args:
        selected_paths (List[str]): The paths to the selected images.
        output_dir (str): The path to the output directory.
    """
    import os
    import shutil

    os.makedirs(output_dir, exist_ok=True)
    for path in selected_paths:
        shutil.copy(path, os.path.join(output_dir, os.path.basename(path)))

    print(f"Copied images to {output_dir}")