"""
Sequential Similarity-based Clustering

This module implements a sequential clustering method that processes images in time order
and selects only those that are sufficiently dissimilar from already selected representatives.

The algorithm:
1. Process images in time order (sorted by file modification time)
2. Maintain a set of representative embeddings
3. For each new image:
   - Extract its feature embedding using a neural network
   - Compare it to existing representatives using cosine similarity
   - If max similarity >= threshold: discard
   - Else: keep as a new representative

This approach is useful for selecting diverse training samples from a large image dataset.
"""
from typing import Dict, Optional

import torch
import numpy as np

def _get_feature_norm(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute the norm of each feature vector in the dictionary.
    Args:
        features: A dictionary containing path:feature vector pairs
    Returns:
        A tensor containing the norms of the feature vectors
    """
    feature_norm = torch.stack(list(features.values())).norm(dim=0) #.norm(dim=1, keepdim=True) + 1e-8)
    return feature_norm

def _no_faiss_fallback(new_features: Dict[str, torch.Tensor], preexisting_features: Optional[Dict[str, torch.Tensor]], similarity_threshold: float, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Fallback implementation if FAISS library is not found
    """

    # Lists to store representative embeddings and their paths
    representative_embeddings = []
    representative_paths = []

    if preexisting_features:
        feature_norm = _get_feature_norm(preexisting_features)
        for path, feature in preexisting_features.items():
            feature = (feature/feature_norm).flatten()
            representative_embeddings.append(feature)
            representative_paths.append(path)

    feature_norm = _get_feature_norm(new_features)

    for path, feature in new_features.items():
        normed_feature = (feature / feature_norm).flatten()
        if len(representative_embeddings) == 0:
            # First image is always a representative
            representative_embeddings.append(normed_feature)
            representative_paths.append(path)
        else:
            # Compute cosine similarity to all existing representatives
            # Since features are normalized, cosine similarity is just the dot product
            similarities = np.array([np.dot(normed_feature, rep_emb) for rep_emb in representative_embeddings])
            # TODO: Optimize similarity computation with sklearn.neighbors NearestNeighbors ?
            max_similarity = similarities.max()

            # Keep image if it's sufficiently different from all representatives
            if max_similarity < similarity_threshold:
                representative_embeddings.append(normed_feature)
                representative_paths.append(path)

    print(f"Selected {len(representative_paths)} representative images out of {len(new_features)}")
    print(f"Reduction: {100 * (1 - len(representative_paths) / len(new_features)):.1f}%")

    # retrieve unnormalized feature vectors for the selected representative paths
    return dict(zip(representative_paths, [new_features[path] for path in representative_paths]))

def cluster(new_features: Dict[str, torch.Tensor], similarity_threshold: float, preexisting_features: Optional[Dict[str, torch.Tensor]]=None, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Sequential similarity-based clustering with FAISS for faster similarity search.

    This is an optimized version that uses FAISS library for fast nearest-neighbor search.
    Falls back to the standard implementation if FAISS is not available.

    Args:
        :param new_features: A dictionary containing path:feature vector pairs to cluster
        :param similarity_threshold: Largest acceptable cosine similarity between selected images (range: [0-1] )
        :param preexisting_features: A dictionary containing path:feature vector pairs that are already selected and should be used as reference

    Returns:
        A dictionary containing the path:feature vector pairs of the selected representative images.
    """

    assert 0 <= similarity_threshold <= 1, "Error: similarity_threshold must lie in the range [0, 1]!"

    try:
        import faiss
    except ImportError:
        print("FAISS library not available, falling back to standard implementation.\nInstallation of FAISS is recommended to cluster much faster!!")
        return _no_faiss_fallback(new_features, preexisting_features, similarity_threshold)

    feature_dim = new_features[next(iter(new_features))].flatten().shape[0]  # Assuming all new_features have the same dimension
    print("feature_dim: ",feature_dim)
    # Initialize FAISS index for cosine similarity (inner product with normalized vectors)
    index = faiss.IndexFlatIP(feature_dim)

    # Add preexisting features to the FAISS index if provided
    if preexisting_features is not None:
        assert feature_dim == preexisting_features[next(iter(preexisting_features))].flatten().shape[0], "Feature dimensions of new and preexisting features must match!"
        # features_cpu = features.cpu().numpy()
        feature_norm = _get_feature_norm(preexisting_features)
        for _, feature in preexisting_features.items():
            # Normalize features for cosine similarity
            feature = feature / feature_norm
            feature = feature.flatten()#.unsqueeze(0)  # Ensure feature is a 1D vector
            index.add(feature)

    selected = {}
    feature_norm = _get_feature_norm(new_features)
    # Process each image in the batch
    for img_path, feature in new_features.items():
        # print(f"processing path {img_path}")
        # print("Feature: ", feature.shape)
        feature = (feature/feature.norm()).flatten().unsqueeze(0)  # Ensure feature is a normed 1D vector
        if index.ntotal == 0:
            # First image is always a representative
            index.add(feature)
            selected[img_path] = feature # Store unnormalized features to preserve consistency
        else:
            # Search for nearest neighbor in FAISS index
            # print(feature.shape)
            similarities, _ = index.search(feature, 1)
            max_similarity = similarities[0, 0]
            # print("max sim: ", max_similarity) # TODO: Feature to collect similarities for all images and plot histogram to visualize distribution of similarities
            # Keep image if it's sufficiently different from all representatives
            if max_similarity < similarity_threshold:
                index.add(feature)
                selected[img_path] = feature # Store unnormalized features to preserve consistency

    print(f"Selected {len(selected)} representative images out of {len(new_features)}")
    print(f"Reduction: {100 * (1 - len(selected) / len(new_features)):.1f}%")

    return selected