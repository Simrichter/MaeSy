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

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from maesy.dataset import UnlabeledDataset, MultiDataset, MaesyDataset
from maesy.dataset.clustering_methods.base_clustering import BaseClustering
from maesy.evaluation.inferer import Inferer
from maesy.model import ResnetFeatureExtractor, BaseModel


class ResnetFaiss(BaseClustering):

    def _create_model(self, forward_scale):
        # Setup model for feature extraction
        model: BaseModel = ResnetFeatureExtractor("resnet50", img_size=forward_scale, out_layers=["c6"])
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def _cluster_without_FAISS(self, multi_dataloader, model, similarity_threshold):
        """
        Fallback implementation if FAISS is not found
        """

        # Lists to store representative embeddings and their paths
        representative_embeddings = []
        representative_paths = []

        # Process images in batches for efficiency
        print("Extracting features and selecting representatives...")
        inferer = Inferer(model=model, data_loader=multi_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        all_predictions, _ = inferer.infer()
        features = torch.cat(all_predictions)  # TODO: Check if this still works after i changed [].append to [].extend in infer()!
        features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        features = features.to("cpu").detach().numpy()

        # Process each image in the batch
        for i, feature in enumerate(tqdm(features)):
            if len(representative_embeddings) == 0:
                # First image is always a representative
                representative_embeddings.append(feature)
                representative_paths.append(multi_dataloader.dataset.get_image_path(i))
            else:
                # Compute cosine similarity to all existing representatives
                # Since features are normalized, cosine similarity is just the dot product
                similarities = np.array([np.dot(feature, rep_emb) for rep_emb in representative_embeddings])
                # TODO: Optimize similarity computation with sklearn.neighbors NearestNeighbors ?
                max_similarity = similarities.max()

                # Keep image if it's sufficiently different from all representatives
                if max_similarity < similarity_threshold:
                    representative_embeddings.append(feature)
                    representative_paths.append(multi_dataloader.dataset.get_image_path(i))

        print(f"Selected {len(representative_paths)} representative images out of {features.shape[0]}")
        print(f"Reduction: {100 * (1 - len(representative_paths) / features.shape[0]):.1f}%")

        return representative_paths

    def _cluster(self, new_dataloader, existing_dataloader, model, **kwargs):
        """
        Sequential similarity-based clustering with FAISS for faster similarity search.

        This is an optimized version that uses FAISS library for fast nearest-neighbor search.
        Falls back to the standard implementation if FAISS is not available.

        Args:
            :param similarity_threshold: Maximum cosine similarity threshold (0-1)

        Returns:
            List of Path objects pointing to selected representative images
        """

        similarity_threshold = kwargs.get("similarity_threshold", -1)
        assert similarity_threshold != -1, "Error: similarity_threshold must be provided as a keyword argument when using FAISS clustering"

        try:
            import faiss
        except ImportError:
            print("FAISS not available, falling back to standard implementation.\nInstallation of FAISS is heavily recommended due to much better performance!")
            return self._cluster_without_FAISS(new_dataloader, model, similarity_threshold)

        feature_dim = 2048  # ResNet50 feature dimension after global average pooling
        # Initialize FAISS index for cosine similarity (inner product with normalized vectors)
        index = faiss.IndexFlatIP(feature_dim)

        if existing_dataloader is not None:
            for batch_tensor, _ in tqdm(existing_dataloader, desc="Loading existing data"):
                batch_tensor = batch_tensor.to(self.device)
                # Extract features
                with torch.no_grad():
                    features = model(batch_tensor)["c6"]
                    # Normalize features for cosine similarity
                    features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
                features_cpu = features.cpu().numpy()
                for feature in features_cpu:
                    index.add(feature.reshape(1, -1).astype(np.float32))

        representative_paths = []

        # Process images in batches for efficiency
        print("Extracting features and selecting representatives with FAISS...")

        for i, (batch_tensor, _) in enumerate(tqdm(new_dataloader, desc="Clustering...")):
            batch_tensor = batch_tensor.to(self.device)
            # Extract features
            with torch.no_grad():
                features = model(batch_tensor)["c6"]
                # Normalize features for cosine similarity
                features = features / (features.norm(dim=1, keepdim=True) + 1e-8)

            features_cpu = features.cpu().numpy()

            # Process each image in the batch
            for j, feature in enumerate(features_cpu):
                if index.ntotal == 0:
                    # First image is always a representative
                    index.add(feature.reshape(1, -1).astype(np.float32))
                    img_path = new_dataloader.dataset.get_image_path(i * batch_tensor.shape[0] + j)
                    representative_paths.append(img_path)
                else:
                    # Search for nearest neighbor in FAISS index
                    feature_query = feature.reshape(1, -1).astype(np.float32)
                    similarities, _ = index.search(feature_query, 1)
                    max_similarity = similarities[0, 0]

                    # Keep image if it's sufficiently different from all representatives
                    if max_similarity < similarity_threshold:
                        index.add(feature_query)
                        img_path = new_dataloader.dataset.get_image_path(i * batch_tensor.shape[0] + j)
                        representative_paths.append(img_path)

        return representative_paths

if __name__ == "__main__":
    data_path = "/media/simon/42A099D63E90C520/K1 Logs/k1_log_20260602_222938_(Garage_1)/k1_log_20260602_222938_0/boostercamera/head/raw/right/rgb"
    ResnetFaiss().run([data_path], similarity_threshold=0.85, batch_size=4)