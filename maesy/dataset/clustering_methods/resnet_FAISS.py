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
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from maesy.dataset import UnlabeledDataset, MultiDataset
from maesy.evaluation.inferer import Inferer
from maesy.model import ResnetFeatureExtractor, BaseModel


def cluster(paths, similarity_threshold=0.85, batch_size=128, forward_scale=128, filetype=".jpg", step=1, start_index=0):
    """
    Select diverse images using sequential similarity-based filtering.
    
    Args:
        :param paths: List of paths to image directories
        :param similarity_threshold: Maximum cosine similarity threshold (0-1).
                            Images with similarity >= threshold to any representative are discarded.
                            Default: 0.85 (keeps only images with <85% similarity)
        :param batch_size: Batch size for neural network inference
        :param forward_scale: Size to resize images to before feature extraction (default: 128)
        :param filetype: File extension to filter for (default: ".jpg")
        :param step: Step size for selecting images from directories (default: 1)
        :param start_index: Starting index for selecting images from directories (default: 0)
    
    Returns:
        List of Path objects pointing to selected representative images
    """
    
    # Setup transforms for feature extraction
    img_transforms = transforms.Compose([
        transforms.Resize(size=(forward_scale, forward_scale)),
        transforms.ToTensor(),
    ])
    
    # Create dataset from all image directories
    multi_dataset = MultiDataset([
        UnlabeledDataset(images_dir=path, transforms=img_transforms, step=step, start_index=start_index)
        for path in paths
    ])
    multi_dataloader = DataLoader(multi_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                  drop_last=False)
    
    # Setup model for feature extraction
    model: BaseModel = ResnetFeatureExtractor("resnet50")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Lists to store representative embeddings and their paths
    representative_embeddings = []
    representative_paths = []
    
    # Process images in batches for efficiency
    print("Extracting features and selecting representatives...")
    inferer = Inferer(model=model, data_loader=multi_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    all_predictions, _ = inferer.infer()
    features = torch.cat(all_predictions)
    features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    features = features.to("cpu").detach().numpy()

    # Process each image in the batch
    for i, feature in enumerate(tqdm(features)):
        if len(representative_embeddings) == 0:
            # First image is always a representative
            representative_embeddings.append(feature)
            representative_paths.append(multi_dataset.get_image_path(i))
        else:
            # Compute cosine similarity to all existing representatives
            # Since features are normalized, cosine similarity is just the dot product
            similarities = np.array([
                np.dot(feature, rep_emb)
                for rep_emb in representative_embeddings
            ])
# TODO: Optimize similarity computation with sklearn.neighbors NearestNeighbors
            max_similarity = similarities.max()

            # Keep image if it's sufficiently different from all representatives
            if max_similarity < similarity_threshold:
                representative_embeddings.append(feature)
                representative_paths.append(multi_dataset.get_image_path(i))
    
    print(f"Selected {len(representative_paths)} representative images out of {features.shape[0]}")
    print(f"Reduction: {100 * (1 - len(representative_paths) / features.shape[0]):.1f}%")
    
    return representative_paths


def cluster_with_faiss(paths, chosen_paths, similarity_threshold=0.85, batch_size=256, forward_scale=224, filetype=".jpg", step=1, start_index=0):
    """
    Sequential similarity-based clustering with FAISS for faster similarity search.
    
    This is an optimized version that uses FAISS library for fast nearest-neighbor search.
    Falls back to the standard implementation if FAISS is not available.
    
    Args:
        :param paths: List of paths to image directories
        :param chosen_paths: List of already chosen image paths to include as initial representatives (pass None if not wanted)
        :param similarity_threshold: Maximum cosine similarity threshold (0-1)
        :param batch_size: Batch size for neural network inference
        :param forward_scale: Size to resize images to before feature extraction
        :param filetype: File extension to filter for (default: ".jpg")
        :param step: Step size for selecting images from directories (default: 1)
        :param start_index: Starting index for selecting images from directories (default: 0)
    
    Returns:
        List of Path objects pointing to selected representative images
    """
    try:
        import faiss
    except ImportError:
        print("FAISS not available, falling back to standard implementation")
        return cluster(paths, similarity_threshold, batch_size, forward_scale, filetype, step, start_index)
    
    # Setup transforms for feature extraction
    img_transforms = transforms.Compose([
        transforms.Resize(size=(forward_scale, forward_scale)),
        transforms.ToTensor(),
    ])

    # Create dataset from all image directories
    multi_dataset = MultiDataset([
        UnlabeledDataset(images_dir=path, transforms=img_transforms, step=step, start_index=start_index)
        for path in paths
    ])
    multi_dataloader = DataLoader(multi_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=False)
    
    # Setup model for feature extraction
    model: BaseModel = ResnetFeatureExtractor("resnet50")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    feature_dim = 2048  # ResNet50 feature dimension after global average pooling
    # Initialize FAISS index for cosine similarity (inner product with normalized vectors)
    index = faiss.IndexFlatIP(feature_dim)

    if chosen_paths is not None:
        print(f"Adding pre-chosen images to FAISS index...")
        # Create dataset from all image directories
        chosen_multi = MultiDataset([
            UnlabeledDataset(images_dir=path, transforms=img_transforms, step=step, start_index=start_index)
            for path in chosen_paths
        ])
        multi_dataloader = DataLoader(chosen_multi, batch_size=batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True,
                                      drop_last=False)
        for batch_tensor in tqdm(multi_dataloader):
            batch_tensor = batch_tensor.to(device)
            # Extract features
            with torch.no_grad():
                features = model(batch_tensor)
                # Normalize features for cosine similarity
                features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
            features_cpu = features.cpu().numpy()
            for feature in features_cpu:
                index.add(feature.reshape(1, -1).astype(np.float32))

    representative_paths = []
    
    # Process images in batches for efficiency
    print("Extracting features and selecting representatives with FAISS...")

    for i, batch_tensor in enumerate(tqdm(multi_dataloader)):
        batch_tensor = batch_tensor.to(device)
        # Extract features
        with torch.no_grad():
            features = model(batch_tensor)
            # Normalize features for cosine similarity
            features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        
        features_cpu = features.cpu().numpy()
        
        # Process each image in the batch
        for j, feature in enumerate(features_cpu):
            if index.ntotal == 0:
                # First image is always a representative
                index.add(feature.reshape(1, -1).astype(np.float32))
                img_path = multi_dataset.get_image_path(i*batch_size+j)
                representative_paths.append(img_path)
            else:
                # Search for nearest neighbor in FAISS index
                feature_query = feature.reshape(1, -1).astype(np.float32)
                similarities, _ = index.search(feature_query, 1)
                max_similarity = similarities[0, 0]
                
                # Keep image if it's sufficiently different from all representatives
                if max_similarity < similarity_threshold:
                    index.add(feature_query)
                    img_path = multi_dataset.get_image_path(i*batch_size+j)
                    representative_paths.append(img_path)
    
    return representative_paths

# if __name__=="__main__":
#     used_paths = cluster_with_faiss([r"/media/simon/42A099D63E90C520/Raw Training Data/DutchSalvador/temp/GP3_DutchNaoTeam_Salvador_2025-08-15-14-57-23_out"])