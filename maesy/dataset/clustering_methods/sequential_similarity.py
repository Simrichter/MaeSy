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
from pathlib import Path
from tqdm import tqdm
from maesy.dataset import UnlabeledDataset, MultiDataset
from maesy.evaluation.inferer import Inferer
from maesy.model import ResnetFeatureExtractor, BaseModel


def cluster(paths, similarity_threshold=0.85, batch_size=64, forward_scale=128, filetype=".jpg"):
    """
    Select diverse images using sequential similarity-based filtering.
    
    Args:
        paths: List of paths to image directories
        similarity_threshold: Maximum cosine similarity threshold (0-1). 
                            Images with similarity >= threshold to any representative are discarded.
                            Default: 0.85 (keeps only images with <85% similarity)
        batch_size: Batch size for neural network inference
        forward_scale: Size to resize images to before feature extraction (default: 128)
        filetype: File extension to filter for (default: ".jpg")
    
    Returns:
        List of Path objects pointing to selected representative images
    """
    
    # Setup transforms for feature extraction
    transfs = transforms.Compose([
        transforms.Resize(size=(forward_scale, forward_scale)),
        transforms.ToTensor(),
    ])
    
    # Create dataset from all image directories
    multi_dataset = MultiDataset([
        UnlabeledDataset(images_dir=path, transforms=transfs, filetype=filetype) 
        for path in paths
    ])
    
    # Get all image paths and sort by modification time
    all_image_paths = [multi_dataset.get_image_path(i) for i in range(len(multi_dataset))]
    
    # Sort images by modification time to process in temporal order
    sorted_indices = sorted(
        range(len(all_image_paths)),
        key=lambda i: Path(all_image_paths[i]).stat().st_mtime
    )
    
    print(f"Processing {len(all_image_paths)} images in time order...")
    
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
    for batch_start in tqdm(range(0, len(sorted_indices), batch_size)):
        batch_indices = sorted_indices[batch_start:batch_start + batch_size]
        
        # Load and process batch
        batch_images = []
        batch_paths = []
        for idx in batch_indices:
            img = multi_dataset[idx]
            batch_images.append(img)
            batch_paths.append(all_image_paths[idx])
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(batch_tensor)
            # Normalize features for cosine similarity
            features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        
        features_cpu = features.cpu().numpy()
        
        # Process each image in the batch
        for i, (feature, img_path) in enumerate(zip(features_cpu, batch_paths)):
            if len(representative_embeddings) == 0:
                # First image is always a representative
                representative_embeddings.append(feature)
                representative_paths.append(img_path)
            else:
                # Compute cosine similarity to all existing representatives
                # Since features are normalized, cosine similarity is just the dot product
                similarities = np.array([
                    np.dot(feature, rep_emb) 
                    for rep_emb in representative_embeddings
                ])
                
                max_similarity = similarities.max()
                
                # Keep image if it's sufficiently different from all representatives
                if max_similarity < similarity_threshold:
                    representative_embeddings.append(feature)
                    representative_paths.append(img_path)
    
    print(f"Selected {len(representative_paths)} representative images out of {len(all_image_paths)}")
    print(f"Reduction: {100 * (1 - len(representative_paths) / len(all_image_paths)):.1f}%")
    
    return representative_paths


def cluster_with_faiss(paths, similarity_threshold=0.85, batch_size=64, forward_scale=128, filetype=".jpg"):
    """
    Sequential similarity-based clustering with FAISS for faster similarity search.
    
    This is an optimized version that uses FAISS library for fast nearest-neighbor search.
    Falls back to the standard implementation if FAISS is not available.
    
    Args:
        paths: List of paths to image directories
        similarity_threshold: Maximum cosine similarity threshold (0-1)
        batch_size: Batch size for neural network inference
        forward_scale: Size to resize images to before feature extraction
        filetype: File extension to filter for (default: ".jpg")
    
    Returns:
        List of Path objects pointing to selected representative images
    """
    try:
        import faiss
    except ImportError:
        print("FAISS not available, falling back to standard implementation")
        return cluster(paths, similarity_threshold, batch_size, forward_scale, filetype)
    
    # Setup transforms for feature extraction
    transfs = transforms.Compose([
        transforms.Resize(size=(forward_scale, forward_scale)),
        transforms.ToTensor(),
    ])
    
    # Create dataset from all image directories
    multi_dataset = MultiDataset([
        UnlabeledDataset(images_dir=path, transforms=transfs, filetype=filetype) 
        for path in paths
    ])
    
    # Get all image paths and sort by modification time
    all_image_paths = [multi_dataset.get_image_path(i) for i in range(len(multi_dataset))]
    
    # Sort images by modification time to process in temporal order
    sorted_indices = sorted(
        range(len(all_image_paths)),
        key=lambda i: Path(all_image_paths[i]).stat().st_mtime
    )
    
    print(f"Processing {len(all_image_paths)} images in time order with FAISS...")
    
    # Setup model for feature extraction
    model: BaseModel = ResnetFeatureExtractor("resnet50")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get embedding dimension from model
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, forward_scale, forward_scale).to(device)
        dummy_output = model(dummy_input)
        embedding_dim = dummy_output.shape[1]
    
    # Initialize FAISS index for cosine similarity (inner product with normalized vectors)
    index = faiss.IndexFlatIP(embedding_dim)
    
    representative_paths = []
    
    # Process images in batches for efficiency
    print("Extracting features and selecting representatives with FAISS...")
    for batch_start in tqdm(range(0, len(sorted_indices), batch_size)):
        batch_indices = sorted_indices[batch_start:batch_start + batch_size]
        
        # Load and process batch
        batch_images = []
        batch_paths = []
        for idx in batch_indices:
            img = multi_dataset[idx]
            batch_images.append(img)
            batch_paths.append(all_image_paths[idx])
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(batch_tensor)
            # Normalize features for cosine similarity
            features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        
        features_cpu = features.cpu().numpy()
        
        # Process each image in the batch
        for i, (feature, img_path) in enumerate(zip(features_cpu, batch_paths)):
            if index.ntotal == 0:
                # First image is always a representative
                index.add(feature.reshape(1, -1).astype(np.float32))
                representative_paths.append(img_path)
            else:
                # Search for nearest neighbor in FAISS index
                feature_query = feature.reshape(1, -1).astype(np.float32)
                similarities, _ = index.search(feature_query, 1)
                max_similarity = similarities[0, 0]
                
                # Keep image if it's sufficiently different from all representatives
                if max_similarity < similarity_threshold:
                    index.add(feature_query)
                    representative_paths.append(img_path)
    
    print(f"Selected {len(representative_paths)} representative images out of {len(all_image_paths)}")
    print(f"Reduction: {100 * (1 - len(representative_paths) / len(all_image_paths)):.1f}%")
    
    return representative_paths
