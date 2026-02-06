# Sequential Similarity Clustering

This document describes the `sequential_similarity` clustering method for selecting diverse training images.

## Overview

The sequential similarity clustering method processes images in temporal order and selects only those that are sufficiently dissimilar from already selected representatives. This approach is useful for:

- Reducing dataset size while maintaining diversity
- Removing near-duplicate images
- Creating representative training sets from large image collections

## Algorithm

1. **Process in temporal order**: Images are sorted by file modification time
2. **Maintain representatives**: Keep a set of selected image embeddings
3. **Compare new images**: For each new image:
   - Extract its feature embedding using ResNet50
   - Compute cosine similarity to all existing representatives
   - If max similarity ≥ threshold: discard (too similar)
   - Else: keep as a new representative

## Usage

### Command Line

```bash
python -m maesy dataset \
    --data-paths /path/to/images \
    --dataset-name my_dataset \
    --cluster-method sequential_similarity
```

### Python API

```python
from maesy.dataset.clustering_methods.sequential_similarity import cluster

# Select diverse images with default threshold (0.85)
selected_paths = cluster(
    paths=['/path/to/images'],
    similarity_threshold=0.85,  # Keep images with <85% similarity
    batch_size=64,              # Batch size for inference
    forward_scale=128           # Image resize for feature extraction
)

print(f"Selected {len(selected_paths)} images")
```

### With FAISS (for large datasets)

For faster processing with large datasets, use the FAISS-accelerated version:

```python
from maesy.dataset.clustering_methods.sequential_similarity import cluster_with_faiss

selected_paths = cluster_with_faiss(
    paths=['/path/to/images'],
    similarity_threshold=0.85,
    batch_size=64,
    forward_scale=128
)
```

**Note**: FAISS is optional. If not available, it automatically falls back to the standard implementation.

## Parameters

- **similarity_threshold** (float, default=0.85): Maximum cosine similarity threshold (0-1)
  - Lower values (e.g., 0.7): More strict, keeps only very different images
  - Higher values (e.g., 0.95): More permissive, keeps more images
  
- **batch_size** (int, default=64): Batch size for neural network inference
  - Larger values use more memory but are faster
  
- **forward_scale** (int, default=128): Size to resize images to before feature extraction
  - Smaller values are faster but less accurate
  - Larger values are slower but more accurate

## Comparison with resnet_kmeans

| Feature | sequential_similarity | resnet_kmeans |
|---------|----------------------|---------------|
| **Selection strategy** | Sequential, time-based | Cluster-based |
| **Parameters** | Similarity threshold | Number of clusters |
| **Temporal order** | Preserves time order | Ignores time order |
| **Memory** | Lower (incremental) | Higher (batch) |
| **Speed** | Slower (sequential) | Faster (parallel) |
| **Use case** | Time-series data, removing near-duplicates | Balanced representative sampling |

## Tips

1. **Choosing the threshold**:
   - Start with 0.85 (default)
   - For more diversity: lower to 0.75-0.80
   - For near-duplicate removal only: raise to 0.90-0.95

2. **Performance**:
   - Use larger batch sizes on GPU for faster processing
   - Use FAISS version for datasets >10,000 images
   - Reduce forward_scale to 64 for faster (but less accurate) processing

3. **Results**:
   - Check the reduction percentage in the output
   - If too many images are kept, lower the threshold
   - If too few images are kept, raise the threshold

## Example Output

```
Processing 1000 images in time order...
Extracting features and selecting representatives...
100%|████████████████████████████████| 16/16 [00:42<00:00,  2.67s/it]
Selected 234 representative images out of 1000
Reduction: 76.6%
```
