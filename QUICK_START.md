# MaeSy Quick Reference Guide

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Basic Usage

### 1. Dataset Preparation

```python
from maesy.dataset import DatasetManager, ObjectDetectionDataset, get_train_transforms
from torch.utils.data import DataLoader
from maesy.dataset.transforms import collate_fn

# Initialize dataset manager
dm = DatasetManager(data_root="./data")

# Create dataset
dataset = ObjectDetectionDataset(
    images_dir="./data/train/images",
    annotations_file="./data/train/annotations.json",
    transforms=get_train_transforms(image_size=224)
)

# Create data loader
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
```

### 2. Model Creation

```python
from maesy.model import VisionTransformerDetector, ModelConfig

config = ModelConfig(
    image_size=224,
    patch_size=16,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    num_classes=80,
    num_queries=100
)

model = VisionTransformerDetector(config)
```

### 3. Training

```python
from maesy.training import Trainer, TrainingConfig

training_config = TrainingConfig(
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-4,
    save_dir="./checkpoints",
    log_dir="./logs",
    use_amp=True
)

trainer = Trainer(model, train_loader, val_loader, training_config)
trainer.train()
```

### 4. Evaluation

```python
from maesy.evaluation import evaluate_model

results = evaluate_model(
    model=model,
    data_loader=val_loader,
    confidence_threshold=0.5,
    iou_threshold=0.5
)

print(f"mAP: {results['mAP']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
```

### 5. Inference

```python
from maesy.evaluation import Evaluator
import torch

evaluator = Evaluator(model, data_loader)

# Load and preprocess image
image = load_image("test.jpg")  # Your preprocessing function

# Make prediction
predictions = evaluator.predict(image, confidence_threshold=0.5)

# Visualize
evaluator.visualize_predictions(
    image=image,
    predictions=predictions,
    category_names=["class1", "class2", ...],
    save_path="output.jpg"
)
```

## Configuration Options

### ModelConfig Parameters

- `image_size`: Input image size (default: 224)
- `patch_size`: Patch size for embedding (default: 16)
- `embed_dim`: Embedding dimension (default: 768)
- `num_layers`: Number of transformer encoder layers (default: 12)
- `num_heads`: Number of attention heads (default: 12)
- `num_classes`: Number of object classes (default: 80)
- `num_queries`: Number of object queries (default: 100)
- `mlp_ratio`: MLP expansion ratio (default: 4.0)
- `dropout`: Dropout rate (default: 0.1)

### TrainingConfig Parameters

- `num_epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size (default: 16)
- `learning_rate`: Learning rate (default: 1e-4)
- `weight_decay`: Weight decay (default: 1e-4)
- `warmup_epochs`: Number of warmup epochs (default: 5)
- `optimizer`: Optimizer type ("adamw", "adam", "sgd") (default: "adamw")
- `lr_scheduler`: LR scheduler type ("cosine", "step") (default: "cosine")
- `use_amp`: Use mixed precision training (default: False)
- `max_grad_norm`: Gradient clipping norm (default: 0.1)
- `save_frequency`: Checkpoint save frequency in epochs (default: 5)
- `log_frequency`: Logging frequency in steps (default: 10)

## Dataset Format

MaeSy uses COCO format annotations:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "object_class"
    }
  ]
}
```

## Model Architecture

```
Input Image (224x224x3)
    ↓
Patch Embedding (Conv2d)
    ↓
Add [CLS] Token + Positional Encoding
    ↓
Transformer Encoder Blocks (×12)
    ↓
Detection Head (Transformer Decoder)
    ↓
Classification Head + Bbox Head
    ↓
Predictions (logits, boxes)
```

## Tips and Best Practices

1. **Mixed Precision Training**: Enable `use_amp=True` for faster training on GPUs
2. **Batch Size**: Adjust based on GPU memory (typical: 8-32)
3. **Learning Rate**: Start with 1e-4 and adjust based on convergence
4. **Warmup**: Use 5-10 warmup epochs for stable training
5. **Image Size**: Larger images (384, 448) improve accuracy but require more memory
6. **Data Augmentation**: Implement custom transforms for better generalization
7. **Checkpointing**: Save checkpoints frequently during long training runs

## Common Issues

### Out of Memory
- Reduce batch size
- Reduce image size
- Reduce model size (embed_dim, num_layers)
- Enable gradient checkpointing

### Slow Training
- Enable mixed precision training (use_amp=True)
- Increase batch size
- Use multiple workers for data loading
- Use GPU instead of CPU

### Poor Performance
- Train for more epochs
- Increase model size
- Improve data quality
- Add data augmentation
- Adjust learning rate

## Examples

See the `examples/` directory for complete working examples:
- `examples/train.py` - Training script
- `examples/evaluate.py` - Evaluation script
- `examples/inference.py` - Inference script
