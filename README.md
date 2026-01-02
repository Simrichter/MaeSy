# MaeSy

Leveraging Pretraining and Synthetic Finetuning to Train Data-Efficient Vision Transformers in the Context of Robot Soccer

## Overview

MaeSy is a Python framework for training custom Vision Transformers for object detection. The framework provides a modular architecture with separate components for dataset management, model architecture, training, and evaluation.

## Features

- 🎯 **Custom Vision Transformer Architecture**: Built from scratch with attention mechanisms optimized for object detection
- 📦 **Dataset Management**: Easy-to-use tools for downloading, managing, and preprocessing datasets in COCO format
- 🚀 **Training Pipeline**: Complete training infrastructure with mixed precision support, learning rate scheduling, and checkpointing
- 📊 **Evaluation Tools**: Comprehensive evaluation metrics including mAP, precision, recall, and visualization utilities
- 🔧 **Modular Design**: Clean separation of concerns with independent modules for each major component

## Installation

### From Source

```bash
git clone https://github.com/Simrichter/MaeSy.git
cd MaeSy
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- Pillow >= 9.0.0
- pycocotools >= 2.0.4

## Framework Architecture

MaeSy consists of four main modules:

### 1. Dataset Module (`maesy.dataset`)

Handles dataset downloading, management, and data loading:

```python
from maesy.dataset import DatasetManager, ObjectDetectionDataset, get_train_transforms

# Initialize dataset manager
dataset_manager = DatasetManager(data_root="./data")

# Create dataset
dataset = ObjectDetectionDataset(
    images_dir="./data/train/images",
    annotations_file="./data/train/annotations.json",
    transforms=get_train_transforms(image_size=224)
)
```

### 2. Model Module (`maesy.model`)

Vision Transformer architecture for object detection:

```python
from maesy.model import VisionTransformerDetector, ModelConfig

# Configure model
config = ModelConfig(
    image_size=224,
    patch_size=16,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    num_classes=80,
    num_queries=100
)

# Create model
model = VisionTransformerDetector(config)
```

### 3. Training Module (`maesy.training`)

Complete training pipeline with monitoring and checkpointing:

```python
from maesy.training import Trainer, TrainingConfig

# Configure training
training_config = TrainingConfig(
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-4,
    save_dir="./checkpoints",
    log_dir="./logs"
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config
)

# Start training
trainer.train()
```

### 4. Evaluation Module (`maesy.evaluation`)

Comprehensive evaluation and visualization tools:

```python
from maesy.evaluation import evaluate_model, Evaluator

# Evaluate model
results = evaluate_model(
    model=model,
    data_loader=val_loader,
    confidence_threshold=0.5
)

# Visualize predictions
evaluator = Evaluator(model, val_loader)
predictions = evaluator.predict(image)
evaluator.visualize_predictions(image, predictions, save_path="output.jpg")
```

## Quick Start

### Training a Model

```python
import torch
from torch.utils.data import DataLoader
from maesy.dataset import ObjectDetectionDataset, get_train_transforms, collate_fn
from maesy.model import VisionTransformerDetector, ModelConfig
from maesy.training import Trainer, TrainingConfig

# Create dataset
train_dataset = ObjectDetectionDataset(
    images_dir="./data/train/images",
    annotations_file="./data/train/annotations.json",
    transforms=get_train_transforms(image_size=224)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# Create model
model_config = ModelConfig(num_classes=80)
model = VisionTransformerDetector(model_config)

# Train
trainer = Trainer(model, train_loader)
trainer.train()
```

See the [examples/](examples/) directory for complete training, evaluation, and inference scripts.

## Example Scripts

The `examples/` directory contains ready-to-use scripts:

- **`train.py`**: Complete training pipeline example
- **`evaluate.py`**: Model evaluation with metrics computation
- **`inference.py`**: Single image inference with visualization

Run an example:

```bash
python examples/train.py
```

## Model Architecture

The Vision Transformer Detector consists of:

1. **Patch Embedding**: Converts images into patch embeddings
2. **Transformer Encoder**: Multi-layer transformer with self-attention
3. **Detection Head**: Transformer decoder with object queries for detection
4. **Prediction Heads**: Classification and bounding box regression

Key features:
- Learnable positional encodings
- Multi-head self-attention
- Hungarian matching for training
- DETR-style object detection

## Dataset Format

MaeSy supports COCO format datasets. Your dataset should have:

```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

The `annotations.json` should follow the COCO format with:
- `images`: List of image metadata
- `annotations`: List of bounding box annotations
- `categories`: List of object categories

## Configuration

### Model Configuration

```python
ModelConfig(
    image_size=224,           # Input image size
    patch_size=16,            # Patch size for embedding
    embed_dim=768,            # Embedding dimension
    num_layers=12,            # Number of transformer layers
    num_heads=12,             # Number of attention heads
    num_classes=80,           # Number of object classes
    num_queries=100,          # Number of object queries
)
```

### Training Configuration

```python
TrainingConfig(
    num_epochs=100,           # Number of training epochs
    batch_size=16,            # Batch size
    learning_rate=1e-4,       # Learning rate
    weight_decay=1e-4,        # Weight decay
    warmup_epochs=5,          # Warmup epochs
    use_amp=True,             # Mixed precision training
    save_dir="./checkpoints", # Checkpoint directory
    log_dir="./logs"          # TensorBoard log directory
)
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=./logs
```

## Evaluation Metrics

The framework computes:
- **mAP** (mean Average Precision)
- **Precision** and **Recall**
- **F1 Score**
- **Per-class AP** (Average Precision for each class)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MaeSy in your research, please cite:

```bibtex
@software{maesy2026,
  title={Leveraging Pretraining and Synthetic
Finetuning to Train Data-Efficient
Vision Transformers in the Context of
Robot Soccer},
  author={Simon Ian Richter},
  year={2026},
  url={https://github.com/Simrichter/MaeSy}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This framework is designed for training Vision Transformers in the context of robot soccer and general object detection tasks.
