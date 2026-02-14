# Object Detection with Vision Transformers

This document describes how to use the Vision Transformer architecture for object detection in the MaeSy framework.

## Overview

The implementation provides two approaches for object detection:

1. **ViTDetector** - Train a Vision Transformer for object detection from scratch
2. **TransformerDetectionModel** - Use MAE pretrained backbones for object detection (transfer learning)

Both models follow the MaeSy framework structure with separate backbone and head components.

## Architecture

```
Input Image [B, 3, H, W]
    ↓
Patchification [B, N, P²×3]
    ↓
TransformerBackbone (Encoder)
    - Patch Embedding
    - Positional Encoding  
    - Transformer Blocks (12 layers)
    ↓
Features [B, N, D]
    ↓
DetectionHead (Decoder)
    - Object Queries (100 learnable embeddings)
    - Transformer Decoder (6 layers)
    - Classification Head → pred_logits [B, 100, C+1]
    - Bounding Box Head → pred_boxes [B, 100, 4]
```

## Quick Start

### 1. Prepare Your Dataset

The implementation uses YOLO format for object detection datasets:

```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
└── val/
    ├── images/
    └── labels/
```

Label format (YOLO): `class_id center_x center_y width height` (normalized to [0,1])

### 2. Train from Scratch

```python
from maesy.model import ViTDetector, ViTDetectorConfig
from maesy.training import DetectionTrainer, TrainingConfig
from maesy.training.utils import collate_detection_fn
from maesy.dataset import ObjectDetectionDataset
from torch.utils.data import DataLoader

# Create model
config = ViTDetectorConfig(
    image_size=224,
    patch_size=16,
    num_classes=80,  # Your number of classes
    num_queries=100
)
model = ViTDetector(config)

# Load dataset
train_dataset = ObjectDetectionDataset("dataset/train")
train_loader = DataLoader(
    train_dataset, 
    batch_size=8,
    collate_fn=collate_detection_fn,
    shuffle=True
)

# Train
trainer = DetectionTrainer(
    model=model,
    train_loader=train_loader,
    config=TrainingConfig(criterion="DetectionLoss"),
    project_name="my_detector"
)
trainer.train()
```

### 3. Use MAE Pretrained Backbone

```python
from maesy.model import MaskedAutoencoderViT, MAEConfig
from maesy.model import TransformerDetectionModel, TransformerDetectorConfig

# Load pretrained MAE model
mae_model = MaskedAutoencoderViT(MAEConfig())
checkpoint = torch.load("mae_pretrained.pth")
mae_model.load_state_dict(checkpoint['model_state_dict'])

# Create detection model
det_model = TransformerDetectionModel(TransformerDetectorConfig())

# Transfer backbone weights
det_model.backbone.load_state_dict(mae_model.backbone.state_dict())

# Optionally freeze backbone
for param in det_model.backbone.parameters():
    param.requires_grad = False

# Train detection head
trainer = DetectionTrainer(model=det_model, ...)
trainer.train()
```

## Model Configurations

### ViTDetectorConfig

```python
@dataclass
class ViTDetectorConfig:
    # Image parameters
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    
    # Backbone parameters
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Detection head parameters
    num_classes: int = 80
    num_queries: int = 100
    num_decoder_layers: int = 6
    decoder_num_heads: int = 8
    hidden_dim: int = 256
    
    # Loss weights
    bbox_loss_coef: float = 5.0
    class_loss_coef: float = 1.0
    giou_loss_coef: float = 2.0
```

### TrainingConfig for Detection

```python
TrainingConfig(
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-4,
    optimizer="adamw",
    lr_scheduler="cosine",
    warmup_epochs=5,
    criterion="DetectionLoss",  # Important: use DetectionLoss
    use_amp=True,  # Automatic mixed precision
    max_grad_norm=1.0
)
```

## Training Tips

### 1. Learning Rate
- **From scratch**: Start with `1e-4`
- **With frozen backbone**: Use `1e-3` (higher LR for head only)
- **Fine-tuning**: Start with `1e-4` to `1e-5`

### 2. Batch Size
- Recommended: 8-16 per GPU
- Use gradient accumulation if memory is limited

### 3. Data Augmentation
Create custom transforms for your dataset:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

dataset = ObjectDetectionDataset("path", transforms=transform)
```

### 4. Number of Queries
- Start with 100 queries (default)
- Increase if you have many objects per image
- Decrease for simpler scenes or faster inference

## Inference

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = ViTDetector(config)
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess image
image = Image.open("image.jpg").convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    predictions = model(image_tensor)

# Post-process
pred_logits = predictions['pred_logits']  # [1, 100, num_classes+1]
pred_boxes = predictions['pred_boxes']    # [1, 100, 4] in [cx, cy, w, h]

# Filter by confidence
probs = torch.softmax(pred_logits, dim=-1)
scores, labels = probs[0, :, :-1].max(dim=-1)  # Exclude no-object class
keep = scores > 0.5

detected_boxes = pred_boxes[0][keep]
detected_labels = labels[keep]
detected_scores = scores[keep]
```

## Loss Function

The implementation uses **DetectionLoss** with:
- **Hungarian matching** for assigning predictions to ground truth
- **Classification loss** (cross-entropy)
- **Bounding box loss** (L1 loss)
- **GIoU loss** (Generalized Intersection over Union)

Loss weights can be configured:
```python
ViTDetectorConfig(
    bbox_loss_coef=5.0,    # Weight for bbox L1 loss
    class_loss_coef=1.0,   # Weight for classification loss
    giou_loss_coef=2.0     # Weight for GIoU loss
)
```

## Monitoring Training

Training is logged to Weights & Biases (wandb):
- Total loss
- Classification loss
- Bounding box loss
- GIoU loss
- Learning rate

Access your runs at: https://wandb.ai/

## Example Scripts

See `examples/object_detection_training.py` for complete examples:

```bash
# Train from scratch
python examples/object_detection_training.py \
    --mode scratch \
    --dataset /path/to/dataset \
    --output ./checkpoints

# Train with MAE pretrained backbone
python examples/object_detection_training.py \
    --mode mae_pretrained \
    --mae_checkpoint mae_pretrained.pth \
    --dataset /path/to/dataset \
    --freeze_backbone

# Run inference
python examples/object_detection_training.py \
    --mode inference \
    --checkpoint checkpoint.pth \
    --image test_image.jpg
```

## Performance Tips

1. **Use mixed precision training** (`use_amp=True`) for faster training
2. **Adjust num_queries** based on your dataset complexity
3. **Use MAE pretraining** for better performance with limited data
4. **Fine-tune backbone** after training head for best results
5. **Use data augmentation** to improve generalization

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce `embed_dim` or `num_layers`
- Use gradient accumulation
- Enable mixed precision training

### Poor Detection Performance
- Check if bounding boxes are normalized correctly
- Increase number of training epochs
- Use MAE pretraining
- Adjust loss weights
- Add data augmentation

### Slow Training
- Enable mixed precision (`use_amp=True`)
- Increase batch size if memory allows
- Use multiple GPUs
- Reduce number of queries or decoder layers

## References

This implementation is based on:
- Vision Transformer (ViT) architecture
- DETR (DEtection TRansformer)
- Masked Autoencoder (MAE) pretraining

For more details, see the main MaeSy documentation.
