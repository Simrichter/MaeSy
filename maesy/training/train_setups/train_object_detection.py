"""
Example: Training a Vision Transformer for Object Detection

This script demonstrates how to train both ViTDetector
for object detection using the MaeSy framework.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from torchvision import transforms

# Import models
from maesy.model import ViTDetector, ViTDetectorConfig
from maesy.model import MaskedAutoencoderViT, MAEConfig

# Import training components
from maesy.training import DetectionTrainer, TrainingConfig
from maesy.training.utils import collate_detection_fn

# Import dataset
from maesy.dataset import ObjectDetectionDataset


def train_vit_detector(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    freeze_backbone: bool,
    continue_from_checkpoint: bool,
    enable_wandb: bool
):
    """
    Train an object detection model using MAE pretrained backbone.

    Args:
        checkpoint_path: Path to pretrained MAE checkpoint
        dataset_path: Path to object detection dataset
        output_dir: Directory to save checkpoints
        freeze_backbone: Whether to freeze the backbone during training
        continue_from_checkpoint: Whether to continue training from an existing OD checkpoint (in that case, checkpoint_path should point to an OD checkpoint instead of a MAE checkpoint)
        enable_wandb: Whether to enable Weights & Biases logging
    """
    print("=" * 60)
    print("Training with MAE Pretrained Backbone")
    print("=" * 60)

    # Create detection model with same backbone configuration
    det_config = ViTDetectorConfig(
        image_size=224,
        patch_size=16,
        in_channels=3,

        # Backbone parameters
        embed_dim=384,
        num_layers=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,

        # Detection head parameters
        num_classes=3,
        num_queries=100,
        num_decoder_layers=3,
        decoder_num_heads=4,
        hidden_dim=256,

        # Loss weights
        bbox_loss_coef=5.0,
        class_loss_coef=1.0,
        giou_loss_coef=2.0
    )

    model = ViTDetector(det_config)
    # Optionally freeze backbone
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Froze backbone parameters - only training detection head")
    else:
        print("Fine-tuning entire model (backbone + head)")

    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Create datasets and dataloaders
    train_dataset = ObjectDetectionDataset(f"{dataset_path}/train", transforms=train_transforms)
    val_dataset = ObjectDetectionDataset(f"{dataset_path}/val", transforms=val_transforms)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_detection_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=4,
        collate_fn=collate_detection_fn,
        pin_memory=True
    )

    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=100,
        learning_rate=1e-4 if not freeze_backbone else 1e-3,  # Higher LR when only training head
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler="cosine",
        warmup_epochs=5,
        save_dir=output_dir,
        criterion="DetectionLoss",
        use_amp=True
    )

    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        project_name="mae_pretrained_detection",
        enable_wandb=enable_wandb
    )

    if checkpoint_path != "":
        # Transfer backbone weights
        trainer.load_checkpoint(checkpoint_path, model_only=not continue_from_checkpoint)

    # Train
    trainer.train()

def inference_example(checkpoint_path: str, image_path: str):
    """
    Example of using trained model for inference.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        image_path: Path to input image
    """
    print("=" * 60)
    print("Inference Example")
    print("=" * 60)
    
    # Load model
    config = ViTDetectorConfig(num_classes=80)
    model = ViTDetector(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    from PIL import Image
    from torchvision import transforms
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Post-process predictions
    pred_logits = predictions['pred_logits']  # [1, num_queries, num_classes+1]
    pred_boxes = predictions['pred_boxes']    # [1, num_queries, 4]
    
    # Get class probabilities and filter by confidence
    probs = torch.softmax(pred_logits, dim=-1)
    scores, labels = probs[0, :, :-1].max(dim=-1)  # Exclude no-object class
    
    confidence_threshold = 0.5
    keep = scores > confidence_threshold
    
    # Get filtered detections
    detected_boxes = pred_boxes[0][keep]
    detected_labels = labels[keep]
    detected_scores = scores[keep]
    
    print(f"Detected {keep.sum().item()} objects:")
    for i, (box, label, score) in enumerate(zip(detected_boxes, detected_labels, detected_scores)):
        cx, cy, w, h = box
        print(f"  Object {i+1}: class={label.item()}, score={score.item():.3f}, "
              f"bbox=({cx.item():.3f}, {cy.item():.3f}, {w.item():.3f}, {h.item():.3f})")


# if __name__ == "__main__":
#     args = None
#
#     if args.mode == "scratch":
#         train_vit_detector(args.dataset, args.output)
#     elif args.mode == "mae_pretrained":
#         train_with_mae_pretrained(
#             args.mae_checkpoint,
#             args.dataset,
#             args.output,
#             args.freeze_backbone
#         )
#     elif args.mode == "inference":
#         inference_example(args.checkpoint, args.image)
#     else:
#         parser.print_help()
