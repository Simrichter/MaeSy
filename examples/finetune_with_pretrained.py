"""Example script for fine-tuning with pretrained weights."""

import torch
from torch.utils.data import DataLoader

from maesy.dataset import ObjectDetectionDataset, get_train_transforms, get_val_transforms
from maesy.dataset.transforms import collate_fn
from maesy.model import VisionTransformerDetector, ModelConfig
from maesy.training import Trainer, TrainingConfig
from maesy.pretraining import load_mae_pretrained_weights, load_classification_pretrained_weights, freeze_encoder


def main():
    """Main fine-tuning function with pretrained weights."""
    
    # Configuration
    image_size = 224
    num_classes = 80  # COCO dataset has 80 classes
    batch_size = 8
    num_epochs = 50  # Fewer epochs needed with pretraining
    
    # Choose which pretrained weights to use
    use_mae_pretrained = True  # Set to True to use MAE pretrained weights
    use_classification_pretrained = False  # Set to True to use classification pretrained weights
    freeze_encoder_weights = False  # Set to True to freeze encoder during fine-tuning
    
    # Paths to pretrained checkpoints
    mae_checkpoint = "./mae_checkpoints/mae_best_model.pth"
    classification_checkpoint = "./classification_checkpoints/classification_best_model.pth"
    
    # Prepare dataset paths
    train_images_dir = "./data/train/images"
    train_annotations = "./data/train/annotations.json"
    val_images_dir = "./data/val/images"
    val_annotations = "./data/val/annotations.json"
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ObjectDetectionDataset(
        images_dir=train_images_dir,
        annotations_file=train_annotations,
        transforms=get_train_transforms(image_size=image_size)
    )
    
    val_dataset = ObjectDetectionDataset(
        images_dir=val_images_dir,
        annotations_file=val_annotations,
        transforms=get_val_transforms(image_size=image_size)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model_config = ModelConfig(
        image_size=image_size,
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_classes=num_classes,
        num_queries=100
    )
    
    model = VisionTransformerDetector(model_config)
    
    # Load pretrained weights
    if use_mae_pretrained:
        print(f"\nLoading MAE pretrained weights from {mae_checkpoint}...")
        try:
            model = load_mae_pretrained_weights(model, mae_checkpoint, strict=False)
            print("MAE pretrained weights loaded successfully!")
        except FileNotFoundError:
            print(f"Warning: MAE checkpoint not found at {mae_checkpoint}")
            print("Training from scratch...")
        except (RuntimeError, KeyError, ValueError) as e:
            print(f"Warning: Failed to load MAE checkpoint: {e}")
            print("Training from scratch...")
    
    if use_classification_pretrained:
        print(f"\nLoading classification pretrained weights from {classification_checkpoint}...")
        try:
            model = load_classification_pretrained_weights(model, classification_checkpoint, strict=False)
            print("Classification pretrained weights loaded successfully!")
        except FileNotFoundError:
            print(f"Warning: Classification checkpoint not found at {classification_checkpoint}")
            print("Training from scratch...")
        except (RuntimeError, KeyError, ValueError) as e:
            print(f"Warning: Failed to load classification checkpoint: {e}")
            print("Training from scratch...")
    
    # Optionally freeze encoder
    if freeze_encoder_weights:
        print("\nFreezing encoder weights...")
        model = freeze_encoder(model)
        print("Encoder frozen. Only detection head will be trained.")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Create training config
    # Use lower learning rate when fine-tuning from pretrained weights
    learning_rate = 1e-5 if (use_mae_pretrained or use_classification_pretrained) else 1e-4
    
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        warmup_epochs=3,
        save_dir="./finetuned_checkpoints",
        log_dir="./finetuned_logs",
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=4,
        use_amp=True
    )
    
    print(f"\nUsing learning rate: {learning_rate}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # Start training
    print("\nStarting fine-tuning...")
    trainer.train()
    
    print("\nFine-tuning completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {training_config.save_dir}")
    print(f"Logs saved to: {training_config.log_dir}")


if __name__ == "__main__":
    main()
