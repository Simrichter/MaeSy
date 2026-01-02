"""Example training script for Vision Transformer object detection."""

import torch
from torch.utils.data import DataLoader

from maesy.dataset import DatasetManager, ObjectDetectionDataset, get_train_transforms, get_val_transforms
from maesy.dataset.transforms import collate_fn
from maesy.model import VisionTransformerDetector, ModelConfig
from maesy.training import Trainer, TrainingConfig


def main():
    """Main training function."""
    
    # Configuration
    image_size = 224
    num_classes = 80  # COCO dataset has 80 classes
    batch_size = 8
    num_epochs = 100
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(data_root="./data")
    
    # Example: Download and prepare dataset
    # For this example, you would need to provide actual dataset URLs
    # dataset_manager.download_dataset(
    #     url="YOUR_DATASET_URL",
    #     dataset_name="my_dataset",
    #     extract=True
    # )
    
    # Prepare dataset
    # Replace these paths with your actual dataset paths
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
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create training config
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_epochs=5,
        save_dir="./checkpoints",
        log_dir="./logs",
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=4,
        use_amp=True  # Use mixed precision training
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # Start training
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {training_config.save_dir}")
    print(f"Logs saved to: {training_config.log_dir}")


if __name__ == "__main__":
    main()
