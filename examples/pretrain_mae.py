"""Example script for MAE (Masked Autoencoder) pretraining."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from maesy.model import ModelConfig
from maesy.pretraining import (
    MaskedAutoencoderViT,
    MaskedAutoencoderPretrainer,
    MAEPretrainingConfig
)


def main():
    """Main MAE pretraining function."""
    
    # Configuration
    image_size = 224
    batch_size = 64
    num_epochs = 100
    mask_ratio = 0.75
    
    # Create model config
    model_config = ModelConfig(
        image_size=image_size,
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        in_channels=3
    )
    
    # Create MAE model
    print("Creating MAE model...")
    model = MaskedAutoencoderViT(
        config=model_config,
        decoder_embed_dim=512,
        decoder_num_layers=8
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Prepare dataset
    # Example: Using ImageNet or any image dataset
    # Adjust these paths to your dataset
    print("Loading datasets...")
    
    # Data transforms (simple augmentations for MAE)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Example: Load ImageFolder dataset
    # Replace these paths with your actual dataset paths
    train_dataset = datasets.ImageFolder(
        root="./data/train",  # Replace with your training data path
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root="./data/val",  # Replace with your validation data path
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create pretraining config
    pretraining_config = MAEPretrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=1.5e-4,
        weight_decay=0.05,
        warmup_epochs=10,
        mask_ratio=mask_ratio,
        save_dir="./mae_checkpoints",
        log_dir="./mae_logs",
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=4,
        use_amp=True
    )
    
    # Create pretrainer
    pretrainer = MaskedAutoencoderPretrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=pretraining_config
    )
    
    # Start pretraining
    print("\nStarting MAE pretraining...")
    pretrainer.train()
    
    print("\nMAE pretraining completed!")
    print(f"Best validation loss: {pretrainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {pretraining_config.save_dir}")
    print(f"Logs saved to: {pretraining_config.log_dir}")
    print("\nTo use the pretrained weights for object detection:")
    print("  from maesy.pretraining import load_mae_pretrained_weights")
    print("  detector = VisionTransformerDetector(config)")
    print(f"  detector = load_mae_pretrained_weights(detector, '{pretraining_config.save_dir}/mae_best_model.pth')")


if __name__ == "__main__":
    main()
