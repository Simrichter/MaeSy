"""Example script for classification pretraining."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from maesy.dataset import UnlabeledDataset, MaesyDataset
from maesy.model import ClassificationCNN, ClassificationCNNConfig, PatchClassificatorConfig, PatchClassificator
from maesy.model.classification_ViT_model import ClassificationViTConfig, ClassificationViT
from maesy.training import ClassificationTrainer
from maesy.training.base_trainer import BaseTrainingConfig

@dataclass
class PatchClassificationTrainingConfig(BaseTrainingConfig):
    """Configuration for Classification pretraining."""
    criterion: str = "ClassificationLoss"

    # Checkpoint and logging
    save_dir: str = "./patch_classification_checkpoints"


def train_patch_classification(dataset_path, enable_wandb, batch_size=64, num_epochs=50, num_classes=2, patch_shape: Tuple[int, int] = (24, 48)):
    """Main setup function."""
    
    # Create model config
    model_config = PatchClassificatorConfig(
        resnet_version="resnet18",
        embed_dim=256,
        num_classes=num_classes
    )

    model = PatchClassificator(model_config)
    
    # Data transforms (with augmentations for classification)
    train_transforms = transforms.Compose([
        transforms.Resize((patch_shape[0], patch_shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((patch_shape[0], patch_shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = MaesyDataset(dataset_path, split="train")
    
    val_dataset = UnlabeledDataset(
        Path(dataset_path)/"val/images",  # Replace with your validation data path
        transforms=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
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
    print(f"Number of classes: {num_classes}")
    
    # Create pretraining config
    pretraining_config = ClassificationPretrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_epochs=5,
        save_dir="./classification_checkpoints",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True
    )
    
    # Create pretrainer
    pretrainer = ClassificationTrainer(
        model=model,
        project_name="maesy-Classification_Pretraining",
        train_loader=train_loader,
        val_loader=val_loader,
        config=pretraining_config,
        enable_wandb=enable_wandb
    )
    
    # Start pretraining
    print("\nStarting classification pretraining...")
    pretrainer.train()
    
    print("\nClassification pretraining completed!")
    print(f"Best validation loss: {pretrainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {pretraining_config.save_dir}")
