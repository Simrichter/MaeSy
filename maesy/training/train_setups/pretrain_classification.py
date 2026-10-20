"""Example script for classification pretraining."""
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from _maesy_core.dataset import MaesyDataset
from maesy.training.collate_functions import collate_classification_fn
from _maesy_core.model import ClassificationCNN, ClassificationCNNConfig
from maesy.training import ClassificationTrainer
from maesy.training.base_trainer import BaseTrainingConfig

@dataclass
class ClassificationPretrainingConfig(BaseTrainingConfig):
    """Configuration for Classification pretraining."""
    criterion: str = "ClassificationLoss"

    # Checkpoint and logging
    save_dir: str = "./classification_pretrain_checkpoints"

def train_classification(dataset_path, enable_wandb, batch_size=64, num_epochs=50, num_classes=3):
    """Main classification pretraining function."""
    
    # Create model config
    model_config = ClassificationCNNConfig(
        resnet_version="resnet50",
        embed_dim=512,
        num_classes=num_classes
    )

    model = ClassificationCNN(model_config)
    if model.head.config.num_classes != num_classes:
        raise ValueError(
            f"Classification model head has {model.head.config.num_classes} outputs, "
            f"but {num_classes} classes were requested"
        )
    
    # Data transforms (with augmentations for classification)
    train_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = MaesyDataset(
        dataset_path,
        split="train",
        annotation_type="classification",
        transforms=train_transforms,
    )
    val_dataset = MaesyDataset(
        dataset_path,
        split="val",
        annotation_type="classification",
        transforms=val_transforms,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Classification training and validation splits must both contain images")

    observed_labels = {
        int(image.stem.rsplit("_", 1)[-1])
        for dataset in (train_dataset, val_dataset)
        for image in dataset.images
    }
    invalid_labels = sorted(label for label in observed_labels if label < 0 or label >= num_classes)
    if invalid_labels:
        raise ValueError(
            f"Classification dataset labels {invalid_labels} are outside the "
            f"model class range [0, {num_classes})"
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_classification_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_classification_fn,
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
