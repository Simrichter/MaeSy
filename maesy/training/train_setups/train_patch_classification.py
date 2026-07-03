"""Example script for classification pretraining."""
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from maesy.dataset import MaesyDataset, TrainPatchTransforms, ValPatchTransforms
from maesy.model import PatchClassificatorConfig, PatchClassificator
from maesy.training import ClassificationTrainer
from maesy.training.base_trainer import BaseTrainingConfig
from maesy.training.utils import collate_classification_fn


@dataclass
class PatchClassificationTrainingConfig(BaseTrainingConfig):
    """Configuration for Classification pretraining."""
    criterion: str = "ClassificationLoss"

    # Checkpoint and logging
    save_dir: str = "./patch_classification_checkpoints"


def train_patches(dataset_path, enable_wandb, batch_size=64, num_epochs=50, num_classes=2, patch_shape: Tuple[int, int] = (24, 48)):
    """Main setup function."""

    # Create model config
    model_config = PatchClassificatorConfig(
        resnet_version="resnet18",
        head_in_dim=2304,
        num_classes=num_classes
    )

    model = PatchClassificator(model_config)

    # Data transforms (with augmentations for classification)
    train_transforms = TrainPatchTransforms()
    val_transforms = ValPatchTransforms()

    train_dataset = MaesyDataset(dataset_path, split="train", annotation_type="classification", transforms=train_transforms)

    val_dataset = MaesyDataset(dataset_path, split="val", annotation_type="classification", transforms=val_transforms)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_classification_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_classification_fn
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")

    # Create pretraining config
    pretraining_config = BaseTrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        criterion="ClassificationLoss",
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_epochs=5,
        save_dir="./classification_checkpoints",
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
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
