"""Example script for MAE (Masked Autoencoder) pretraining."""
from dataclasses import dataclass

import torch

from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from _maesy_core.model.model_tools.model_factory import create_model_from_dict
from maesy.training import BaseTrainingConfig
from _maesy_core.dataset import MaesyDataset
from _maesy_core.model.model_tools.model_factory import known_architectures, read_yaml, create_model_from_checkpoint

from maesy.training import MaeTrainer

@dataclass
class MAETrainingConfig(BaseTrainingConfig):
    """Configuration for training."""

    # Training parameters
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    backbone_learning_rate: float = learning_rate
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    criterion: str = "None"  # e.g., DetectionLoss, MaskedMSE, etc.

    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd
    momentum: float = 0.9  # For SGD

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, step, multistep
    lr_step_size: int = 30  # For step scheduler

    # Checkpoint and logging
    save_dir: str = "./checkpoints"
    save_frequency: int = 10  # Save every n epochs
    log_frequency: int = 10  # Log every n global steps

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    num_workers: int = 4

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Mixed precision training
    use_amp: bool = True

def train_mae(
    model_info,
    dataset_path,
    continue_training_from_checkpoint: bool,
    image_size = 224,
    batch_size = 64,
    num_epochs = 200,
    mask_ratio = 0.5,
    output_dir: str = "mae_checkpoints",
    enable_wandb = True,
    seed: int = 42,
):
    """
        Main MAE pretraining function.
    """
      # TODO: Make mask_ratio scheduled
    print("=" * 60)
    print("Starting MAE pretraining")
    print("=" * 60)

    torch.manual_seed(seed)

    # Create training configuration
    training_config = MAETrainingConfig(
        batch_size=batch_size,
        num_epochs=3000, # 750,
        learning_rate=1e-4,
        backbone_learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler= "plateau", # "cosine",
        plateau_metric="val_losses/total_loss",
        plateau_mode="min",
        patience=80,
        lr_step_factor=0.3,
        min_num_epochs_per_plateau=100,
        warmup_epochs=4,
        save_frequency=100,
        log_frequency=50,
        save_dir=output_dir,
        criterion="MaskedMSE",  # "YOLOv8Loss", #
        use_amp=True,
    )

    # Create MAE model
    if model_info.lower() in known_architectures:
        mae_config = read_yaml(f"cfg/{model_info.lower()}.yaml")
        model = create_model_from_dict(mae_config)
    elif not model_info.endswith(".pth"):
        raise ValueError(f"Model {model_info} is neither in {known_architectures} nor is it a path to a training checkpoint (must end with '.pth')")
    else:
        model = create_model_from_checkpoint(model_info)


    # Prepare dataset
    print("Loading datasets...")

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

    train_dataset = MaesyDataset(dataset_path, "train", "None", transforms=train_transforms)
    val_dataset = MaesyDataset(dataset_path, "val", "None", transforms=val_transforms)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Create pretrainer
    pretrainer = MaeTrainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        project_name="maesy-multiscale-mae",
        config=training_config,
        enable_wandb=enable_wandb
    )

    # Start pretraining
    if continue_training_from_checkpoint:
        print("Continuing training from checkpoint")
        pretrainer.resume(model_info)
    else:
        print("\nStarting MAE pretraining...")
        pretrainer.train()

    print("\nMAE pretraining completed!")
    print(f"Best validation loss: {pretrainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {training_config.save_dir}")

if __name__ == "__main__":
    train_mae("mae-multiscale", "data/RobertLE", enable_wandb=False)