"""Example script for MAE (Masked Autoencoder) pretraining."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from maesy.model import MAEConfig, MaskedAutoencoderViT
from maesy.dataset import UnlabeledDataset
from maesy.model_tools.model_factory import create_model

from maesy.training import MaeTrainer, MAEPretrainingConfig


def train_mae(
    dataset_path,
    image_size = 224,
    batch_size = 128,
    num_epochs = 200,
    mask_ratio = 0.5,
    checkpoint = "",
    enable_wandb = True
):
    """Main MAE pretraining function."""
    # Configuration
      # TODO: Make mask_ratio scheduled

    # Create training config
    mae_config = MAEConfig(
        image_size=image_size,
        patch_size=16,
        embed_dim=384,
        num_layers=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,
        decoder_embed_dim=384,
        decoder_num_layers=4
    )

    # Create MAE model
    print("Creating MAE model...")
    model = create_model("mae", mae_config)

    # Prepare dataset
    print("Loading datasets...")
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = UnlabeledDataset(
        Path(dataset_path) / "train",
        transforms=train_transforms
    )

    val_dataset = UnlabeledDataset(
        Path(dataset_path) / "val",
        transforms=val_transforms
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
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create pretraining config
    pretraining_config = MAEPretrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=0.001,
        warmup_epochs=5,
        mask_ratio=mask_ratio,
        save_dir="./mae_checkpoints",
        save_frequency=100,
        output_predicted_images=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=4,
        use_amp=True
    )

    # Create pretrainer
    pretrainer = MaeTrainer(
        model=model,
        project_name="maesy-MAE_Pretraining",
        train_loader=train_loader,
        val_loader=val_loader,
        config=pretraining_config,
        enable_wandb=enable_wandb
    )


    if checkpoint != "" and Path(checkpoint).exists():
        pretrainer.load_checkpoint(checkpoint)

    # Start pretraining
    print("\nStarting MAE pretraining...")
    pretrainer.train()

    print("\nMAE pretraining completed!")
    print(f"Best validation loss: {pretrainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {pretraining_config.save_dir}")
