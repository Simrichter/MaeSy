"""Example script for MAE (Masked Autoencoder) pretraining."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from maesy.model import MAEConfig, MaskedAutoencoderViT
from maesy.dataset import UnlabeledDataset

from maesy.training import MaeTrainer, MAEPretrainingConfig


def testMAE():
    """Main MAE pretraining function."""

    # Configuration
    image_size = 224
    batch_size = 128
    num_epochs = 200
    mask_ratio = 0.5

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
    model = MaskedAutoencoderViT(
        config=mae_config
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Prepare dataset
    # Adjust these paths to your dataset
    print("Loading datasets...")

    # Data transforms
    train_transforms = transforms.Compose([
        # transforms.Resize(size=(image_size, image_size)),
        # transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet Mean and Std values, recompute for other datasets!
    ])

    val_transforms = transforms.Compose([
        # transforms.Resize(size=(image_size, image_size)),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_path = Path(r"/home/simon/Desktop/maesy-training/data/BeijingDataset")

    train_dataset = UnlabeledDataset(
        dataset_path/"train",
        transforms=train_transforms,
        repeat_factor=2,
        # use_first_n=384,
        filetype=".jpg"
    )

    val_dataset = UnlabeledDataset(
        dataset_path/"val",
        transforms=val_transforms,
        # use_first_n=640, # 5 batches
        filetype=".jpg"
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
        weight_decay=0.01,
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
        config=pretraining_config
    )

    checkpoint = "/home/simon/Desktop/maesy-training/mae_checkpoints/pious-vortex-40/latest_model.pth"
    if Path(checkpoint).exists():
        pretrainer.load_checkpoint(checkpoint)

    # Start pretraining
    print("\nStarting MAE pretraining...")
    pretrainer.train()

    print("\nMAE pretraining completed!")
    print(f"Best validation loss: {pretrainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {pretraining_config.save_dir}")
    # print("\nTo use the pretrained weights for object detection:")
    # print("  from maesy.pretraining import load_mae_pretrained_weights")
    # print("  detector = VisionTransformerDetector(config)")
    # print(f"  detector = load_mae_pretrained_weights(detector, '{pretraining_config.save_dir}/mae_best_model.pth')")


if __name__ == "__main__":
    testMAE()
