"""Training configuration."""

from dataclasses import dataclass
from typing import Optional

import torch.cuda


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    criterion: str = "None"  # e.g., DetectionLoss, MaskedMSE, etc.
    
    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd
    momentum: float = 0.9  # For SGD
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, step, multistep
    lr_step_size: int = 30  # For step scheduler
    lr_gamma: float = 0.1  # For step scheduler
    
    # Checkpoint and logging
    save_dir: str = "./checkpoints"
    save_frequency: int = 10 # Save every n epochs
    log_frequency: int = 10 # Log every n global steps
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    
    # Mixed precision training
    use_amp: bool = True

@dataclass
class MAEPretrainingConfig(TrainingConfig):
    """Configuration for Masked Autoencoder pretraining."""
    criterion: str = "MaskedMSE"

    # MAE specific parameters
    mask_ratio: float = 0.75  # Ratio of patches to mask

    # Checkpoint and logging
    save_dir: str = "./MAEpretrain_checkpoints"
    output_predicted_images: bool = False # Whether to save predicted images during validation