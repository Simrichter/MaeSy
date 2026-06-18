"""Training configuration."""

from dataclasses import dataclass
from typing import Optional

import torch.cuda


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    num_epochs: int = 100
    # batch_size: int = 16
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    warmup_epochs: int = 5
    criterion: str = "None"  # e.g., DetectionLoss, MaskedMSE, etc.
    
    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd
    momentum: float = 0.9  # For SGD
    
    # Learning rate schedule
    learning_rate: float = 1e-4
    min_lr: float = 1e-7
    early_stop_on_min_lr: bool = True
    patience: int = 10  # For ReduceLROnPlateau scheduler
    min_num_epochs_per_plateau: int = 100 # For ReduceLROnPlateau scheduler
    backbone_learning_rate: float = learning_rate / 10
    lr_scheduler: str = "cosine"  # cosine, step, multistep
    lr_step_size: int = 30  # For step scheduler
    lr_step_factor: float = 0.3  # For step scheduler
    
    # Checkpoint and logging
    save_checkpoints: bool = True
    save_dir: str = "./checkpoints"
    save_frequency: int = 10 # Save every n epochs
    log_frequency: int = 10 # Log every n global steps
    
    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # num_workers: int = 4
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    
    # Mixed precision training
    use_amp: bool = True

@dataclass
class ClassificationPretrainingConfig(TrainingConfig):
    """Configuration for Classification pretraining."""
    criterion: str = "ClassificationLoss"

    # Checkpoint and logging
    save_dir: str = "./classification_pretrain_checkpoints"