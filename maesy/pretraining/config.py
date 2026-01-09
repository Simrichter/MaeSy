"""Pretraining configuration classes."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MAEPretrainingConfig:
    """Configuration for Masked Autoencoder pretraining."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    
    # MAE specific parameters
    mask_ratio: float = 0.75  # Ratio of patches to mask
    
    # Optimizer
    optimizer: str = "adamw"  # adamw, adam
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, step
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    
    # Checkpoint and logging
    save_dir: str = "./pretrain_checkpoints"
    log_dir: str = "./pretrain_logs"
    save_frequency: int = 10
    log_frequency: int = 10
    output_predicted_images: bool = False

    device: str = "cuda"
    
    # Device
    device: str = "cuda"  # cuda, cpu
    num_workers: int = 4
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True


@dataclass
class ClassificationPretrainingConfig:
    """Configuration for classification pretraining."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Classification parameters
    num_classes: int = 1000  # ImageNet-1k default
    label_smoothing: float = 0.1
    
    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd
    momentum: float = 0.9  # For SGD
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, step
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    
    # Checkpoint and logging
    save_dir: str = "./pretrain_checkpoints"
    log_dir: str = "./pretrain_logs"
    save_frequency: int = 10
    log_frequency: int = 10
    
    # Device
    device: str = "cuda"  # cuda, cpu
    num_workers: int = 4
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True
