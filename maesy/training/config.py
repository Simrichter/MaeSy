"""Training configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    criterion: str = "DetectionLoss"
    
    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd
    momentum: float = 0.9  # For SGD
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, step, multistep
    lr_step_size: int = 30  # For step scheduler
    lr_gamma: float = 0.1  # For step scheduler
    
    # Checkpoint and logging
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_frequency: int = 5
    log_frequency: int = 10
    
    # Device
    device: str = "cuda"  # cuda, cpu
    num_workers: int = 4
    
    # Gradient clipping
    max_grad_norm: float = 0.1
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    
    # Mixed precision training
    use_amp: bool = False
