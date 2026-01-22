from dataclasses import asdict
from typing import Protocol
from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any

from .config import TrainingConfig
from .losses import DetectionLoss, MaskedMSE, BaseLoss
from ..model import VisionTransformerDetector, ModelConfig, BaseModel


class BaseTrainer(ABC):

    def __init__(
            self,
            model: BaseModel,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            config: Optional[TrainingConfig] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        # Setup device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        self.loss: BaseLoss = self._create_loss()

        # Setup directories
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup wandb
        self.wandb_run = wandb.init(
            entity="simon-richter-tu-dortmund",
            project="maesy-Finetuning",
            config=asdict(self.config)
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs - self.config.warmup_epochs
            )
        elif self.config.lr_scheduler.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        else:
            print(f"Warning: Unknown scheduler '{self.config.lr_scheduler}'")
            return None

    def _create_loss(self) -> Optional[BaseLoss]:
        if self.config.criterion == "DetectionLoss":
            loss = DetectionLoss(
                num_classes=self.model.config.num_classes,
                bbox_loss_coef=self.model.config.bbox_loss_coef,
                class_loss_coef=self.model.config.class_loss_coef,
                giou_loss_coef=self.model.config.giou_loss_coef
            )
            return loss
        elif self.config.criterion == "MaskedMSE":
            return MaskedMSE()
        else:
            print(f"Warning: Unknown loss '{self.config.criterion}'")
            return None

    def forward_model(self, images: torch.Tensor, targets: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Manages the forward pass through the model.
        Can be overwritten to add model-specific preprocessing
        """
        return self.loss(self.model(images), targets)

    def handle_raw_batch(self, batch: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract images and targets from raw batch data.
        Can be overwritten to handle different batch formats.
        """
        targets = None
        if isinstance(batch, dict):
            images = batch['images']
            targets = batch['targets']
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            targets = batch[1]
        else:
            images = batch

        images = images.to(self.device, non_blocking=True)
        if targets is not None:
            targets = batch['targets'].to(self.device, non_blocking=True)
        return images, targets

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.loss.reset_metrics()

        for batch_idx, batch in enumerate(pbar := tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")):
            images, targets = self.handle_raw_batch(batch)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                losses = self.forward_model(images, targets)
                loss = losses['loss']

            # Backward pass
            self.optimizer.zero_grad()

            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Log to tensorboard
            if self.global_step % self.config.log_frequency == 0:
                data = {f"train/{k}": v.item() for k, v in losses.items()}
                data['train/lr'] = self.optimizer.param_groups[0]['lr']
                self.wandb_run.log(data=data, step=self.global_step, commit=True)

            self.global_step += 1

        return self.loss.get_metrics()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.loss.reset_metrics()
        self.model.eval()

        for batch in tqdm(self.val_loader, desc="Validation"):
            images, targets = self.handle_raw_batch(batch)

            _ = self.forward_model(images, targets) # return value "losses" is not needed, because the loss function automatically accumulates the average metrics

        return self.loss.get_metrics()

    def train(self) -> None:
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            if self.val_loader is not None:
                val_metrics = {f"val/{k}": v for k, v in self.validate().items()}
                self.wandb_run.log(data=val_metrics)

                print(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val/total_loss']:.4f}")

                # Save best model
                if val_metrics['val/total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val/total_loss']
                    self.save_checkpoint('best_model.pth')
            else:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}")

            # Step scheduler
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

        # Save final model
        self.save_checkpoint('final_model.pth')
        self.wandb_run.finish()
        print("Training completed!")

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {filepath}")
