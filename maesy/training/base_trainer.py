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
from .losses import DetectionLoss, MaskedMSE
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

        self.loss, self.loss_metrics = self._create_loss()

        # Setup directories
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

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

    def _create_loss(self):
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

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        for batch_idx, batch in enumerate(pbar := tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")):
            images = batch['images'].to(self.device)
            images, preprocess_data = self.model.preprocess(images)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in target.items()}
                for target in batch['targets']
            ]

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                predictions = self.model(images)
                losses = self.loss(predictions, targets)
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
                metrics = self.loss.get_metrics()
                metrics["lr"] = self.optimizer.param_groups[0]['lr']
                self.wandb_run.log(metrics)

            self.global_step += 1

        # Average metrics
        num_batches = len(self.train_loader)
        metrics = {k: v/num_batches for k, v in self.loss.get_metrics().items()}

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_bbox = 0.0
        total_loss_giou = 0.0

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['images'].to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in target.items()}
                for target in batch['targets']
            ]

            #TODO: Add masking support (maybe over preprocess fun??)
            predictions = self.model(images)
            losses = self.loss(predictions, targets)

            total_loss += losses['loss'].item()
            total_loss_ce += losses['loss_ce'].item()
            total_loss_bbox += losses['loss_bbox'].item()
            total_loss_giou += losses['loss_giou'].item()

        # Average metrics
        num_batches = len(self.val_loader)
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_loss_ce': total_loss_ce / num_batches,
            'val_loss_bbox': total_loss_bbox / num_batches,
            'val_loss_giou': total_loss_giou / num_batches
        }

        return metrics

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
                val_metrics = self.validate()

                # Log validation metrics
                for key, value in val_metrics.items():
                    # Remove 'val_' prefix if present since we add it in tensorboard path
                    metric_name = key[4:] if key.startswith('val_') else key
                    self.wandb_run.log({f'val/{metric_name}': value})

                print(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")

                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pth')
            else:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}")

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
