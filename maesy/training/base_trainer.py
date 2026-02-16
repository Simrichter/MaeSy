from dataclasses import asdict
from typing import Protocol
from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any

from wandb import Image

from .config import TrainingConfig
from .losses import DetectionLoss, MaskedMSE, BaseLoss
from ..model import ModelConfig, BaseModel
from .utils import handle_raw_batch
from ..model_tools.checkpoint_handler import CheckpointHandler


class BaseTrainer(ABC):

    def __init__(
            self,
            model: BaseModel,
            train_loader: DataLoader,
            project_name: str,
            val_loader: Optional[DataLoader] = None,
            config: Optional[TrainingConfig] = None,
            enable_wandb: bool = True
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
        self.enable_wandb = enable_wandb

        if self.enable_wandb:
            # Setup wandb
            self.wandb_run = wandb.init(
                entity="simon-richter-tu-dortmund",
                project=project_name,
                config=asdict(self.config)
            )

        # Setup directories
        self.save_dir = Path(self.config.save_dir)/(self.wandb_run.name if self.enable_wandb else "offline_run")
        self.checkpoint_handler = CheckpointHandler(self.device, self.save_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Mixed precision training
        self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp else None



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
            warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(
                (step + 1) / self.config.warmup_epochs, 1.0))
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs - self.config.warmup_epochs
            )
            return torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup, cosine],  milestones=[self.config.warmup_epochs])
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

    def forward_model(self, images: torch.Tensor, targets: Optional[torch.Tensor], val: bool) -> Dict[str, torch.Tensor]:
        """
        Manages the forward pass through the model.
        Can be overwritten to add model-specific preprocessing
        """
        return self.loss(self.model(images), targets)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.loss.reset_metrics()

        for batch_idx, batch in enumerate(pbar := tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")):
            images, targets = handle_raw_batch(batch, self.device)

            # Forward pass
            with torch.amp.autocast("cuda", enabled=self.config.use_amp):
                losses = self.forward_model(images, targets, val=False)
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
                data = {f"train/{k}": v.item() for k, v in losses.items() if not k.startswith('img_')}
                data['train/lr'] = self.optimizer.param_groups[0]['lr']
                if self.enable_wandb: self.wandb_run.log(data=data, step=self.global_step, commit=True)

            self.global_step += 1

        return self.loss.get_metrics()

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.loss.reset_metrics()
        self.model.eval()
        losses: dict[str, Any] = {}
        for batch in tqdm(self.val_loader, desc="Validation"):
            images, targets = handle_raw_batch(batch, self.device)

            losses = self.forward_model(images, targets, val=True)


        save_path = self.save_dir / "images"
        save_path.mkdir(parents=True, exist_ok=True)
        for name, img in losses.items():
            if name.startswith('img_'):
                save_image(img, f"{save_path}/predicted_image{self.global_step}_{name}.png")
        metrics = self.loss.get_metrics()
        if self.enable_wandb:
            imgs_to_log: dict[str, Image] = {k: wandb.Image(v * 255) for k, v in losses.items() if k.startswith('img_')}
            metrics.update(imgs_to_log)

        return metrics

    def train(self) -> None:
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Validate
            if self.val_loader is not None:
                val_metrics = {f"val/{k}": v for k, v in self.validate().items()}  # Preparations for logging
                if self.enable_wandb: self.wandb_run.log(data=val_metrics, step=self.global_step)

                print(f"Epoch {self.current_epoch + 1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val/total_loss']:.4f}")

                # Save best model
                if val_metrics['val/total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val/total_loss']
                    self.checkpoint_handler.save_checkpoint(self.current_epoch, self.global_step, self.model, self.optimizer, self.best_val_loss, self.config,  'best_model.pth', self.scheduler)
            else:
                print(f"Epoch {self.current_epoch + 1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}")

            # Save most recent epoch
            self.checkpoint_handler.save_checkpoint(self.current_epoch, self.global_step, self.model, self.optimizer, self.best_val_loss, self.config, 'latest_model.pth', self.scheduler)

            # Save checkpoint periodically
            if (self.current_epoch + 1) % self.config.save_frequency == 0:
                self.checkpoint_handler.save_checkpoint(self.current_epoch, self.global_step, self.model, self.optimizer, self.best_val_loss, self.config, f'checkpoint_epoch_{self.current_epoch + 1}.pth', self.scheduler)

        # Save final model
        self.checkpoint_handler.save_checkpoint(self.current_epoch, self.global_step, self.model, self.optimizer,
                                                self.best_val_loss, self.config,
                                                'final_model.pth', self.scheduler)
        if self. enable_wandb: self.wandb_run.finish()
        print("Training completed!")


    def load_checkpoint(self, filepath: str, model_only=False) -> None:
        """
        Load training checkpoint.
        Args:
            :param filepath: Path to checkpoint file
            :param model_only: If True, only load model weights (ignore optimizer, scheduler, epoch, etc.)
        """
        if model_only:
            _, _, _ = self.checkpoint_handler.load_checkpoint(filepath, self.model)
        else:
            self.current_epoch, self.global_step, self.best_val_loss = self.checkpoint_handler.load_checkpoint(filepath, self.model, self.optimizer, self.scheduler)
