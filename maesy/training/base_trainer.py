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

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        self.enable_wandb = enable_wandb
        if self.enable_wandb:
            # Setup wandb
            self.wandb_run = wandb.init(
                entity="simon-richter-tu-dortmund",
                project=project_name,
                config=asdict(self.config)
            )

        # Setup device
        self.device = self.config.device
        if not torch.cuda.is_available() and self.device != torch.device("cpu"):
            print(f"Warning: CUDA is not available, switching to CPU!")
            self.device = torch.device("cpu")

        # Setup directories
        self.save_dir = Path(self.config.save_dir) / (self.wandb_run.name if self.enable_wandb else "offline_run")
        self.checkpoint_handler = CheckpointHandler(self.device, self.save_dir)

        # Mixed precision training
        self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp else None

        if issubclass(type(model), BaseModel):
            self.model = model
        else:
            raise ValueError(f"Unknown model type '{type(model)}'")

        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        self.loss: BaseLoss = self._create_loss()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')



    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer.lower() == "adamw":
            params = [{"params": self.model.backbone.parameters(), "lr": self.config.backbone_learning_rate},
                      {"params": self.model.head.parameters(), "lr": self.config.learning_rate}]
            return torch.optim.AdamW(
                params, #self.model.parameters()
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
                T_max=self.config.num_epochs - self.config.warmup_epochs,
                eta_min=1e-6 # TODO: Make this configurable?
            )
            constant = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=1000000)
            return torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup, cosine, constant],  milestones=[self.config.warmup_epochs, self.config.num_epochs])
        elif self.config.lr_scheduler.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        else:
            print(f"Warning: Unknown scheduler '{self.config.lr_scheduler}'")
            return None

    def _create_loss(self) -> BaseLoss:
        if self.config.criterion == "DetectionLoss":

            loss = DetectionLoss(
                num_classes=getattr(self.model.config, "num_classes", 1),
                bbox_loss_coef=getattr(self.config, "bbox_loss_coef", 5.0),
                class_loss_coef=getattr(self.config, "class_loss_coef", 1.0),
                giou_loss_coef=getattr(self.config, "giou_loss_coef", 2.0),
                label_smoothing=getattr(self.config, "label_smoothing", 0.0),
                aux_loss_coef=getattr(self.config, "aux_loss_coef", 0.5),
                line_loss_coef=getattr(self.config, "line_loss_coef", 2.0),
                dn_loss_coef=getattr(self.config, "dn_loss_coef", 1.0),
                enable_line_detection=getattr(self.model.config, "enable_line_detection", False),
                enable_ellipse_detection=getattr(self.model.config, "enable_ellipse_detection", False),
                line_class_id=getattr(self.model.config, "line_class_id", -1),
                ellipse_class_id=getattr(self.model.config, "ellipse_class_id", -1),
                eos_coef=getattr(self.config, "eos_coef", 0.1),
                device=self.device
            )
            return loss
        elif self.config.criterion == "MaskedMSE":
            return MaskedMSE()
        elif self.config.criterion == "YOLOv8Loss":
            from .losses import YOLOv8Loss
            return YOLOv8Loss(
                num_classes=self.model.config.num_classes,
                device=self.device
            )
        elif self.config.criterion == "ClassificationLoss":
            from .losses import ClassificationLoss
            return ClassificationLoss()
        else:
            raise ValueError(f"Warning: Unknown loss '{self.config.criterion}'")

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
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr_bbone': self.optimizer.param_groups[0]['lr'],
                'lr_head': self.optimizer.param_groups[-1]['lr']
            })

            # Log to wandb
            if self.global_step % self.config.log_frequency == 0:
                data = {f"train/{k}": v.item() for k, v in losses.items() if not k.startswith('img_')}
                data['train/lr'] = self.optimizer.param_groups[-1]['lr']
                data['train/lr_bbone'] = self.optimizer.param_groups[0]['lr']
                data['train/gradient_norm'] = total_norm.item()
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
        # self.model.train()
        validation_start_hook = getattr(self, "_validation_start", None)
        if callable(validation_start_hook):
            validation_start_hook()

        losses: dict[str, Any] = {}
        for batch in tqdm(self.val_loader, desc="Validation"):
            images, targets = handle_raw_batch(batch, self.device)

            losses = self.forward_model(images, targets, val=True)
            validation_step_hook = getattr(self, "_validation_step", None)
            if callable(validation_step_hook):
                validation_step_hook(images, targets, losses)


        save_path = self.save_dir / "images"
        save_path.mkdir(parents=True, exist_ok=True)
        for name, img in losses.items():
            if name.startswith('img_'):
                save_image(img, f"{save_path}/predicted_image{self.global_step}_{name}.png")
        metrics = self.loss.get_metrics()
        validation_finalize_hook = getattr(self, "_validation_finalize", None)
        if callable(validation_finalize_hook):
            metrics.update(validation_finalize_hook())

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

        # torch.autograd.set_detect_anomaly(True) # TODO: Make opt-in (as this slows down training massively)

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

                val_msg = (f"Epoch {self.current_epoch + 1}/{self.config.num_epochs} - "
                           f"Train Loss: {train_metrics['total_loss']:.4f}, "
                           f"Val Loss: {val_metrics['val/total_loss']:.4f}")
                if 'val/mAP50' in val_metrics and 'val/mAP50_95' in val_metrics:
                    val_msg += (f", mAP50: {val_metrics['val/mAP50']:.4f}, "
                                f"mAP50-95: {val_metrics['val/mAP50_95']:.4f}")
                if 'val/precision50' in val_metrics and 'val/recall50' in val_metrics:
                    val_msg += (f", P50: {val_metrics['val/precision50']:.4f}, "
                                f"R50: {val_metrics['val/recall50']:.4f}")
                print(val_msg)

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

    def resume(self, checkpoint_path: str):
        self.current_epoch, self.global_step, self.best_val_loss = self.checkpoint_handler.load_training_state(checkpoint_path, self.optimizer, self.scheduler)
        self.current_epoch += 1  # To continue with the next epoch and not repeat an already trained one
        self.train()
