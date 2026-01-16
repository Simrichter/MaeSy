"""Trainer for Masked Autoencoder pretraining."""

import torch
import wandb
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict

from .config import MAEPretrainingConfig
from maesy.model.mae_model import MaskedAutoencoderViT


class MaskedAutoencoderPretrainer:
    """Trainer for MAE pretraining."""
    
    def __init__(
        self,
        model: MaskedAutoencoderViT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[MAEPretrainingConfig] = None
    ):
        """
        Initialize MAE pretrainer.
        
        Args:
            model: MAE model to pretrain
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Pretraining configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or MAEPretrainingConfig()
        
        # Setup device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup directories
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb
        self.wandb_run = wandb.init(
            entity="simon-richter-tu-dortmund",
            project="maesy-MAE_Pretraining",
            config=self.config
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
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95)
            )
        elif self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
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
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0

        for batch_idx, batch in enumerate(pbar := tqdm(self.train_loader, desc=f"MAE Epoch {self.current_epoch + 1}")):
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch['images'].to(self.device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
            else:
                images = batch.to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                loss, pred, mask = self.model(images, self.config.mask_ratio)
            
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
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

            if self.global_step % self.config.log_frequency == 0:
                self.wandb_run.log({"train/loss": loss.item(), "train/lr": self.optimizer.param_groups[0]['lr']})
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        assert num_batches > 0, f"Error: division by zero, {self.train_loader} has len 0"
        metrics = {
            'loss': total_loss / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0

        pred = None # Dummy declaration to keep the predictions of the last batch in memory
        for batch in tqdm(self.val_loader, desc="MAE Validation"):
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch['images'].to(self.device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
            else:
                images = batch.to(self.device)
            
            loss, pred, mask = self.model(images, self.config.mask_ratio)
            
            total_loss += loss.item()

        if self.config.output_predicted_images:
            save_path = self.save_dir / "images"
            save_path.mkdir(parents=True, exist_ok=True)
            img = wandb.Image(self.model.unpatchify(pred)[0], caption=f"Predicted Image at step {self.global_step}")
            self.wandb_run.log({"predicted_image": img})
            # save_image(self.model.unpatchify(pred)[0], f"{save_path}/predicted_image{self.global_step}.png")

        # Average metrics
        num_batches = len(self.val_loader)
        metrics = {
            'val_loss': total_loss / num_batches
        }
        
        return metrics
    
    def train(self) -> None:
        """Main training loop."""
        print(f"Starting MAE pretraining for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Mask ratio: {self.config.mask_ratio}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Adjust learning rate for warmup
            if epoch < self.config.warmup_epochs:
                lr_scale = (epoch + 1) / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * lr_scale
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    metric_name = key[4:] if key.startswith('val_') else key
                    self.wandb_run.log({f"val/{metric_name}": value})
                
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('mae_best_model.pth')
            else:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}")
            
            # Step scheduler
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(f'mae_checkpoint_epoch_{epoch + 1}.pth')
        
        # Save final model
        self.save_checkpoint('mae_final_model.pth')
        self.wandb_run.finish()
        print("MAE pretraining completed!")
    
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
