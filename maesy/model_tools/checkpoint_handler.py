from pathlib import Path
from typing import Any, Optional

import torch

class CheckpointHandler:

    def __init__(self, device: torch.device, save_dir: Optional[str | Path]=None):
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def save_checkpoint(self, current_epoch, global_step, model, optimizer, best_val_loss, config, filename: str, scheduler = None) -> None:
        """
        Save model checkpoint.

        Args:
            :param current_epoch: Current epoch number
            :param global_step: Current global step number
            :param model: Model to save state dict from
            :param optimizer: Optimizer to save state dict from
            :param best_val_loss: Best validation loss so far
            :param config: Training configuration to save with checkpoint
            :param filename: Name of the checkpoint file (e.g., "latest_model.pth")
            :param scheduler: Learning rate scheduler to save state dict from (optional)
        """
        checkpoint = {
            'epoch': current_epoch,
            'global_step': global_step,
            'backbone': model.backbone.state_dict(),
            'backbonetype': model.backbone.type,
            'backboneconfig': model.backbone.config.__dict__,
            'head': model.head.state_dict(),
            'headtype': model.head.type,
            'headconfig': model.head.config.__dict__,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            # 'config': config
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    @staticmethod
    def _legacy_load_model(checkpoint, model) :
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Failed to load legacy checkpoint: 'model_state_dict' key not found")

    @staticmethod
    def _check_head_configs(checkpoint, model):
        if checkpoint['headconfig'] != model.head.config.__dict__:
            raise ValueError(
                f"Failed to load head due to incompatible configuration.\n \n Checkpoint head config:\n{checkpoint['headconfig']}\n \n actual head config:\n{model.head.config.__dict__}")

    @staticmethod
    def _check_bb_configs(checkpoint, model):
        if checkpoint['backboneconfig'] != model.backbone.config.__dict__:
            raise ValueError(
                f"Failed to load backbone due to incompatible configuration. Checkpoint backbone config: {checkpoint['backboneconfig']}, actual backbone config: {model.backbone.config.__dict__}")

    def _load_model(self, checkpoint, model):
        if (model.backbone.type != checkpoint['backbonetype']) and (model.head.type != checkpoint['headtype']):
            raise ValueError(
                f"Incompatible model architecture. Checkpoint backbone type: {checkpoint['backbonetype']}, actual backbone type: {model.backbone.type}. Checkpoint head type: {checkpoint['headtype']}, actual head type: {model.head.type}.")
        if (model.backbone.type != checkpoint['backbonetype']) or (model.head.type != checkpoint['headtype']):
            if model.backbone.type != checkpoint['backbonetype']:
                print(f"Warning: Loading only head of type {model.head.type}, NOT backbone!")
                self._check_head_configs(checkpoint, model)
                model.head.load_state_dict(checkpoint['head'])
            else:
                print(f"Loading only backbone of type {model.backbone.type}")
                self._check_bb_configs(checkpoint, model)
                model.backbone.load_state_dict(checkpoint['backbone'])
        else:
            self._check_head_configs(checkpoint, model)
            self._check_bb_configs(checkpoint, model)
            model.backbone.load_state_dict(checkpoint['backbone'])
            model.head.load_state_dict(checkpoint['head'])

    def load_checkpoint(self, filepath: str, model, optimizer=None, scheduler=None) -> tuple[Any, Any, Any]:
        """
        Load model checkpoint.
        Args:
            :param filepath: Path to checkpoint file
            :param model: Model to load state dict into
            :param optimizer: Optimizer to load state dict into
            :param scheduler: Learning rate scheduler to load state dict into (optional)
        Returns:
            Tuple of (current_epoch, global_step, best_val_loss)
        """
        print(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)

        if 'backbone' in checkpoint and 'head' in checkpoint:
            self._load_model(checkpoint, model)
        else:
            print("Legacy checkpoint format detected. Attempting to load...")
            self._legacy_load_model(checkpoint, model)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            best_val_loss = checkpoint['best_val_loss']
        else:
            # print(f"Loaded only Model, training starts from epoch 0")
            current_epoch = 0
            global_step = 0
            best_val_loss = float('inf')

        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # print(f"Checkpoint loaded from {filepath}")
        return current_epoch, global_step, best_val_loss