from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from maesy.model import BaseModel

class CheckpointHandler:
    @staticmethod
    def _assert_compatible_config(checkpoint_config: dict, current_config: dict, part_name: str):
        mismatches = {}
        for key, value in checkpoint_config.items():
            if key in current_config and current_config[key] != value:
                mismatches[key] = {"checkpoint": value, "current": current_config[key]}
        if mismatches:
            raise ValueError(
                f"Failed to load {part_name} due to incompatible configuration values: {mismatches}\n"
                f"Checkpoint {part_name} config: {checkpoint_config}\n"
                f"Current {part_name} config: {current_config}"
            )

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
            'modelconfig': model.config.__dict__,
            # 'trainingconfig': config
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    @staticmethod
    def _check_head_configs(checkpoint, model):
        CheckpointHandler._assert_compatible_config(
            checkpoint_config=checkpoint['headconfig'],
            current_config=model.head.config.__dict__,
            part_name="head",
        )

    @staticmethod
    def _check_bb_configs(checkpoint, model):
        CheckpointHandler._assert_compatible_config(
            checkpoint_config=checkpoint['backboneconfig'],
            current_config=model.backbone.config.__dict__,
            part_name="backbone",
        )

    def _load_head(self, checkpoint, model):
        if model.head.type != checkpoint['headtype']:
            return False
        self._check_head_configs(checkpoint, model)
        # Handle legacy key name for dn_query_content
        if "dn_query_content.weight" in checkpoint['head'].keys():
            checkpoint['head']['dn_query_embedding.weight'] = checkpoint['head'].pop('dn_query_content.weight')
        model.head.load_state_dict(checkpoint['head'])
        return True

    def _load_backbone(self, checkpoint, model):
        if model.backbone.type != checkpoint['backbonetype']:
            print(f"{model.backbone.type} != {checkpoint['backbonetype']}")
            return False
        self._check_bb_configs(checkpoint, model)
        model.backbone.load_state_dict(checkpoint['backbone'])
        return True

    def _autoload_model(self, checkpoint, model):
        bb_success = self._load_backbone(checkpoint, model)
        head_success = self._load_head(checkpoint, model)

        if bb_success and head_success:
            print(f"Model weights loaded successfully")
        elif head_success: # Loaded Head
                print(f"\nWarning: Could only load head weights of type {model.head.type}, NOT backbone!!!!\n")
        elif bb_success: # Loaded Backbone
            print(f"Loaded only backbone weights of type {model.backbone.type}")
        else: # Load failed
            raise ValueError(
                f"Incompatible model architecture. Checkpoint backbone type: {checkpoint['backbonetype']}, actual backbone type: {model.backbone.type}. Checkpoint head type: {checkpoint['headtype']}, actual head type: {model.head.type}.")

    def load_model(self, filepath: str, model=None) -> "BaseModel":
        """
        Load model with weights.
        No optimizer or scheduler states are loaded.
        Args:
            :param filepath: Path to checkpoint file
            :param model: Model to load state dict into. If None, a matching model is instantiated based on the checkpoint config
        Returns:
            The loaded model
        """
        print(f"Loading model from checkpoint {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)

        if model is None:
            from maesy.model_tools.model_factory import create_model_from_config  # import here to prevent circular import error
            model = create_model_from_config(checkpoint['modelconfig'])

        if 'backbone' in checkpoint and 'head' in checkpoint:
            self._autoload_model(checkpoint, model)
        else:
            raise KeyError("'backbone' and 'head' keys not present in checkpoint. Cant load model")

        return model

    def load_backbone(self, filepath: str, model) -> None:
        """
        Load backbone weights into the given model.
        Expects backbone types to match
        Args:
            :param filepath: Path to checkpoint file that contains a backbone
            :param model: Model to load backbone state dict into
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        if not self._load_backbone(checkpoint, model):
            raise KeyError(f"Cant load backbone type {checkpoint['backbonetype']} for model with backbone type {model.backbone.type}")

        print(f"Backbone loaded from {filepath}")

    def load_training_state(self, filepath:str, optimizer=None, scheduler=None) -> Tuple[int, int, float]:
        """
        Load the training state of a preceding training run.
        Loads optimizer and scheduler states as well as current_epoch, global_step and best_val_loss.

        Args:
            :param filepath: Path to checkpoint file
            :param optimizer: Optimizer to load state dict (not loaded if None)
            :param scheduler: Scheduler to load state dict (not loaded if None)

        Returns:
            current_epoch: The epoch to continue training from
            global_step: The global step to continue training from
            best_val_loss: The best validation loss achieved in the preceding training run
        """
        checkpoint = torch.load(filepath, map_location=self.device)
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
        return current_epoch, global_step, best_val_loss