"""Example script for classification pretraining."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from _maesy_core.dataset import MaesyDataset, TrainPatchTransforms, ValPatchTransforms, MultiDataset
from _maesy_core.model import PatchClassificatorConfig, PatchClassificator
from _maesy_core.model.model_tools.model_factory import create_model_from_checkpoint, create_model_from_config
from maesy.training import ClassificationTrainer
from maesy.training.base_trainer import BaseTrainingConfig
from maesy.training.utils import collate_classification_fn


@dataclass
class PatchClassificationTrainingConfig(BaseTrainingConfig):
    """Configuration for Classification pretraining."""
    criterion: str = "ClassificationLoss"

    # Checkpoint and logging
    save_dir: str = "./patch_classification_checkpoints"


def train_patches(dataset_paths, enable_wandb, batch_size=64, num_epochs=50, num_classes=2, patch_shape: Tuple[int, int] = (24, 48)):
    """Main setup function."""

    # Create model config
    model_config = PatchClassificatorConfig(
        resnet_version="resnet18",
        head_in_dim=2304,
        num_classes=num_classes
    )

    model = PatchClassificator(model_config)

    # Data transforms (with augmentations for classification)
    train_transforms = TrainPatchTransforms()
    val_transforms = ValPatchTransforms()

    train_dataset = MultiDataset([MaesyDataset(dataset_path, split="train", annotation_type="classification", transforms=train_transforms) for dataset_path in dataset_paths])
    val_dataset = MultiDataset([MaesyDataset(dataset_path, split="val", annotation_type="classification", transforms=val_transforms) for dataset_path in dataset_paths])

    # train_dataset = MaesyDataset(dataset_path, split="train", annotation_type="classification", transforms=train_transforms)
    # val_dataset = MaesyDataset(dataset_path, split="val", annotation_type="classification", transforms=val_transforms)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_classification_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_classification_fn
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")

    # Create pretraining config
    pretraining_config = BaseTrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        criterion="ClassificationLoss",
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_epochs=5,
        save_dir="./classification_checkpoints",
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        use_amp=True
    )

    # Create pretrainer
    pretrainer = ClassificationTrainer(
        model=model,
        project_name="maesy-Classification_Pretraining",
        train_loader=train_loader,
        val_loader=val_loader,
        config=pretraining_config,
        enable_wandb=enable_wandb
    )

    # Start pretraining
    print("\nStarting classification pretraining...")
    pretrainer.train()

    print("\nClassification pretraining completed!")
    print(f"Best validation loss: {pretrainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {pretraining_config.save_dir}")

def export_patch_classificator(
    model_info: str,
    output_path: str,
    name: str,
    num_classes: int = -1,
    enable_line_detection: bool = False,
    enable_ellipse_detection: bool = False,
    line_class_id: int = -1,
    ellipse_class_id: int = -1,
) -> None:
    """
    Export a trained object detection model to ONNX format for deployment.

    Args:
        :param model_info: String that either specifies a model architecture or the path to a trained checkpoint
        :param output_path: Path to save the exported ONNX model
        # :param detector_arch: Architecture of the model. If None, the architecture will be inferred from the checkpoint. (e.g., "detr" or "rt_detr")
    """

    if model_info.lower() in known_architectures:
        assert num_classes != -1 and (line_class_id != -1 or not enable_line_detection) and (ellipse_class_id != -1 or not enable_ellipse_detection) and output_path != "", f"If using an architecture specifier, additional input is required"
        config = read_yaml(f"cfg/{model_info.lower()}.yaml")
        if config["num_classes"] != -1 and config["num_classes"] != num_classes:
            raise ValueError("num_classes parameter in model config does not match the datasets 'nc' parameter. Leave value in config on '-1' to enable auto-detect.")
        config["num_classes"] = num_classes
        if enable_line_detection:
            if config["line_class_id"] != -1 and config["line_class_id"] != line_class_id:
                raise ValueError(
                    f"line_class_id parameter in model config ({config['line_class_id']}) does not match the datasets line_class_id parameter ({line_class_id}). Leave value in config on '-1' to enable auto-detect.")
            config["line_class_id"] = line_class_id
            config["enable_line_detection"] = True
        else:
            config["enable_line_detection"] = False
        if enable_ellipse_detection:
            if config["ellipse_class_id"] != -1 and config["ellipse_class_id"] != ellipse_class_id:
                raise ValueError(
                    f"ellipse_class_id parameter in model config ({config['ellipse_class_id']}) does not match the datasets ellipse_class_id parameter ({ellipse_class_id}). Leave value in config on '-1' to enable auto-detect.")
            config["ellipse_class_id"] = ellipse_class_id
            config["enable_ellipse_detection"] = True
        else:
            config["enable_ellipse_detection"] = False
        model = create_model_from_config(config)
    elif model_info.endswith(".pth"):
        model = create_model_from_checkpoint(model_info)
    else:
        raise ValueError(f"Model {model_info} is neither in {known_architectures} nor is it a path to a training checkpoint that ends with '.pth'")

    model.eval()

    example_inputs = (torch.randn(1, 3, 224, 224),)
    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
    if onnx_program is None:
        print("FAILED: Model could not be exported.")
        return

    path = Path(model_info).parent if output_path == "" else Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Starting ONNX export...")
    save_name = path / (f"{model.config.type}.onnx" if name == "" else f"{name}.onnx")
    onnx_program.save(save_name)
    print("=" * 60)
    print(f"Success! Model has been exported to {save_name}")
