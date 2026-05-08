import shutil
from dataclasses import dataclass
from typing import List, Any

import torch
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path

from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from maesy.evaluation.inferer import Inferer
# Import models
from maesy.model_tools import replace_bn_with_frozenbn, CheckpointHandler, read_yaml
from maesy.model_tools.model_factory import create_model_from_config, known_architectures, create_model_from_checkpoint

# Import training components
from maesy.training import DetectionTrainer, TrainingConfig
from maesy.training.utils import collate_detection_fn

# Import dataset
from maesy.dataset import UnlabeledDataset, MaesyDataset # TODO Move unlabeled dataset to MaesyDataset(Unlabeled=True)


@dataclass
class ODTrainingConfig(TrainingConfig):
    """Configuration for training."""

    # Training parameters
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    backbone_learning_rate: float = learning_rate
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    criterion: str = "None"  # e.g., DetectionLoss, MaskedMSE, etc.

    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd
    momentum: float = 0.9  # For SGD

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, step, multistep
    lr_step_size: int = 30  # For step scheduler
    lr_gamma: float = 0.1  # For step scheduler

    # Loss coefficients
    bbox_loss_coef: float = 5.0,
    class_loss_coef: float = 1.0,
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1
    aux_loss_coef: float = 0.5
    enc_loss_coef: float = 0.3
    line_loss_coef: float = 2.0
    ellipse_loss_coef: float = 2.0
    dn_loss_coef: float = 1.0

    # Checkpoint and logging
    save_dir: str = "./checkpoints"
    save_frequency: int = 10  # Save every n epochs
    log_frequency: int = 10  # Log every n global steps

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    num_workers: int = 4

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Mixed precision training
    use_amp: bool = True

def train_vit_detector(
    model_info: str,
    dataset_path: str,
    output_dir: str,
    freeze: bool,
    continue_training_from_checkpoint: bool,
    pretrained_backbone: str,
    enable_wandb: bool,
    enable_denoising: bool = False,
    denoising_num_queries: int = 0,
    denoising_label_noise_ratio: float = 0.2,
    denoising_box_noise_scale: float = 0.4,
    enable_line_detection: bool = True,
    enable_ellipse_detection: bool = True,
    seed: int = 42
):
    """
    Train an object detection model with the selected detector architecture.

    Args:
        :param model_info: String that specifies the model configuration to be used. Either a path to a checkpoint, or a model architecture like "rt-detr"
        :param dataset_path: Path to object detection dataset
        :param output_dir: Directory to save checkpoints
        :param freeze: Whether to freeze the backbone
        :param continue_training_from_checkpoint: Whether to continue training from an existing OD checkpoint (in that case, checkpoint_path should point to an OD checkpoint instead of a MAE checkpoint)
        :param pretrained_backbone: Path to a checkpoint that contains a backbone to be reused in od training
        :param enable_wandb: Whether to enable Weights & Biases logging
        :param seed: Random seed for reproducibility (default: 42)
        :param enable_line_detection: Whether to enable line detection (if the dataset contains line annotations and the model supports it)
        :param enable_ellipse_detection: Whether to enable ellipse detection (if the dataset contains ellipse annotations and the model supports it)
    """
    print("=" * 60)
    print("Starting object detection training")
    print("=" * 60)

    torch.manual_seed(seed)

    train_transform_steps: List[Any] = [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    if not (enable_line_detection or enable_ellipse_detection):
        train_transform_steps.append(transforms.RandomAffine(degrees=8, translate=(0.15, 0.15), scale=(0.95, 1.05)))
        # TODO: Affine should be possible with lines as well?
    train_transform_steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    train_transforms = transforms.Compose(train_transform_steps)
    val_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # None activates auto-infer, -1 deactivates class in MaesyDataset
    special_class_deactivation_dict = {'line_class_id': None if enable_line_detection else -1,
                                       'ellipse_class_id': None if enable_ellipse_detection else -1}

    train_dataset = MaesyDataset(dataset_path, "train", "detection", train_transforms, enable_lines=enable_line_detection, enable_ellipses=enable_ellipse_detection)
    val_dataset = MaesyDataset(dataset_path, "val", "detection", val_transforms, enable_lines=enable_line_detection, enable_ellipses=enable_ellipse_detection)
    assert train_dataset.get_special_classes() == val_dataset.get_special_classes(), "Error, train and val datasets must have same special classes!"

    # Create training configuration
    training_config = ODTrainingConfig(
        batch_size=64,
        num_epochs=750,
        learning_rate=1e-4 if freeze else 1e-4,  # Higher LR when only training head
        backbone_learning_rate=0.0 if freeze else 1e-5,  # Lower LR for backbone if fine-tuning, otherwise 0
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler="cosine",
        warmup_epochs=4,
        label_smoothing=0.1,
        save_frequency=100,
        log_frequency=50,
        save_dir=output_dir,
        criterion="DetectionLoss",  # "YOLOv8Loss", #
        use_amp=True,
        bbox_loss_coef = 5.0,
        class_loss_coef = 1.0,
        giou_loss_coef = 2.0,
        eos_coef = 0.15,
        aux_loss_coef = 0.5,
        enc_loss_coef = 0.3,
        line_loss_coef = 2.0,
        ellipse_loss_coef = 2.0,
        dn_loss_coef = 1.0,
    )

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        persistent_workers=True,
        collate_fn=collate_detection_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        persistent_workers=True,
        collate_fn=collate_detection_fn,
        shuffle=True,
        pin_memory=True
    )

    if model_info.lower() in known_architectures:
        config = read_yaml(f"cfg/{model_info.lower()}.yaml")
        if config["num_classes"] != -1 and config["num_classes"] != train_dataset.get_num_classes():
            raise ValueError("num_classes parameter in model config does not match the datasets 'nc' parameter. Leave value in config on '-1' to enable auto-detect.")
        config["num_classes"] = train_dataset.get_num_classes()
        if enable_line_detection:
            if config["line_class_id"] != -1 and config["line_class_id"] != train_dataset.get_special_classes()["line_class_id"]:
                raise ValueError(
                    f"line_class_id parameter in model config ({config['line_class_id']}) does not match the datasets line_class_id parameter ({train_dataset.get_special_classes()['line_class_id']}). Leave value in config on '-1' to enable auto-detect.")
            config["line_class_id"] = train_dataset.get_special_classes()["line_class_id"]
            config["enable_line_detection"] = True
        else:
            config["enable_line_detection"] = False
        if enable_ellipse_detection:
            if config["ellipse_class_id"] != -1 and config["ellipse_class_id"] != train_dataset.get_special_classes()["ellipse_class_id"]:
                raise ValueError(
                    f"ellipse_class_id parameter in model config ({config['ellipse_class_id']}) does not match the datasets ellipse_class_id parameter ({train_dataset.get_special_classes()['ellipse_class_id']}). Leave value in config on '-1' to enable auto-detect.")
            config["ellipse_class_id"] = train_dataset.get_special_classes()["ellipse_class_id"]
            config["enable_ellipse_detection"] = True
        else:
            config["enable_ellipse_detection"] = False
        model = create_model_from_config(config)
    elif not model_info.endswith(".pth"):
        raise ValueError(f"Model {model_info} is neither in {known_architectures} nor is it a path to a training checkpoint (must end with '.pth')")
    else:
        model = create_model_from_checkpoint(model_info)

    if pretrained_backbone is not "":
        print(f"Loading pretrained backbone weights from {pretrained_backbone}...")
        CheckpointHandler(device=training_config.device).load_backbone(pretrained_backbone, model)

    if not continue_training_from_checkpoint and (model.config.num_classes != train_dataset.get_num_classes() or (enable_line_detection and model.config.line_class_id != train_dataset.get_special_classes()["line_class_id"]) or (enable_ellipse_detection and model.config.ellipse_class_id != train_dataset.get_special_classes()["ellipse_class_id"])):
        print(f"!!!!!!!!!!!!!!!!!!!!\nDetected different classification setup:\nModel has {model.config.num_classes} classes, dataset provides {train_dataset.get_num_classes()}\nLine class of model is {model.config.line_class_id} and Ellipse class is {model.config.ellipse_class_id}, dataset provides {train_dataset.get_special_classes()}\nCreating a new classification head with {train_dataset.get_num_classes()} classes and matching special classes.\n!!!!!!!!!!!!!!!!!!!!")
        if not hasattr(model.head, "create_class_heads"):
            raise AttributeError(f"\nmodel.head has no 'create_class_heads()' method that could be used to create a new classification head. Found content: {[f for f in dir(model.head) if not f.startswith('_') and callable(getattr(model.head, f))]}")
        model.update_head_conf(train_dataset.get_num_classes(), train_dataset.get_special_classes())


    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        project_name="maesy-object-detection",
        enable_wandb=enable_wandb
    )

    replace_bn_with_frozenbn(trainer.model.backbone)

    # Train
    if continue_training_from_checkpoint:
        assert model.config.num_classes == train_dataset.get_num_classes(), "Error: The number of classes in the model config does not match the number of classes in the dataset. Please make sure they match before continuing training."
        if enable_line_detection:
            assert model.config.line_class_id == train_dataset.get_special_classes()["line_class_id"], f"Error: The line_class_id in the model config ({model.config.line_class_id}) does not match the line_class_id in the dataset ({train_dataset.get_special_classes()['line_class_id']}). Please make sure they match before continuing training."
        if enable_line_detection:
            assert model.config.ellipse_class_id == train_dataset.get_special_classes()["ellipse_class_id"], f"Error: The ellipse_class_id in the model config ({model.config.ellipse_class_id}) does not match the ellipse_class_id in the dataset ({train_dataset.get_special_classes()['ellipse_class_id']}). Please make sure they match before continuing training."
        trainer.resume(model_info)
    else:
        trainer.train()

def infer_vit_detector(
    checkpoint_path: str,
    dataset_path: str,
    out_path: str,
    visualize: bool,
    device: torch.device,
    # detector_arch: str | None = None,
) -> None:
    """
    Run inference with a trained object detection model.

    Args:
        :param checkpoint_path: Path to trained model checkpoint
        :param dataset_path: Path to input images for inference
        :param out_path: Path to save inference results (predicted bounding boxes and labels)
        :param visualize: Whether to save visualizations of predictions (e.g., images with predicted boxes drawn)
        :param device: Device to run inference on (e.g., "cuda" or "cpu")
        # :param detector_arch: The architecture to be used (e.g., "detr" or "rt_detr")
    """
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    # Load model
    model = create_model_from_checkpoint(checkpoint_path) #CheckpointHandler(device=device).load_model(checkpoint_path)
    model.eval()

    # dataset = UnlabeledDataset(Path(images_path), transforms=transforms.Compose([
    #     transforms.ToImage(),
    #     transforms.ToDtype(torch.float32, scale=True),
    #     transforms.Resize((224, 224)),
    #     # transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])) #, use_first_n=30 # , step=50
    dataset = MaesyDataset(dataset_path, "val", "None", transforms=transforms.Compose([ # TODO: make blank image folder possible again, "auto-infer" split? Maybe through 'None' -> All splits
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    inferer = Inferer(model=model, data_loader=dataloader, device=device)
    preds, _ = inferer.infer() # List[Dict] with keys "pred_boxes" (B X num_querys X 4) and "pred_logits" (B X num_queries)

    print("=" * 60)
    print(f"Saving inference results to {out_path}...")
    print("=" * 60)

    # torch.cat(preds, dim=0) # Total number of images X num_queries X (4 or num_classes)

    images_dir = dataset.images_dir
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    save_all_predictions = False  # Intentionally keep all query outputs for debugging.
    for p in tqdm(zip(dataset.images, preds)):
        img_path = images_dir/p[0]
        shutil.copy(img_path, out_path/p[0])
        with open(out_path/Path(p[0]).with_suffix(".txt"), "w") as f:
            boxes = torch.unbind(p[1]["pred_boxes"].squeeze(0), dim=0)
            labels = torch.unbind(p[1]["pred_logits"].squeeze(0), dim=0)
            for box, label in zip(boxes, labels):
                cx, cy, w, h = box
                score, l = label.max(-1)
                if save_all_predictions or (l < 3 and score >= 0.3):
                    f.write(f"{l.item()} {cx.item()} {cy.item()} {w.item()} {h.item()}\n")
    if visualize:
        from maesy.evaluation import visualize_data
        visualize_data(out_path, "")

def export_vit_detector(
    checkpoint_path: str,
    output_path: str,
) -> None:
    """
    Export a trained object detection model to ONNX format for deployment.

    Args:
        :param checkpoint_path: Path to trained model checkpoint
        :param output_path: Path to save the exported ONNX model
        # :param detector_arch: Architecture of the model. If None, the architecture will be inferred from the checkpoint. (e.g., "detr" or "rt_detr")
    """

    model = create_model_from_checkpoint(checkpoint_path)
    model.eval()

    example_inputs = (torch.randn(1, 3, 224, 224),)
    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)

    path = Path(checkpoint_path).parent if output_path == "" else Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Starting ONNX export...")
    save_name = path / f"{model.config.type}.onnx"
    onnx_program.save(save_name)
    print("=" * 60)
    print(f"Success! Model has been exported to {save_name}")


if __name__ == "__main__":
    #argparse:
    # import argparse
    # parser = argparse.ArgumentParser(description="Train a ViTDetector for object detection")
    # parser.add_argument("--checkpoint", type=str, default="", help="Path to pretrained MAE checkpoint (or OD checkpoint if --continue_from_checkpoint is set)")
    # parser.add_argument("--dataset", type=str, default="/home/simon/Desktop/maesy-training/data/AllData (ObjectDetection)", help="Path to object detection dataset")
    # parser.add_argument("--output", type=str, default="./od_checkpoints", help="Directory to save checkpoints")
    # parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    #
    # args = parser.parse_args
    checkpoint = "/home/simon/Desktop/maesy-training/od_checkpoints/super-moon-127/final_model.pth" # ""
    dataset = "/home/simon/Desktop/maesy-training/data/CvatLE" # r"/home/simon/Desktop/maesy-training/data/"
    output = r"/home/simon/Desktop/maesy-training/od_checkpoints"
    resume = checkpoint != ""
    train_vit_detector(checkpoint, dataset, output, True, continue_training_from_checkpoint=False, enable_wandb=False, enable_line_detection=True, enable_ellipse_detection=True)

    # checkpoint = r"/home/simon/Desktop/maesy-training/od_checkpoints/leafy-music-38/best_model.pth"
    # images = r"/home/simon/Desktop/maesy-training/data/AllData (ObjectDetection)/train/images"
    # out = r"/home/simon/Desktop/maesy-training/inference_results"
    # infer_vit_detector(checkpoint, images, out, True, torch.device("cuda"))