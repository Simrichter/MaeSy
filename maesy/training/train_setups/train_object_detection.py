import shutil
from dataclasses import dataclass
from typing import Optional

# import imgaug
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from maesy.evaluation.inferer import Inferer
# Import models
from maesy.model import DETR, DETRConfig, RTDETR, RTDETRConfig
from maesy.model_tools import replace_bn_with_frozenbn, CheckpointHandler, read_yaml
from maesy.model_tools.model_factory import create_model_from_config, known_architectures, create_model_from_checkpoint

# Import training components
from maesy.training import DetectionTrainer, TrainingConfig
from maesy.training.utils import collate_detection_fn

# Import dataset
from maesy.dataset import ObjectDetectionDataset, UnlabeledDataset, MaesyDataset


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

    # Checkpoint and logging
    save_dir: str = "./checkpoints"
    save_frequency: int = 10  # Save every n epochs
    log_frequency: int = 10  # Log every n global steps

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # num_workers: int = 4

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Early stopping
    # early_stopping_patience: Optional[int] = None TODO

    # Mixed precision training
    use_amp: bool = True

# DETR_CONFIG = DETRConfig(
#
#         image_size=224,
#
#         # Backbone parameters
#         embed_dim=256,
#         resnet_version="resnet50",
#         # freeze_backbone=True,
#
#         # Detection head parameters
#         num_classes=3,
#         num_queries=30, # 100
#         num_encoder_layers=4,
#         num_decoder_layers=4,
#         encoder_num_heads=4,
#         decoder_num_heads=4,
#         hidden_dim_out_layers=512,
#
#         # Loss weights
#         bbox_loss_coef=5, #1.0,
#         class_loss_coef=2.0,#2.0,
#         giou_loss_coef=2.0, #1.0,
#         eos_coef=0.05,
#         aux_loss_coef=0.5,
#     )

# RT_DETR_CONFIG = RTDETRConfig(
#     image_size=224,
#     resnet_version="resnet18", #"resnet50",
#     num_classes=11,
#     num_queries=40,
#     embed_dim=128, # 256
#     num_decoder_layers=2, # 4
#     decoder_num_heads=8,
#     hidden_dim_out_layers=256, # 512
#     enable_line_detection=True,
#     bbox_loss_coef=5.0,
#     line_loss_coef=5.0,
#     class_loss_coef=2.0,
#     giou_loss_coef=2.0,
#     eos_coef=0.05,
#     aux_loss_coef=0.5,
# )

def train_vit_detector(
    # checkpoint_path: str,
    model_info: str,
    dataset_path: str,
    output_dir: str,
    freeze: bool,
    continue_from_checkpoint: bool,
    enable_wandb: bool,
    detector_arch: str = "rt_detr",
    enable_denoising: bool = False,
    denoising_num_queries: int = 0,
    denoising_label_noise_ratio: float = 0.2,
    denoising_box_noise_scale: float = 0.4,
    enable_line_detection: bool = True,
    seed: int = 42
):
    """
    Train an object detection model with the selected detector architecture.

    Args:
        # :param checkpoint_path: Path to a checkpoint (pretrained or full OD checkpoint)
        :param model: String that specifies the model configuration to be used. Either a path to a checkpoint, or a model architecture like "rt-detr"
        :param dataset_path: Path to object detection dataset
        :param output_dir: Directory to save checkpoints
        :param freeze: Whether to freeze the backbone
        :param continue_from_checkpoint: Whether to continue training from an existing OD checkpoint (in that case, checkpoint_path should point to an OD checkpoint instead of a MAE checkpoint)
        :param enable_wandb: Whether to enable Weights & Biases logging
        :param seed: Random seed for reproducibility (default: 42)
    """
    print("=" * 60)
    print("Starting object detection training")
    print("=" * 60)

    torch.manual_seed(seed)

    train_transform_steps = [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    if not enable_line_detection:
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

    train_dataset = MaesyDataset(dataset_path, "train", "detection", train_transforms)
    val_dataset = MaesyDataset(dataset_path, "val", "detection", val_transforms)
    assert train_dataset.get_special_classes() == val_dataset.get_special_classes(), "Error, train and val datasets must have same special classes!"

    # Create training configuration
    training_config = ODTrainingConfig(
        batch_size=64,
        num_epochs=500,
        learning_rate=1e-4 if freeze else 1e-4,  # Higher LR when only training head
        backbone_learning_rate=0.0 if freeze else 1e-5,  # Lower LR for backbone if fine-tuning, otherwise 0
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler="cosine",
        warmup_epochs=2,
        save_frequency=100,
        log_frequency=50,
        save_dir=output_dir,
        criterion="DetectionLoss",  # "YOLOv8Loss", #
        use_amp=True,
        # device=torch.device("cpu")
    )

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        collate_fn=collate_detection_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_detection_fn,
        shuffle=True,
        pin_memory=True
    )

    if model_info.lower() in known_architectures:
        config = read_yaml(f"cfg/{model_info.lower()}.yaml")
        config["line_class_id"] = train_dataset.get_special_classes()["line_class_id"]
        model = create_model_from_config(config)
    elif not model_info.endswith(".pth"):
        raise ValueError(f"Model {model_info} is neither in {known_architectures} nor is it a path to a training checkpoint (must end with '.pth')")
    else:
        model = create_model_from_checkpoint(model_info)
        assert model.config.line_class_id == train_dataset.get_special_classes()["line_class_id"], "Error: The line_class_id in the model config does not match the line_class_id in the dataset. Please make sure they match before continuing training."

    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        project_name="maesy-object-detection",
        enable_wandb=enable_wandb
    )

    # if checkpoint_path != "":
    #     # Transfer backbone weights
    #     trainer.load_checkpoint(checkpoint_path, model_only=not continue_from_checkpoint)

    replace_bn_with_frozenbn(trainer.model.backbone)

    # Train
    if continue_from_checkpoint:
        trainer.resume(model_info)
    else:
        trainer.train()

def infer_vit_detector(
    checkpoint_path: str,
    images_path: str,
    out_path: str,
    visualize: bool,
    device: torch.device,
    detector_arch: str | None = None,
) -> None:
    """
    Run inference with a trained object detection model.

    Args:
        :param checkpoint_path: Path to trained model checkpoint
        :param images_path: Path to input image for inference
        :param out_path: Path to save inference results (predicted bounding boxes and labels)
        :param visualize: Whether to save visualizations of predictions (e.g., images with predicted boxes drawn)
        :param device: Device to run inference on (e.g., "cuda" or "cpu")
        :param detector_arch: The architecture to be used (e.g., "detr" or "rt_detr")
    """
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    # Load model
    model = CheckpointHandler(device=device).load_model(checkpoint_path)
    model.eval()

    dataset = UnlabeledDataset(Path(images_path), transforms=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])) #, use_first_n=30 # , step=50
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
        from maesy.evaluation import visualize_annotations
        visualize_annotations(out_path, "")

def export_vit_detector(
    checkpoint_path: str,
    output_path: str,
) -> None:
    """
    Export a trained object detection model to ONNX format for deployment.

    Args:
        :param checkpoint_path: Path to trained model checkpoint
        :param output_path: Path to save the exported ONNX model
        :param detector_arch: Architecture of the model. If None, the architecture will be inferred from the checkpoint. (e.g., "detr" or "rt_detr")
    """
    device = torch.device("cpu") # For exporting no GPU is required

    model = CheckpointHandler(device=device).load_model(checkpoint_path)
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
    checkpoint = "" # "/home/simon/Desktop/maesy-training/od_checkpoints/scarlet-plant-57/latest_model.pth"
    dataset = "/home/simon/Desktop/maesy-training/data/AllData (ObjectDetection)" #r"/home/simon/Desktop/maesy-training/data/"
    output = r"/home/simon/Desktop/maesy-training/od_checkpoints"
    resume = checkpoint != ""
    train_vit_detector(checkpoint, dataset, output, True, resume, False)

    # checkpoint = r"/home/simon/Desktop/maesy-training/od_checkpoints/leafy-music-38/best_model.pth"
    # images = r"/home/simon/Desktop/maesy-training/data/AllData (ObjectDetection)/train/images"
    # out = r"/home/simon/Desktop/maesy-training/inference_results"
    # infer_vit_detector(checkpoint, images, out, True, torch.device("cuda"))