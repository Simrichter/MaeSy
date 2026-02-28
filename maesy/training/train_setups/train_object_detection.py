"""
Example: Training a Vision Transformer for Object Detection

This script demonstrates how to train both ViTDetector
for object detection using the MaeSy framework.
"""
import os
import shutil

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from torchvision import transforms
from tqdm import tqdm

from maesy.evaluation.inferer import Inferer
# Import models
from maesy.model import ViTDetector, ViTDetectorConfig, DETR, DETRConfig
from maesy.model import MaskedAutoencoderViT, MAEConfig
from maesy.model.yolo_v2_model import YoloV2Model

# Import training components
from maesy.training import DetectionTrainer, TrainingConfig
from maesy.training.utils import collate_detection_fn

# Import dataset
from maesy.dataset import ObjectDetectionDataset, UnlabeledDataset

det_config = ViTDetectorConfig(
        image_size=224,
        patch_size=16,
        in_channels=3,

        # Backbone parameters
        embed_dim=384,
        num_layers=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,

        # Detection head parameters
        num_classes=3,
        num_queries=40, # 100
        num_decoder_layers=3,
        decoder_num_heads=4,
        hidden_dim=256,

        # Loss weights
        bbox_loss_coef=5.0,
        class_loss_coef=1.0,
        giou_loss_coef=2.0,
        eos_coef=2 #TODO: Move these to training config? Maybe add scheduling?
    )

detr_config = DETRConfig(
        image_size=224,
        patch_size=16,
        in_channels=3,

        # Backbone parameters
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,

        # Detection head parameters
        num_classes=3,
        num_queries=40, # 100
        num_decoder_layers=2,
        decoder_num_heads=4,
        hidden_dim=128,

        # Loss weights
        bbox_loss_coef=5.0,
        class_loss_coef=1.0,
        giou_loss_coef=2.0,
        eos_coef=0.1 #TODO: Move these to training config? Maybe add scheduling?
    )

def train_vit_detector(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    freeze_backbone: bool,
    continue_from_checkpoint: bool,
    enable_wandb: bool,
    seed: int = 42
):
    """
    Train an object detection model using MAE pretrained backbone.

    Args:
        checkpoint_path: Path to pretrained MAE checkpoint
        dataset_path: Path to object detection dataset
        output_dir: Directory to save checkpoints
        freeze_backbone: Whether to freeze the backbone during training
        continue_from_checkpoint: Whether to continue training from an existing OD checkpoint (in that case, checkpoint_path should point to an OD checkpoint instead of a MAE checkpoint)
        enable_wandb: Whether to enable Weights & Biases logging
        seed: Random seed for reproducibility (default: 42)
    """
    print("=" * 60)
    print("Training with MAE Pretrained Backbone")
    print("=" * 60)

    torch.manual_seed(seed)

    # Create detection model
    # model = ViTDetector(det_config)
    # model = YoloV2Model()
    model = DETR(detr_config)

    # Optionally freeze backbone
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Froze backbone parameters - only training detection head")
    else:
        print("No freeze: Fine-tuning entire model (backbone + head)")

    train_transforms = transforms.Compose([
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Create datasets and dataloaders
    train_dataset = ObjectDetectionDataset(f"{dataset_path}/train", transforms=train_transforms)
    val_dataset = ObjectDetectionDataset(f"{dataset_path}/val", transforms=val_transforms)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_detection_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        num_workers=4,
        collate_fn=collate_detection_fn,
        shuffle=True,
        pin_memory=True
    )

    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=100,
        learning_rate=1e-4 if not freeze_backbone else 1e-3,  # Higher LR when only training head
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler="cosine",
        warmup_epochs=5,
        save_frequency=20,
        save_dir=output_dir,
        criterion= "DetectionLoss", #"YOLOv8Loss", #
        use_amp=True,
        # device=torch.device("cpu")
    )

    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        project_name="maesy-object-detection",
        enable_wandb=enable_wandb
    )

    if checkpoint_path != "":
        # Transfer backbone weights
        trainer.load_checkpoint(checkpoint_path, model_only=not continue_from_checkpoint)

    # Train
    trainer.train()

def infer_vit_detector(checkpoint_path: str, images_path: str, out_path: Path, visualize: bool, device: torch.device) -> None:
    """
    Run inference with a trained object detection model.

    Args:
        :param checkpoint_path: Path to trained model checkpoint
        :param images_path: Path to input image for inference
        :param out_path: Path to save inference results (predicted bounding boxes and labels)
        :param visualize: Whether to save visualizations of predictions (e.g., images with predicted boxes drawn)
        :param device: Device to run inference on (e.g., "cuda" or "cpu")
    """
    from maesy.model_tools import CheckpointHandler
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    # Load model
    # model = ViTDetector(det_config)
    model = YoloV2Model()

    CheckpointHandler(device=device).load_checkpoint(checkpoint_path, model=model)
    model.eval()

    dataset = UnlabeledDataset(Path(images_path), transforms=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]), step=50, use_first_n=30)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    inferer = Inferer(model=model, data_loader=dataloader, device=device)
    preds, _ = inferer.infer() # List[Dict] with keys "pred_boxes" (B X num_querys X 4) and "pred_logits" (B X num_queries)]

    print("=" * 60)
    print(f"Saving inference results to {out_path}...")
    print("=" * 60)

    # torch.cat(preds, dim=0) # Total number of images X num_queries X (4 or num_classes)

    images_dir = dataset.images_dir
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    for p in tqdm(zip(dataset.images, preds)):
        img_path = images_dir/p[0]
        shutil.copy(img_path, out_path/p[0])
        with open(out_path/Path(p[0]).with_suffix(".txt"), "w") as f:
            boxes = torch.unbind(p[1]["pred_boxes"].squeeze(0), dim=0)
            labels = torch.unbind(p[1]["pred_logits"].squeeze(0), dim=0)
            for box, label in zip(boxes, labels):
                cx, cy, w, h = box
                score, l = label.max(-1)
                if l != 3 and score>=0.8:
                    f.write(f"{l} {cx.item()} {cy.item()} {w.item()} {h.item()}\n")
    if visualize:
        from maesy.evaluation import visualize_annotations
        visualize_annotations(out_path, "")

if __name__ == "__main__":
    #argparse:
    import argparse
    parser = argparse.ArgumentParser(description="Train a ViTDetector for object detection")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to pretrained MAE checkpoint (or OD checkpoint if --continue_from_checkpoint is set)")
    parser.add_argument("--dataset", type=str, default="/home/simon/Desktop/maesy-training/data/AllData (ObjectDetection)", help="Path to object detection dataset")
    parser.add_argument("--output", type=str, default="./od_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")

    args = parser.parse_args()
    train_vit_detector(args.checkpoint, args.dataset, args.output, False, False, True)