import shutil

# import imgaug
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from maesy.evaluation.inferer import Inferer
# Import models
from maesy.model import DETR, DETRConfig, RTDETR, RTDETRConfig
from maesy.model_tools import replace_bn_with_frozenbn

# Import training components
from maesy.training import DetectionTrainer, TrainingConfig
from maesy.training.utils import collate_detection_fn

# Import dataset
from maesy.dataset import ObjectDetectionDataset, UnlabeledDataset

DETR_CONFIG = DETRConfig(

        image_size=224,

        # Backbone parameters
        embed_dim=256,
        resnet_version="resnet50",
        # freeze_backbone=True,

        # Detection head parameters
        num_classes=3,
        num_queries=30, # 100
        num_encoder_layers=4,
        num_decoder_layers=4,
        encoder_num_heads=4,
        decoder_num_heads=4,
        hidden_dim_out_layers=512,

        # Loss weights
        bbox_loss_coef=5, #1.0,
        class_loss_coef=2.0,#2.0,
        giou_loss_coef=2.0, #1.0,
        eos_coef=0.05,
        aux_loss_coef=0.5,
    )

RT_DETR_CONFIG = RTDETRConfig(
    image_size=224,
    resnet_version="resnet50",
    num_classes=3,
    num_queries=30,
    embed_dim=256,
    num_decoder_layers=4,
    decoder_num_heads=8,
    hidden_dim_out_layers=512,
    bbox_loss_coef=5.0,
    class_loss_coef=2.0,
    giou_loss_coef=2.0,
    eos_coef=0.05,
    aux_loss_coef=0.5,
)


def _build_detection_model(detector_arch: str):
    detector_arch = detector_arch.lower()
    if detector_arch == "detr":
        return DETR(DETR_CONFIG)
    if detector_arch == "rt_detr":
        return RTDETR(RT_DETR_CONFIG)
    raise ValueError(f"Unsupported detector architecture: {detector_arch}")


def _infer_detector_arch_from_checkpoint(checkpoint_path: str, device: torch.device) -> str:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    head_type = str(checkpoint.get("headtype", ""))
    if "RTDETR" in head_type:
        return "rt_detr"
    return "detr"

def train_vit_detector(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    no_freeze: bool,
    continue_from_checkpoint: bool,
    enable_wandb: bool,
    detector_arch: str = "rt_detr",
    seed: int = 42
):
    """
    Train an object detection model with the selected detector architecture.

    Args:
        checkpoint_path: Path to a checkpoint (pretrained or full OD checkpoint)
        dataset_path: Path to object detection dataset
        output_dir: Directory to save checkpoints
        no_freeze: Whether to continue training the backbone
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
    model = _build_detection_model(detector_arch)
    print(f"Selected detector architecture: {detector_arch}")

    # Optionally freeze backbone
    if not no_freeze:
        for param in model.backbone.parameters():
            param.requires_grad = False
        # print("Froze backbone parameters - only training detection head")
    else:
        print("No freeze: Fine-tuning entire model (backbone + head)")

    train_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=8, translate=(0.15, 0.15), scale=(0.95, 1.05)),
        # VerticalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    val_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Create datasets and dataloaders
    train_dataset = ObjectDetectionDataset(f"{dataset_path}/train", transforms=train_transforms)
    val_dataset = ObjectDetectionDataset(f"{dataset_path}/val", transforms=val_transforms)
    batch_size = 64 if detector_arch.lower() == "rt_detr" else 64

    # mp.set_sharing_strategy("file_system")
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        collate_fn=collate_detection_fn,
        pin_memory=True,
        # worker_init_fn=lambda worker_id: imgaug.seed(np.random.get_state()[1][0] + worker_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_detection_fn,
        shuffle=True,
        pin_memory=True
    )

    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=1000,
        learning_rate=1e-5 if no_freeze else 1e-4,  # Higher LR when only training head
        backbone_learning_rate=1e-6 if no_freeze else 0.0,  # Lower LR for backbone if fine-tuning, otherwise 0
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler="cosine",
        warmup_epochs=4,
        save_frequency=100,
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

    replace_bn_with_frozenbn(trainer.model.backbone)

    # Train
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
    """
    from maesy.model_tools import CheckpointHandler
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    # Load model
    # model = ViTDetector(det_config)
    # model = YoloV2Model()
    selected_arch = detector_arch or _infer_detector_arch_from_checkpoint(checkpoint_path, device)
    model = _build_detection_model(selected_arch)
    print(f"Selected detector architecture: {selected_arch}")

    CheckpointHandler(device=device).load_checkpoint(checkpoint_path, model=model)
    model.eval()

    dataset = UnlabeledDataset(Path(images_path), transforms=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), step=50) #, use_first_n=30
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
    save_all_predictions = True  # Intentionally keep all query outputs for debugging.
    for p in tqdm(zip(dataset.images, preds)):
        img_path = images_dir/p[0]
        shutil.copy(img_path, out_path/p[0])
        with open(out_path/Path(p[0]).with_suffix(".txt"), "w") as f:
            boxes = torch.unbind(p[1]["pred_boxes"].squeeze(0), dim=0)
            labels = torch.unbind(p[1]["pred_logits"].squeeze(0), dim=0)
            for box, label in zip(boxes, labels):
                cx, cy, w, h = box
                score, l = label.max(-1)
                if save_all_predictions or (l != 3 and score >= 0.1):
                    f.write(f"{l.item()} {cx.item()} {cy.item()} {w.item()} {h.item()}\n")
    if visualize:
        from maesy.evaluation import visualize_annotations
        visualize_annotations(out_path, "")

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