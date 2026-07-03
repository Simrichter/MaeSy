import shutil
from dataclasses import dataclass
from typing import List, Any
import logging

import torch
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path

from tqdm import tqdm

from maesy.evaluation.inferer import Inferer
# Import models
from maesy.model_tools.layer_manipulations import replace_bn_with_frozenbn
from maesy.model_tools.checkpoint_handler import  CheckpointHandler
from maesy.model_tools.model_factory import create_model_from_config, known_architectures, create_model_from_checkpoint, read_yaml

# Import training components
from maesy.training import DetectionTrainer, BaseTrainingConfig
from maesy.training.utils import collate_detection_fn

# Import dataset
from maesy.dataset import MaesyDataset, MultiDataset
from maesy.dataset.augmentations import ODTrainTransforms, ODValTransforms

@dataclass
class ODTrainingConfig(BaseTrainingConfig):
    """Configuration for training."""
    use_focal_loss: bool = False,
    bbox_loss_coef: float = 5.0,
    class_loss_coef: float = 1.0,
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1
    aux_loss_coef: float = 0.5
    enc_loss_coef: float = 0.3
    line_loss_coef: float = 2.0
    line_angle_loss_coef: float = 0.5
    line_length_loss_coef: float = 0.5
    ellipse_loss_coef: float = 2.0
    ellipse_shape_coef: float = 1.0
    dn_loss_coef: float = 1.0


def train_vit_detector(
    model_info: str,
    dataset_paths: list[str],
    output_dir: str,
    finetune: bool,
    continue_training_from_checkpoint: bool,
    pretrained_backbone: str,
    enable_wandb: bool,
    enable_denoising: bool = False,
    denoising_num_queries: int = 0,
    denoising_label_noise_ratio: float = 0.2,
    denoising_box_noise_scale: float = 0.4,
    enable_line_detection: bool = True,
    enable_ellipse_detection: bool = True,
    override_params: dict = {},
    seed: int = 42,
    device: str = "auto",
    fast_mode: bool = False,
    debug: bool = False,
):
    """
    Train an object detection model with the selected detector architecture.

    Args:
        :param model_info: String that specifies the model configuration to be used. Either a path to a checkpoint, or a model architecture like "rt-detr"
        :param dataset_paths: List of paths to object detection MaesyDatasets
        :param output_dir: Directory to save checkpoints
        :param finetune: If set, finetuning parameters are used in config
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    if debug:
        # activate automatic anomaly detection in the gradient calculation
        torch.autograd.set_detect_anomaly(True)
        torch.autograd.profiler.emit_nvtx()

        # Force attention to math backend, because fused operations might hide bugs
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        fast_mode = True # disable checkpointing
        # enable debug outputs (and checks)
        logger = logging.getLogger("MaeSy")
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging is activated")

    torch.manual_seed(seed)

    train_transforms = ODValTransforms(image_size=224)
    val_transforms = ODValTransforms(image_size=224)

    # None activates auto-infer, -1 deactivates class in MaesyDataset
    special_class_deactivation_dict = {'line_class_id': None if enable_line_detection else -1,
                                       'ellipse_class_id': None if enable_ellipse_detection else -1}

    train_dataset = MultiDataset([MaesyDataset(dataset_path, "train", "detection", train_transforms, enable_lines=enable_line_detection, enable_ellipses=enable_ellipse_detection) for dataset_path in dataset_paths])
    val_dataset = MultiDataset([MaesyDataset(dataset_path, "val", "detection", val_transforms, enable_lines=enable_line_detection, enable_ellipses=enable_ellipse_detection) for dataset_path in dataset_paths])
    assert train_dataset.get_special_classes() == val_dataset.get_special_classes(), "Error, train and val datasets must have same special classes!"

    # Create training configuration
    training_config = ODTrainingConfig(
        batch_size= override_params.get("batch_size", 32), # 64 l# TODO: Everything apart from largest model config (rt-detr6) was with 64
        num_epochs=3000,
        learning_rate= override_params.get("learning_rate", 5e-6 if finetune else 5e-5),
        backbone_learning_rate=5e-7 if finetune else 5e-6,
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler= "plateau", # "cosine",
        plateau_metric= "metrics/total_mAP", # "val_losses/total_loss", #
        patience=40 if finetune else 80,
        lr_step_factor=0.3,
        min_num_epochs_per_plateau=50 if finetune else 100,
        warmup_epochs=4, # Not used with Plateau scheduling
        label_smoothing=0.0 if finetune else 0.1,
        save_frequency=100,
        log_frequency=50,
        save_checkpoints=not fast_mode,
        save_dir=output_dir,
        criterion="DetectionLoss",  # "YOLOv8Loss", #
        use_amp=False,
        use_focal_loss=False,
        bbox_loss_coef = 5.0,
        class_loss_coef = 1.0,
        giou_loss_coef = 2.0,
        eos_coef = 0.1, # Was 0.2 for most runs
        aux_loss_coef = 0.5,
        enc_loss_coef = 0.3,
        line_loss_coef = 2.0,
        line_angle_loss_coef = 0.1,
        line_length_loss_coef = 0.05,
        ellipse_loss_coef = 2.0,
        ellipse_shape_coef=1.0,
        dn_loss_coef = 1.0,
        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu') if device=="auto" else torch.device(device)
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
        config["softmax_activated"] = not training_config.use_focal_loss
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
        config["enable_denoising"] = enable_denoising
        config["denoising_num_queries"] = denoising_num_queries
        config["denoising_label_noise_ratio"] = denoising_label_noise_ratio
        config["denoising_box_noise_scale"] = denoising_box_noise_scale
        model = create_model_from_config(config)
    elif model_info.endswith(".pth"):
        model = create_model_from_checkpoint(model_info)
        # Apply denoising parameters to the loaded model's config
        assert model.config.softmax_activated == (not training_config.use_focal_loss), f"Model config softmax_activated ({model.config.softmax_activated}) does not match expected value based on training config use_focal_loss ({training_config.use_focal_loss}). This might lead to unexpected behavior during training. Please make sure the model config and training config are compatible."
        model.config.enable_denoising = enable_denoising
        model.config.denoising_num_queries = denoising_num_queries
        model.config.denoising_label_noise_ratio = denoising_label_noise_ratio
        model.config.denoising_box_noise_scale = denoising_box_noise_scale
        # Recreate denoising query content if needed
        model.head.config.enable_denoising = enable_denoising
        model.head.config.denoising_num_queries = denoising_num_queries
        model.head.config.denoising_label_noise_ratio = denoising_label_noise_ratio
        model.head.config.denoising_box_noise_scale = denoising_box_noise_scale
        if enable_denoising and denoising_num_queries > 0:
            model.head.dn_query_content = torch.nn.Embedding(denoising_num_queries, model.head.config.embed_dim)
        else:
            model.head.dn_query_content = None
    else:
        raise ValueError(f"Model {model_info} is neither in {known_architectures} nor is it a path to a training checkpoint (must end with '.pth')")

    if pretrained_backbone != "":
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
        if enable_ellipse_detection:
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
    split: str,
    confidence: float,
) -> None:
    """
    Run inference with a trained object detection model.

    Args:
        :param checkpoint_path: Path to trained model checkpoint
        :param dataset_path: Path to input images for inference
        :param out_path: Path to save inference results (predicted bounding boxes and labels)
        :param visualize: Whether to save visualizations of predictions (e.g., images with predicted boxes drawn)
        :param device: Device to run inference on (e.g., "cuda" or "cpu")
        :param split: Str to specify dataset split to run inference on (choice of ["train", "val", "test"]).
        :param confidence: Confidence threshold to filter predictions
    """
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    # Load model
    model = create_model_from_checkpoint(checkpoint_path) #CheckpointHandler(device=device).load_model(checkpoint_path)
    model.eval()
    infer_transforms = ODValTransforms(image_size=224)

    dataset = MaesyDataset(dataset_path, split, "None", transforms=infer_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    inferer = Inferer(model=model, data_loader=dataloader, device=device)
    preds, _ = inferer.infer(score_threshold=confidence) # List[Dict] with keys "boxes" (B X num_querys X 4) and "labels" (B X num_queries) (scores, lines etc.)

    print("=" * 60)
    print(f"Saving inference results to {out_path}...")
    print("=" * 60)

    # torch.cat(preds, dim=0) # Total number of images X num_queries X (4 or num_classes)

    images_dir = dataset.images_dir
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    for img, pred in tqdm(zip(dataset.images, preds)):
        img_path = images_dir/img
        shutil.copy(img_path, out_path/img)
        with open(out_path/Path(img).with_suffix(".txt"), "w") as f:
            boxes = pred["boxes"]
            labels = pred["labels"]
            scores = pred["scores"]
            for box, label, score in zip(boxes, labels, scores):
                cx, cy, w, h = box
                f.write(f"{label.item()} {cx.item()} {cy.item()} {w.item()} {h.item()}\n")
            for line in pred["line_points"]:
                f.write(f"{model.head.config.line_class_id} {line[0].item()} {line[1].item()} {line[2].item()} {line[3].item()}\n")
            for ellipse in pred["ellipses"]:
                f.write(f"{model.head.config.ellipse_class_id} {ellipse[0].item()} {ellipse[1].item()} {ellipse[2].item()} {ellipse[3].item()} {ellipse[4].item()} {ellipse[5].item()}\n")

    if visualize:
        from maesy.evaluation import visualize_data
        visualize_data(str(out_path), "", special_classes={"lines": model.head.config.line_class_id, "ellipses": model.head.config.ellipse_class_id})

def export_vit_detector(
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

if __name__ == "__main__":
    train_vit_detector(
        model_info="rt-detr6",
        dataset_paths=["data/Cvat"],
        output_dir="./od_checkpoints",
        finetune=False,
        continue_training_from_checkpoint=False,
        pretrained_backbone="",
        enable_wandb=False,
        enable_denoising=True,
        denoising_num_queries=100,
        denoising_label_noise_ratio=0.0,
        denoising_box_noise_scale=0.4,
        enable_line_detection=True,
        enable_ellipse_detection=False,
        override_params={"batch_size": 2},
        device="cpu"
    )