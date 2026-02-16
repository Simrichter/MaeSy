"""Example evaluation script for Vision Transformer object detection."""

import torch
from torch.utils.data import DataLoader

from maesy.dataset import ObjectDetectionDataset, get_val_transforms
from maesy.dataset.transforms import collate_fn
from maesy.model import ModelConfig
from maesy.evaluation import evaluate_model


def main():
    """Main evaluation function."""
    
    # Configuration
    image_size = 224
    num_classes = 80
    batch_size = 16
    checkpoint_path = "./checkpoints/best_model.pth"
    
    # Prepare dataset
    print("Loading dataset...")
    val_images_dir = "./data/val/images"
    val_annotations = "./data/val/annotations.json"
    
    val_dataset = ObjectDetectionDataset(
        images_dir=val_images_dir,
        annotations_file=val_annotations,
        transforms=get_val_transforms(image_size=image_size)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("Loading model...")
    model_config = ModelConfig(
        image_size=image_size,
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_classes=num_classes,
        num_queries=100
    )
    
    model = VisionTransformerDetector(model_config)
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        confidence_threshold=0.5,
        iou_threshold=0.5
    )
    
    # Print detailed results
    print("\n=== Detailed Results ===")
    print(f"mAP@0.5: {results['mAP']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    # Print per-class AP if available
    if 'per_class_AP' in results:
        print("\n=== Per-Class Average Precision ===")
        category_names = val_dataset.get_category_names()
        for i, ap in enumerate(results['per_class_AP']):
            class_name = category_names[i] if i < len(category_names) else f"Class {i}"
            print(f"{class_name}: {ap:.4f}")


if __name__ == "__main__":
    main()
