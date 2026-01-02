"""Example inference script for Vision Transformer object detection."""

import torch
import argparse
from PIL import Image
import torchvision.transforms as T

from maesy.model import VisionTransformerDetector, ModelConfig
from maesy.evaluation import Evaluator


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Load and preprocess image.
    
    Args:
        image_path: Path to image
        image_size: Target image size
        
    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)


def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(description='Run inference with Vision Transformer detector')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default='./test_image.jpg',
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='./prediction.jpg',
                        help='Path to save output visualization')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num-classes', type=int, default=80,
                        help='Number of object classes')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Configuration
    checkpoint_path = args.checkpoint
    image_path = args.image
    output_path = args.output
    image_size = args.image_size
    num_classes = args.num_classes
    
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    
    # Load image
    print(f"Loading image from {image_path}...")
    image = load_image(image_path, image_size)
    
    # Create evaluator for visualization
    evaluator = Evaluator(model, None, device=str(device))
    
    # Make prediction
    print("Running inference...")
    with torch.no_grad():
        predictions = evaluator.predict(image, confidence_threshold=args.confidence)
    
    # Category names (example for COCO dataset - 80 classes)
    category_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Extend or truncate category names based on num_classes
    if num_classes != len(category_names):
        if num_classes < len(category_names):
            category_names = category_names[:num_classes]
        else:
            category_names.extend([f"class_{i}" for i in range(len(category_names), num_classes)])
    
    # Print predictions
    print(f"\nDetected {len(predictions['boxes'])} objects:")
    for i, (box, label, score) in enumerate(zip(
        predictions['boxes'],
        predictions['labels'],
        predictions['scores']
    )):
        class_name = category_names[label] if label < len(category_names) else f"Class {label}"
        print(f"  {i+1}. {class_name}: {score:.3f}")
    
    # Visualize
    print(f"\nSaving visualization to {output_path}...")
    evaluator.visualize_predictions(
        image=image,
        predictions=predictions,
        category_names=category_names,
        save_path=output_path
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
