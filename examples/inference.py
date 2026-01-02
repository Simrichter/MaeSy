"""Example inference script for Vision Transformer object detection."""

import torch
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
    
    # Configuration
    image_size = 224
    num_classes = 80
    checkpoint_path = "./checkpoints/best_model.pth"
    image_path = "./test_image.jpg"
    output_path = "./prediction.jpg"
    
    # Category names (example for COCO dataset)
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
        predictions = evaluator.predict(image, confidence_threshold=0.5)
    
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
