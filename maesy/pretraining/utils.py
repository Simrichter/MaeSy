"""Utilities for loading pretrained weights."""

import torch
from typing import Dict, Optional
from ..model import VisionTransformerDetector
from .mae_model import MaskedAutoencoderViT
from .classification_model import ClassificationViT


def load_mae_pretrained_weights(
    detector: VisionTransformerDetector,
    mae_checkpoint_path: str,
    strict: bool = False
) -> VisionTransformerDetector:
    """
    Load MAE pretrained weights into a detector model.
    
    Args:
        detector: VisionTransformerDetector model
        mae_checkpoint_path: Path to MAE checkpoint
        strict: Whether to strictly enforce that keys match
        
    Returns:
        Detector model with loaded weights
    """
    # Load checkpoint
    checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
    mae_state_dict = checkpoint['model_state_dict']
    
    # Map MAE encoder weights to detector
    detector_state_dict = detector.state_dict()
    pretrained_dict = {}
    
    # Mapping rules
    for key, value in mae_state_dict.items():
        if key.startswith('patch_embed'):
            # Patch embedding
            detector_key = key
            if detector_key in detector_state_dict:
                pretrained_dict[detector_key] = value
        elif key == 'cls_token':
            # Class token
            if key in detector_state_dict:
                pretrained_dict[key] = value
        elif key == 'pos_embed':
            # Positional embedding
            if key in detector_state_dict:
                pretrained_dict[key] = value
        elif key.startswith('encoder_blocks'):
            # Encoder blocks
            detector_key = key
            if detector_key in detector_state_dict:
                pretrained_dict[detector_key] = value
        elif key == 'encoder_norm':
            # Encoder normalization
            detector_key = 'norm'
            if detector_key in detector_state_dict:
                pretrained_dict[detector_key] = value
    
    # Load the pretrained weights
    detector_state_dict.update(pretrained_dict)
    detector.load_state_dict(detector_state_dict, strict=strict)
    
    print(f"Loaded {len(pretrained_dict)} pretrained weights from MAE checkpoint")
    print(f"Loaded weights for: patch_embed, cls_token, pos_embed, encoder_blocks, norm")
    
    return detector


def load_classification_pretrained_weights(
    detector: VisionTransformerDetector,
    classification_checkpoint_path: str,
    strict: bool = False
) -> VisionTransformerDetector:
    """
    Load classification pretrained weights into a detector model.
    
    Args:
        detector: VisionTransformerDetector model
        classification_checkpoint_path: Path to classification checkpoint
        strict: Whether to strictly enforce that keys match
        
    Returns:
        Detector model with loaded weights
    """
    # Load checkpoint
    checkpoint = torch.load(classification_checkpoint_path, map_location='cpu')
    cls_state_dict = checkpoint['model_state_dict']
    
    # Map classification encoder weights to detector
    detector_state_dict = detector.state_dict()
    pretrained_dict = {}
    
    # Mapping rules
    for key, value in cls_state_dict.items():
        if key.startswith('patch_embed'):
            # Patch embedding
            detector_key = key
            if detector_key in detector_state_dict:
                pretrained_dict[detector_key] = value
        elif key == 'cls_token':
            # Class token
            if key in detector_state_dict:
                pretrained_dict[key] = value
        elif key == 'pos_embed':
            # Positional embedding
            if key in detector_state_dict:
                pretrained_dict[key] = value
        elif key.startswith('encoder_blocks'):
            # Encoder blocks
            detector_key = key
            if detector_key in detector_state_dict:
                pretrained_dict[detector_key] = value
        elif key == 'norm':
            # Normalization layer
            detector_key = key
            if detector_key in detector_state_dict:
                pretrained_dict[detector_key] = value
    
    # Load the pretrained weights
    detector_state_dict.update(pretrained_dict)
    detector.load_state_dict(detector_state_dict, strict=strict)
    
    print(f"Loaded {len(pretrained_dict)} pretrained weights from classification checkpoint")
    print(f"Loaded weights for: patch_embed, cls_token, pos_embed, encoder_blocks, norm")
    
    return detector


def freeze_encoder(detector: VisionTransformerDetector) -> VisionTransformerDetector:
    """
    Freeze encoder parameters in detector model.
    
    Args:
        detector: VisionTransformerDetector model
        
    Returns:
        Detector model with frozen encoder
    """
    # Freeze patch embedding
    for param in detector.patch_embed.parameters():
        param.requires_grad = False
    
    # Freeze cls token and pos encoding
    detector.cls_token.requires_grad = False
    for param in detector.pos_encoding.parameters():
        param.requires_grad = False
    
    # Freeze encoder blocks
    for block in detector.encoder_blocks:
        for param in block.parameters():
            param.requires_grad = False
    
    # Freeze encoder norm
    for param in detector.norm.parameters():
        param.requires_grad = False
    
    print("Encoder parameters frozen")
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in detector.parameters())
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return detector


def unfreeze_encoder(detector: VisionTransformerDetector) -> VisionTransformerDetector:
    """
    Unfreeze encoder parameters in detector model.
    
    Args:
        detector: VisionTransformerDetector model
        
    Returns:
        Detector model with unfrozen encoder
    """
    # Unfreeze all parameters
    for param in detector.parameters():
        param.requires_grad = True
    
    print("All parameters unfrozen")
    
    return detector
