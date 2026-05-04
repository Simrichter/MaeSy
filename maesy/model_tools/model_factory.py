import os
from typing import Dict

import torch
import yaml

from maesy.model import (
    BaseModel,
    DETR,
    DETRConfig,
    MAEConfig,
    RTDETR,
    RTDETRConfig,
    MaeMultiscaleConfig,
    MaskedAutoencoderMultiscale,
    MaskedAutoencoderViT,
)
from maesy.model_tools import CheckpointHandler

known_architectures = ["rt-detr", "detr", "mae", "mae-multiscale"]

def _print_model_info(model: BaseModel):
    """Utility function to print model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def create_model_from_config(config: Dict) -> BaseModel:
    """
        Creates a model according to the provided config.

        Args:
            :param config: The config dict to be passed to the model
        Returns:
            model: The initialized model according to the provided config
    """
    match config.get("type", "").lower():
        case "mae":
            return MaskedAutoencoderViT(config=MAEConfig(**config))
        case "mae-multiscale":
            return MaskedAutoencoderMultiscale(config=MaeMultiscaleConfig(**config))
        case "vit":
            raise ValueError("model type 'vit' is not supported")
        case "detr":
            return DETR(DETRConfig(**config))
        case "rt-detr":
            return RTDETR(RTDETRConfig(**config))
        case _:
            raise ValueError(f"Model type '{config.get('type', '')}' not recognized. Supported types: ['mae', 'mae-multiscale', 'detr', 'rt_detr']")

def create_model_from_checkpoint(checkpoint: str) -> BaseModel:
    """
    Creates a model according to the provided checkpoint.

    Args:
        :param checkpoint: The checkpoint file to be loaded. Must be a .pth file created by this framework in an earlier training run

    Returns:
        model: The initialized model according to the provided checkpoint
    """
    checkpoint_handler = CheckpointHandler(device = torch.device("cpu"))
    model = checkpoint_handler.load_model(checkpoint)
    return model

def read_yaml(path:str) -> Dict:
    """
    Reads a yaml file and returns its contents as a dictionary.
    Args:
        :param path: The path to the yaml file
    Returns:
        A dictionary containing the contents of the yaml file
    """
    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.normpath(os.path.join(project_root, path))
    if not os.path.isfile(path) or not path.endswith((".yaml", ".yml")):
        raise FileNotFoundError(f"File '{path}' is not a yaml file or does not exist")
    with open(path, "r") as f:
        return yaml.full_load(f)

def create_model(model: str, config) -> BaseModel:
    """

    Parameters
    ----------
    :param model:
    :param config: The config to be passed to the model

    Returns
    model: The initialized model according to the provided config
    -------

    """
    if model == "mae":
        from maesy.model import MaskedAutoencoderViT, MAEConfig

        if not isinstance(config, MAEConfig):
            raise TypeError(f"Model 'mae' expects config type MAEConfig, got {type(config).__name__}")
        instance = MaskedAutoencoderViT(config=config)
    elif model == "mae_multiscale":
        from maesy.model import MaskedAutoencoderMultiscale, MaeMultiscaleConfig

        if not isinstance(config, MaeMultiscaleConfig):
            raise TypeError(f"Model 'mae_multiscale' expects config type MaeMultiscaleConfig, got {type(config).__name__}")
        instance = MaskedAutoencoderMultiscale(config=config)
    elif model == "ViTDetector":
        from maesy.model import ViTDetector, ViTDetectorConfig

        if not isinstance(config, ViTDetectorConfig):
            raise TypeError(f"Model 'ViTDetector' expects config type ViTDetectorConfig, got {type(config).__name__}")
        instance = ViTDetector(config=config)
    elif model == "detr":
        from maesy.model import DETR, DETRConfig

        if not isinstance(config, DETRConfig):
            raise TypeError(f"Model 'detr' expects config type DETRConfig, got {type(config).__name__}")
        instance = DETR(config=config)
    elif model == "rt_detr":
        from maesy.model import RTDETR, RTDETRConfig

        if not isinstance(config, RTDETRConfig):
            raise TypeError(f"Model 'rt_detr' expects config type RTDETRConfig, got {type(config).__name__}")
        instance = RTDETR(config=config)
    else:
        raise ValueError(f"Model {model} not recognized. Available models: ['mae', 'mae_multiscale', 'ViTDetector', 'detr', 'rt_detr']")

    _print_model_info(instance)
    return instance
