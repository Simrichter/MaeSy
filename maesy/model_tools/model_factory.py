from maesy.model import BaseModel

def _print_model_info(model: BaseModel):
    """Utility function to print model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

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
        raise ValueError(f"Model {model} not recognized. Available models: ['mae', 'ViTDetector', 'detr', 'rt_detr']")

    _print_model_info(instance)
    return instance
