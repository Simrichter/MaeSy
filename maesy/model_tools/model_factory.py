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
        from maesy.model import MaskedAutoencoderViT, BaseModel

        model = MaskedAutoencoderViT(config=config)
        _print_model_info(model)
        return model
    elif model == "ViTDetector":
        from maesy.model import ViTDetector
        model = ViTDetector(config=config)
        _print_model_info(model)
        return model
    else:
        raise ValueError(f"Model {model} not recognized. Available models: ['mae']")
