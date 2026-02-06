from maesy.model import BaseModel

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

        return MaskedAutoencoderViT(config=config)
    elif model == "transformer_detector":
        from maesy.model import TransformerDetectionModel
        return TransformerDetectionModel(config=config)
    else:
        raise ValueError(f"Model {model} not recognized. Available models: ['mae']")
