from maesy.model import BaseModel
from maesy.model.backbones import ResNetBackbone
from maesy.model.heads import DummyHead


class ResnetFeatureExtractor(BaseModel):
    """
    A feature extractor model using a ResNet backbone and a dummy head.
    Args:
        :param resnet_model: ResNet version ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
    """
    def __init__(self, resnet_model, img_size, remove_layers):
        super().__init__()
        self.backbone = ResNetBackbone(resnet_model, img_size, remove_layers)
        self.head = DummyHead()