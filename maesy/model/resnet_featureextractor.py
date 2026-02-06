from maesy.model import BaseModel
from maesy.model.backbones import ResNetBackbone
from maesy.model.heads import DummyHead


class ResnetFeatureExtractor(BaseModel):
    def __init__(self, resnet_model):
        super().__init__()
        self.backbone = ResNetBackbone(resnet_model)
        self.head = DummyHead()