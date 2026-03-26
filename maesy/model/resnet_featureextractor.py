from maesy.model import BaseModel
from maesy.model.backbones import ResNetBackbone
from maesy.model.backbones.resnet_backbone import ResNetBackboneConfig
from maesy.model.heads import DummyHead


class ResnetFeatureExtractor(BaseModel):
    """
    A feature extractor model using a ResNet backbone and a dummy head.
    Args:
        :param resnet_model: ResNet version ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
    """
    def __init__(self, resnet_model, img_size, out_layers):
        super().__init__()

        bbone_conf = ResNetBackboneConfig(
            version=resnet_model,
            image_size=img_size,
            pretrained=True,
            feature_scales=out_layers
        )
        self.backbone = ResNetBackbone(bbone_conf)
        self.head = DummyHead()