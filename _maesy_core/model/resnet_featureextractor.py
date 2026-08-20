from dataclasses import dataclass, field
from typing import Tuple

from _maesy_core.model import *
from _maesy_core.model.backbones import ResNetBackbone
from _maesy_core.model.backbones.resnet_backbone import ResNetBackboneConfig
from _maesy_core.model.heads import DummyHead

@dataclass
class ResNetFeatureExtractorConfig(BaseConfig):
    resnet_model: str = "resnet18"
    image_size: int = 224
    pretrained: bool = True
    out_layers: Tuple[str, ...] = ("c3", "c4", "c5")
    type: str = "resnet_feature_extractor"

class ResnetFeatureExtractor(BaseModel[ResNetFeatureExtractorConfig]):
    """
        A feature extractor model using a ResNet backbone and a dummy head.
    """
    def __init__(self, config: ResNetFeatureExtractorConfig):
        super().__init__(config)

        bbone_conf = ResNetBackboneConfig(
            version=config.resnet_model,
            image_size=config.image_size,
            pretrained=config.pretrained,
            feature_scales=config.out_layers
        )
        self.backbone = ResNetBackbone(bbone_conf)
        self.head = DummyHead()