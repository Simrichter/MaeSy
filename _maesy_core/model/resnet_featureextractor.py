from dataclasses import dataclass, field
from typing import Tuple, Dict

import torch

from _maesy_core.model import *
from _maesy_core.model.backbones import ResNetBackbone
from _maesy_core.model.backbones.resnet_backbone import ResNetBackboneConfig
from _maesy_core.model.heads import DummyHead

@dataclass
class ResnetFeatureExtractorConfig(BaseConfig):
    resnet_model: str = "resnet18"
    image_size: int = 224
    pretrained: bool = True
    out_layers: Tuple[str, ...] = ("c3", "c4", "c5")
    type: str = "resnet_feature_extractor"

class ResnetFeatureExtractor(BaseModel[ResnetFeatureExtractorConfig]):
    """
        A feature extractor model using a ResNet backbone and a dummy head.
    """
    def __init__(self, config: ResnetFeatureExtractorConfig):
        super().__init__(config)

        bbone_conf = ResNetBackboneConfig(
            version=config.resnet_model,
            image_size=config.image_size,
            pretrained=config.pretrained,
            feature_scales=config.out_layers
        )
        self.backbone = ResNetBackbone(bbone_conf)
        self.head = DummyHead()

    def get_output_dims(self) -> Dict[str, torch.Size]:
        return self.backbone.get_feature_dims()