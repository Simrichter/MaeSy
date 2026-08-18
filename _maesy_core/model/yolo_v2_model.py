from dataclasses import dataclass
from _maesy_core.model import BaseModel
from _maesy_core.model.backbones.yolo_v2_backbone import YoloV2Backbone
from _maesy_core.model.heads.yolo_v2_detection_head import YoloV2Head

@dataclass
class YoloV2Config:
    model: str = "YoloV2Model"
    backbone: str = "YoloV2Backbone"
    head: str = "YoloV2Head"
    num_classes: int = 3
    num_anchors: int = 1
    bbox_loss_coef: float = 5.0
    class_loss_coef: float = 1.0
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1

class YoloV2Model(BaseModel):
    """
    A feature extractor model using a ResNet backbone and a dummy head.

    """
    def __init__(self):
        super().__init__()
        self.config = YoloV2Config
        self.backbone = YoloV2Backbone()
        self.head = YoloV2Head(num_classes=self.config.num_classes+1, num_anchors=self.config.num_anchors)  # Default to COCO classes and anchors, can be modified as needed