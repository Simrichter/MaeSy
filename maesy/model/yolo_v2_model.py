from dataclasses import dataclass
from maesy.model import BaseModel
from maesy.model.backbones.yolo_v2_backbone import YoloV2Backbone
from maesy.model.heads.yolo_v2_detection_head import YoloV2Head

@dataclass
class YoloV2Config:
    model: str = "YoloV2Model"
    backbone: str = "YoloV2Backbone"
    head: str = "YoloV2Head"
    num_classes: int = 4
    num_anchors: int = 3
    bbox_loss_coef: float = 5.0
    class_loss_coef: float = 1.0
    giou_loss_coef: float = 2.0

class YoloV2Model(BaseModel):
    """
    A feature extractor model using a ResNet backbone and a dummy head.

    """
    def __init__(self):
        super().__init__()
        self.config = YoloV2Config
        self.backbone = YoloV2Backbone()
        self.head = YoloV2Head(num_classes=self.config.num_classes, num_anchors=self.config.num_anchors)  # Default to COCO classes and anchors, can be modified as needed