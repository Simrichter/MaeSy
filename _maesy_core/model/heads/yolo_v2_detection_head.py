from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class YoloV2HeadConfig:
    type = "yolo_v2_head"

    num_classes: int
    num_anchors: int

class YoloV2Head(nn.Module):
    def __init__(self, num_classes: int, num_anchors: int):
        super(YoloV2Head, self).__init__()
        self.config = YoloV2HeadConfig(
            num_classes=num_classes+1, # TODO: add one for background class (only necessary if using DETR loss)
            num_anchors=num_anchors
        )
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.neck = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        # The output channels are determined by the number of anchors and classes
        self.det = nn.Conv2d(1024, num_anchors * (4 + num_classes), kernel_size=1)

    def forward(self, x):
        out = self.det(self.neck(x))
        B, _, w,h = out.shape # [B, num_anchors * (4 + num_classes), 14, 14]
        out_dict = {
            'pred_logits': out[:, :self.num_anchors * self.num_classes, :, :].reshape(B, -1, w*h).permute(0, 2, 1), # reshape(B, 14*14, -1),  # Class predictions
            'pred_boxes': out[:, self.num_anchors*self.num_classes:, :, :].reshape(B, -1, w*h).permute(0, 2, 1)
        }
        return out_dict