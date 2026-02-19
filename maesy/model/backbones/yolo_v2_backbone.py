import torch
import torch.nn as nn

class YoloV2Backbone(nn.Module):
    def __init__(self):
        super(YoloV2Backbone, self).__init__()
        self.type = "yolo_v2_backbone"
        # Define the layers of the YOLOv2 backbone (Darknet-19)
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            # Layer 4
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 6
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # Layer 7
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            # Layer 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # Layer 9
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # Layer 10
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # Layer 11
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # Layer 12
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # Layer 13
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 14
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            # Layer 15
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # Layer 16
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            # Layer 17
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # Layer 18
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.features(x)