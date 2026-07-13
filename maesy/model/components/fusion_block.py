import torch
import torch.nn as nn
import torch.nn.functional as F

class RepConv(nn.Module):
    def __init__(self, C: int) -> None:
        super().__init__()
        self.fused_conv = nn.Conv2d(C, C, 3, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, 1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(C)
        self.bn1 = nn.BatchNorm2d(C)
        self.bn0 = nn.BatchNorm2d(C)

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            with torch.no_grad():
                # 3x3 conv:
                kernel_scaling_factor = (self.bn3.weight / torch.sqrt(self.bn3.running_var + self.bn3.eps))
                self.fused_conv.weight.copy_(self.conv3x3.weight * kernel_scaling_factor.view(-1, 1, 1, 1))
                self.fused_conv.bias.copy_(kernel_scaling_factor * -self.bn3.running_mean + self.bn3.bias)

                # 1x1 conv:
                kernel_scaling_factor = (self.bn1.weight / torch.sqrt(self.bn1.running_var + self.bn1.eps))
                w1 = self.conv1x1.weight * kernel_scaling_factor.view(-1, 1, 1, 1)
                self.fused_conv.weight += F.pad(w1, (1, 1, 1, 1), "constant", 0)
                self.fused_conv.bias += kernel_scaling_factor * (-self.bn1.running_mean) + self.bn1.bias

                # identity:
                kernel_scaling_factor = (self.bn0.weight / torch.sqrt(self.bn0.running_var + self.bn0.eps))
                w0 = torch.zeros_like(self.conv3x3.weight)
                for i in range(w0.shape[0]):  # Dirac-style identity kernel to match input channel i to output channel i
                    w0[i, i, 1, 1] = kernel_scaling_factor[i]
                self.fused_conv.weight += w0
                self.fused_conv.bias += kernel_scaling_factor * -self.bn0.running_mean + self.bn0.bias

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.relu(self.bn3(self.conv3x3(x)) + self.bn1(self.conv1x1(x)) + self.bn0(x))
        else:
            return F.relu(self.fused_conv(x))


class FusionBlock(nn.Module):
    """
        Implementation of a Fusion Block as described in the RT-DETR paper.
        Fuses two equally sized feature maps.
        Uses RepVGG blocks that fuse into a single convolution during inference
    """
    def __init__(self, N: int, C: int) -> None:
        super().__init__()
        self.num_repblocks = N
        self.pointwise_conv_upper_path = nn.Sequential(nn.Conv2d(2 * C, C, 1), nn.ReLU())
        self.pointwise_conv_lower_path = nn.Sequential(nn.Conv2d(2 * C, C, 1), nn.ReLU())
        self.rep_blocks = nn.Sequential(*[RepConv(C) for _ in range(N)])

    def forward(self, features_1: torch.Tensor, features_2: torch.Tensor) -> torch.Tensor:
        concat = torch.cat((features_1, features_2), dim=-3)
        upper_path = self.pointwise_conv_upper_path(concat)
        lower_path = self.pointwise_conv_lower_path(concat)
        lower_path = self.rep_blocks(lower_path)

        return upper_path + lower_path