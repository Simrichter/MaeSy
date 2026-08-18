from torch import nn
from torchvision.ops import FrozenBatchNorm2d


def replace_bn_with_frozenbn(module):
    """
        Recursively replace all nn.BatchNorm2d layers in the given module with FrozenBatchNorm2d layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen_bn = FrozenBatchNorm2d(child.num_features)

            # copy weights
            frozen_bn.weight.data = child.weight.data.clone()
            frozen_bn.bias.data = child.bias.data.clone()
            frozen_bn.running_mean.data = child.running_mean.data.clone()
            frozen_bn.running_var.data = child.running_var.data.clone()

            setattr(module, name, frozen_bn)
        else:
            replace_bn_with_frozenbn(child)