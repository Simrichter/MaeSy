import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.linear(x)