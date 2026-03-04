"""
Model definition: ResNet18 adapted for CIFAR-10.
"""


import torch.nn as nn
from torchvision.models import resnet18


class SmallCNN(nn.Module):
    """
    ResNet18 adapted for CIFAR-10 (32x32 images).

    Args:
        n_classes: number of output classes (default: 10 for CIFAR-10)
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        self.net = m

    def forward(self, x):
        """
        Args:
            x: input tensor, shape (B, 3, 32, 32)

        Returns:
            logits tensor, shape (B, n_classes)
        """
        return self.net(x)
