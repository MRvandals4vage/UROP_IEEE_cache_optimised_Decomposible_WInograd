"""ConvBlock1: Conv2d(32,64,k=3,s=1,p=1) + BatchNorm + ReLU
Input: (B,32,7,7) -> Output: (B,64,7,7)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1(nn.Module):
    """First core convolutional block."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv -> bn -> relu."""
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)
