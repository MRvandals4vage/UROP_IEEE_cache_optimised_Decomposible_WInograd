"""ConvStem1: Conv2d(3,16,k=3,s=2,p=1) + BatchNorm + ReLU
Input: (B,3,32,32) -> Output: (B,16,16,16)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvStem1(nn.Module):
    """First convolutional stem layer."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv -> bn -> relu."""
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)
