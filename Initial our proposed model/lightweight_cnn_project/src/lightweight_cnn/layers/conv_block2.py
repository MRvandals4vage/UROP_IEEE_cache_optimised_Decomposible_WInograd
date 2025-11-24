"""ConvBlock2: Conv2d(64,128,k=3,s=1,p=1) + BatchNorm + ReLU
Input: (B,64,7,7) -> Output: (B,128,7,7)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2(nn.Module):
    """Second core convolutional block."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv -> bn -> relu."""
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)
