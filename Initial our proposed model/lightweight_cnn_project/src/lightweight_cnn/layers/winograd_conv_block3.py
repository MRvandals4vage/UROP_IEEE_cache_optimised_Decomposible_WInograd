"""ConvBlock3 with Winograd: WinogradConv2d(128,128,k=3,s=1,p=1) + BatchNorm + ReLU
Input: (B,128,7,7) -> Output: (B,128,7,7)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .winograd_conv import WinogradConv2d

class WinogradConvBlock3(nn.Module):
    """Third core convolutional block with Winograd transform."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = WinogradConv2d(128, 128, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply winograd conv -> bn -> relu."""
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)
