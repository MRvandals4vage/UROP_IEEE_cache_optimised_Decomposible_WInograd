"""GlobalAvgPool: AdaptiveAvgPool2d((1,1))
Input: (B,128,7,7) -> Output: (B,128,1,1)
"""
import torch
import torch.nn as nn

class GlobalAvgPool(nn.Module):
    """Global average pooling to reduce spatial dimensions."""
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive average pooling."""
        return self.pool(x)
