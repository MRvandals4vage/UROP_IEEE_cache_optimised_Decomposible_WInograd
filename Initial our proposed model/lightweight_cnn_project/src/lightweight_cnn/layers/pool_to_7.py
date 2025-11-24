"""PoolTo7: MaxPool2d(kernel=2, stride=1)
Input: (B,32,8,8) -> Output: (B,32,7,7)
"""
import torch
import torch.nn as nn

class PoolTo7(nn.Module):
    """Max pooling layer to reduce spatial dimensions to 7x7."""
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply max pooling."""
        return self.pool(x)
