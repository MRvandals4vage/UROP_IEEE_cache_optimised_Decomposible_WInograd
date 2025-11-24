"""Classifier: Linear(128, 100)
Input: (B,128,1,1) -> Output: (B,100)
"""
import torch
import torch.nn as nn

class Classifier(nn.Module):
    """Final linear classifier for CIFAR-100."""
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation."""
        # Flatten the input from (B, 128, 1, 1) to (B, 128)
        x = x.view(x.size(0), -1)
        return self.fc(x)
