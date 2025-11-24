"""Improved Lightweight CNN model for better CIFAR-100 accuracy."""
import torch
import torch.nn as nn
from .layers.conv_stem1 import ConvStem1
from .layers.conv_stem2 import ConvStem2
from .layers.pool_to_7 import PoolTo7
from .layers.conv_block1 import ConvBlock1
from .layers.conv_block2 import ConvBlock2
from .layers.conv_block3 import ConvBlock3
from .layers.global_avg_pool import GlobalAvgPool
from .layers.classifier import Classifier

class ImprovedLightweightCNN(nn.Module):
    """Improved CNN with more capacity for better CIFAR-100 accuracy."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()

        # Enhanced stem
        self.conv_stem1 = ConvStem1()
        self.conv_stem2 = ConvStem2()
        self.pool_to_7 = PoolTo7()

        # More conv blocks with higher capacity
        self.conv_block1 = ConvBlock1()
        self.conv_block2 = ConvBlock2()
        self.conv_block3 = ConvBlock3()

        # Additional conv blocks
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce to 3x3
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )

        # Enhanced classifier with dropout
        self.global_avg_pool = GlobalAvgPool()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the enhanced network."""
        # Stem
        x = self.conv_stem1(x)
        x = self.conv_stem2(x)
        x = self.pool_to_7(x)

        # Core conv blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Additional conv blocks
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        # Classification head
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.classifier(x)

        return x

def build_improved_lightweight_cnn(num_classes: int = 100) -> nn.Module:
    """Factory function to build an Improved LightweightCNN model."""
    return ImprovedLightweightCNN(num_classes)

if __name__ == "__main__":
    # Demo: Build improved model, print params/MACs
    model = build_improved_lightweight_cnn()

    # Count parameters
    from .utils import count_parameters
    total_params = count_parameters(model)
    print(f"Improved model parameters: {total_params:,}")

    # Compute MACs
    from .macs import compute_macs
    macs_dict = compute_macs(model)
    total_macs = sum(macs_dict.values())
    print(f"Improved model MACs: {total_macs:,}")

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
