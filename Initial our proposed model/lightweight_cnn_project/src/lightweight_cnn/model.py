"""LightweightCNN model composition."""
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

class LightweightCNN(nn.Module):
    """Lightweight CNN for CIFAR-100 classification."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        # Stem layers
        self.conv_stem1 = ConvStem1()
        self.conv_stem2 = ConvStem2()
        self.pool_to_7 = PoolTo7()

        # Core layers
        self.conv_block1 = ConvBlock1()
        self.conv_block2 = ConvBlock2()
        self.conv_block3 = ConvBlock3()

        # Head layers
        self.global_avg_pool = GlobalAvgPool()
        self.classifier = Classifier(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Stem
        x = self.conv_stem1(x)
        x = self.conv_stem2(x)
        x = self.pool_to_7(x)

        # Core
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Head
        x = self.global_avg_pool(x)
        x = self.classifier(x)

        return x

def build_lightweight_cnn(num_classes: int = 100) -> nn.Module:
    """Factory function to build a LightweightCNN model.

    Args:
        num_classes: Number of output classes.

    Returns:
        The constructed LightweightCNN model.
    """
    return LightweightCNN(num_classes)

if __name__ == "__main__":
    # Demo: Build model, print params/MACs, run dummy forward pass
    model = build_lightweight_cnn()

    # Count parameters
    from .utils import count_parameters
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")

    # Compute MACs
    from .macs import compute_macs
    macs_dict = compute_macs(model)
    total_macs = sum(macs_dict.values())
    print(f"Total MACs: {total_macs:,}")

    # Print per-layer breakdown
    print("\nPer-layer MACs:")
    for name, macs in macs_dict.items():
        print(f"  {name}: {macs:,}")

    # Dummy forward pass
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nDummy forward pass output shape: {output.shape}")
        print(f"Expected shape: torch.Size([4, 100])")
        print(f"Output shape matches: {output.shape == torch.Size([4, 100])}")
