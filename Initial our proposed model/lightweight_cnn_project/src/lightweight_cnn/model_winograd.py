"""Winograd-enhanced LightweightCNN model.

This model uses Winograd F(2x2, 3x3) convolution for the 3x3 convolution layers
to reduce computational complexity by ~2.25x compared to standard convolution.
"""
import torch
import torch.nn as nn
from .layers.conv_stem1 import ConvStem1
from .layers.conv_stem2 import ConvStem2
from .layers.pool_to_7 import PoolTo7
from .layers.winograd_conv_block1 import WinogradConvBlock1
from .layers.winograd_conv_block2 import WinogradConvBlock2
from .layers.winograd_conv_block3 import WinogradConvBlock3
from .layers.global_avg_pool import GlobalAvgPool
from .layers.classifier import Classifier

class LightweightCNNWinograd(nn.Module):
    """Lightweight CNN with Winograd transform for 3x3 convolutions.
    
    Architecture:
    - Stem layers: Standard Conv2d (stride=2, less benefit from Winograd)
    - Core layers: Winograd Conv2d (stride=1, optimal for Winograd)
    - Head layers: Global average pooling + classifier
    
    Winograd is applied to ConvBlock1, ConvBlock2, and ConvBlock3 which use
    3x3 convolutions with stride=1, providing ~2.25x reduction in multiplications.
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        # Stem layers (stride=2, standard convolution is fine)
        self.conv_stem1 = ConvStem1()
        self.conv_stem2 = ConvStem2()
        self.pool_to_7 = PoolTo7()

        # Core layers with Winograd optimization
        self.conv_block1 = WinogradConvBlock1()
        self.conv_block2 = WinogradConvBlock2()
        self.conv_block3 = WinogradConvBlock3()

        # Head layers
        self.global_avg_pool = GlobalAvgPool()
        self.classifier = Classifier(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Stem
        x = self.conv_stem1(x)
        x = self.conv_stem2(x)
        x = self.pool_to_7(x)

        # Core (with Winograd)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Head
        x = self.global_avg_pool(x)
        x = self.classifier(x)

        return x

def build_lightweight_cnn_winograd(num_classes: int = 100) -> nn.Module:
    """Factory function to build a Winograd-enhanced LightweightCNN model.

    Args:
        num_classes: Number of output classes.

    Returns:
        The constructed LightweightCNN model with Winograd convolutions.
    """
    return LightweightCNNWinograd(num_classes)

if __name__ == "__main__":
    # Demo: Build model, print params/MACs, run dummy forward pass
    model = build_lightweight_cnn_winograd()

    # Count parameters
    from .utils import count_parameters
    total_params = count_parameters(model)
    print(f"Total parameters (Winograd): {total_params:,}")

    # Compute MACs (theoretical - Winograd reduces multiplications)
    from .macs import compute_macs
    macs_dict = compute_macs(model)
    total_macs = sum(macs_dict.values())
    print(f"Total MACs (Winograd): {total_macs:,}")

    # Print per-layer breakdown
    print("\nPer-layer MACs (Winograd):")
    for name, macs in macs_dict.items():
        print(f"  {name}: {macs:,}")

    # Dummy forward pass
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nWinograd model output shape: {output.shape}")
        print(f"Expected shape: torch.Size([4, 100])")
        print(f"Output shape matches: {output.shape == torch.Size([4, 100])}")
