"""Memory-optimized version of the lightweight CNN with tiling and on-chip buffering."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .layers.conv_stem1 import ConvStem1
from .layers.conv_stem2 import ConvStem2
from .layers.pool_to_7 import PoolTo7
from .layers.global_avg_pool import GlobalAvgPool
from .layers.classifier import Classifier
from .layers.memory_optimized_conv import MemoryOptimizedConv2d

class MemoryOptimizedCNN(nn.Module):
    """
    Memory-optimized CNN combining Winograd convolution with tiling and on-chip buffering.
    
    This model extends the baseline LightweightCNN with additional memory optimizations:
    1. Tiling with overlap for reduced off-chip memory access
    2. On-chip feature map buffering to minimize DRAM accesses
    """
    
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        
        # Stem layers (initial feature extraction)
        self.conv_stem1 = ConvStem1()
        self.conv_stem2 = ConvStem2()
        self.pool_to_7 = PoolTo7()
        
        # Core layers with memory optimizations
        self.conv_block1 = self._make_conv_block(64, 64, 3, 1, 1)
        self.conv_block2 = self._make_conv_block(64, 128, 3, 2, 1)  # Downsample
        self.conv_block3 = self._make_conv_block(128, 256, 3, 2, 1)  # Downsample
        
        # Head layers
        self.global_avg_pool = GlobalAvgPool()
        self.classifier = Classifier(num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_conv_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ) -> nn.Sequential:
        """Create a memory-optimized convolution block."""
        return nn.Sequential(
            # First convolution with tiling and buffering
            MemoryOptimizedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                tile_size=32,  # Can be tuned based on target hardware
                buffer_size=64  # Number of channels to buffer on-chip
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution (depthwise separable)
            MemoryOptimizedConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=out_channels,  # Depthwise convolution
                tile_size=32,
                buffer_size=64
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            MemoryOptimizedConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                tile_size=32,
                buffer_size=64
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Stem
        x = self.conv_stem1(x)
        x = self.conv_stem2(x)
        x = self.pool_to_7(x)
        
        # Core with memory optimizations
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Head
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        
        return x

def build_memory_optimized_cnn(num_classes: int = 100) -> nn.Module:
    """
    Build and return a memory-optimized CNN model.
    
    Args:
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        
    Returns:
        Configured MemoryOptimizedCNN model
    """
    return MemoryOptimizedCNN(num_classes=num_classes)

if __name__ == "__main__":
    # Test the model with a dummy input
    model = build_memory_optimized_cnn()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch of 1, 3 channels, 32x32 images
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
