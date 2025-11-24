"""Compute Multiply-Accumulate operations (MACs) for ResNet-18."""
import torch
import torch.nn as nn
from typing import Dict


def compute_conv2d_macs(in_channels: int, out_channels: int, kernel_size: int,
                        input_h: int, input_w: int, stride: int = 1, padding: int = 0) -> int:
    """Compute MACs for a Conv2d layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size (assumes square kernel)
        input_h: Input height
        input_w: Input width
        stride: Stride
        padding: Padding
    
    Returns:
        Number of MACs
    """
    output_h = (input_h + 2 * padding - kernel_size) // stride + 1
    output_w = (input_w + 2 * padding - kernel_size) // stride + 1
    
    # MACs per output position
    macs_per_position = in_channels * kernel_size * kernel_size
    
    # Total MACs
    total_macs = macs_per_position * output_h * output_w * out_channels
    
    return total_macs


def compute_linear_macs(in_features: int, out_features: int) -> int:
    """Compute MACs for a Linear layer.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
    
    Returns:
        Number of MACs
    """
    return in_features * out_features


def compute_macs(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> Dict[str, int]:
    """Compute total MACs for ResNet-18 on CIFAR-100.
    
    This is a simplified calculation that manually computes MACs for each layer.
    
    Args:
        model: ResNet-18 model
        input_size: Input tensor size (B, C, H, W)
    
    Returns:
        Dictionary mapping layer names to MAC counts
    """
    macs_dict = {}
    
    # Initial conv: 3×3, stride=1, padding=1
    # Input: (3, 32, 32) -> Output: (64, 32, 32)
    macs_dict['conv1'] = compute_conv2d_macs(3, 64, 3, 32, 32, stride=1, padding=1)
    
    # Layer 1: 2 blocks, 64 channels, 32×32
    # Each block has two 3×3 convs
    macs_dict['layer1'] = 2 * 2 * compute_conv2d_macs(64, 64, 3, 32, 32, stride=1, padding=1)
    
    # Layer 2: 2 blocks, 128 channels
    # First block: 64->128 with stride=2 (32×32 -> 16×16), plus 1×1 downsample conv
    # Second block: 128->128 at 16×16
    macs_layer2 = compute_conv2d_macs(64, 128, 3, 32, 32, stride=2, padding=1)  # First conv of first block
    macs_layer2 += compute_conv2d_macs(128, 128, 3, 16, 16, stride=1, padding=1)  # Second conv of first block
    macs_layer2 += compute_conv2d_macs(64, 128, 1, 32, 32, stride=2, padding=0)  # Downsample
    macs_layer2 += 2 * compute_conv2d_macs(128, 128, 3, 16, 16, stride=1, padding=1)  # Second block
    macs_dict['layer2'] = macs_layer2
    
    # Layer 3: 2 blocks, 256 channels
    # First block: 128->256 with stride=2 (16×16 -> 8×8), plus 1×1 downsample conv
    # Second block: 256->256 at 8×8
    macs_layer3 = compute_conv2d_macs(128, 256, 3, 16, 16, stride=2, padding=1)  # First conv of first block
    macs_layer3 += compute_conv2d_macs(256, 256, 3, 8, 8, stride=1, padding=1)  # Second conv of first block
    macs_layer3 += compute_conv2d_macs(128, 256, 1, 16, 16, stride=2, padding=0)  # Downsample
    macs_layer3 += 2 * compute_conv2d_macs(256, 256, 3, 8, 8, stride=1, padding=1)  # Second block
    macs_dict['layer3'] = macs_layer3
    
    # Layer 4: 2 blocks, 512 channels
    # First block: 256->512 with stride=2 (8×8 -> 4×4), plus 1×1 downsample conv
    # Second block: 512->512 at 4×4
    macs_layer4 = compute_conv2d_macs(256, 512, 3, 8, 8, stride=2, padding=1)  # First conv of first block
    macs_layer4 += compute_conv2d_macs(512, 512, 3, 4, 4, stride=1, padding=1)  # Second conv of first block
    macs_layer4 += compute_conv2d_macs(256, 512, 1, 8, 8, stride=2, padding=0)  # Downsample
    macs_layer4 += 2 * compute_conv2d_macs(512, 512, 3, 4, 4, stride=1, padding=1)  # Second block
    macs_dict['layer4'] = macs_layer4
    
    # Fully connected layer: 512 -> 100
    macs_dict['fc'] = compute_linear_macs(512, 100)
    
    return macs_dict


def print_macs_summary(model: nn.Module) -> None:
    """Print a summary of MACs for the model.
    
    Args:
        model: ResNet-18 model
    """
    macs_dict = compute_macs(model)
    total_macs = sum(macs_dict.values())
    
    print("=" * 80)
    print("MACs (Multiply-Accumulate Operations) Summary")
    print("=" * 80)
    
    print(f"\n{'Layer':<20} {'MACs':<20} {'Percentage':<15}")
    print("-" * 60)
    
    for name, macs in macs_dict.items():
        percentage = (macs / total_macs) * 100
        print(f"{name:<20} {macs:>15,}    {percentage:>6.2f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_macs:>15,}    {100.0:>6.2f}%")
    print("=" * 80)
    
    # Comparison with Lightweight CNN
    lightweight_macs = 15_753_728
    ratio = total_macs / lightweight_macs
    print(f"\nComparison with Lightweight CNN:")
    print(f"  Lightweight CNN MACs: {lightweight_macs:,}")
    print(f"  ResNet-18 MACs:       {total_macs:,}")
    print(f"  Ratio (ResNet/Light): {ratio:.1f}x more MACs")
    print("=" * 80)
