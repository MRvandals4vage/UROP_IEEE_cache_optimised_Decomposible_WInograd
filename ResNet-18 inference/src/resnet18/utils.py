"""Utility functions for ResNet-18."""
import torch
import torch.nn as nn
from typing import Dict


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Count parameters for each named module in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    param_dict = {}
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        param_dict[name] = num_params
    return param_dict


def model_summary(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> None:
    """Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (default: CIFAR-100 image)
    """
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    
    # Total parameters
    total_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    
    # Parameters by layer
    print("\nParameters by layer:")
    print(f"{'Layer':<20} {'Parameters':<15}")
    print("-" * 40)
    
    param_dict = count_parameters_by_layer(model)
    for name, params in param_dict.items():
        print(f"{name:<20} {params:>12,}")
    
    # Test forward pass
    print("\n" + "=" * 80)
    print("Testing forward pass...")
    dummy_input = torch.randn(*input_size)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    print("=" * 80)
