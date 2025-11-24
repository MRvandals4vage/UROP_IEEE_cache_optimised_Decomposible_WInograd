"""MACs computation utilities for the Lightweight CNN project."""
import torch
import torch.nn as nn
from typing import Dict, Tuple

def compute_conv_macs(module: nn.Conv2d, input_shape: Tuple[int, int, int, int]) -> int:
    """Compute MACs for a Conv2d layer.

    Args:
        module: The Conv2d layer.
        input_shape: Shape of the input tensor (B, C, H, W).

    Returns:
        Number of multiply-accumulate operations.
    """
    B, C_in, H_in, W_in = input_shape
    C_out, _, K_h, K_w = module.weight.shape

    # Output spatial dimensions
    H_out = (H_in - K_h + 2 * module.padding[0]) // module.stride[0] + 1
    W_out = (W_in - K_w + 2 * module.padding[1]) // module.stride[1] + 1

    # MACs = H_out * W_out * C_out * (C_in * K_h * K_w)
    # Note: We count only the multiplies, not the adds
    return H_out * W_out * C_out * (C_in * K_h * K_w)

def compute_linear_macs(module: nn.Linear, input_shape: Tuple[int, int]) -> int:
    """Compute MACs for a Linear layer.

    Args:
        module: The Linear layer.
        input_shape: Shape of the input tensor (B, in_features).

    Returns:
        Number of multiply-accumulate operations.
    """
    B, in_features = input_shape
    out_features = module.out_features

    # MACs = in_features * out_features
    return in_features * out_features

def compute_macs(model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32)) -> Dict[str, int]:
    """Compute MACs for each layer in the model.

    Args:
        model: The PyTorch model to analyze.
        input_shape: Shape of the input tensor for the model.

    Returns:
        Dictionary mapping layer names to their MAC counts.
    """
    macs_dict = {}
    current_shape = input_shape

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            macs = compute_conv_macs(module, current_shape)
            macs_dict[name] = macs

            # Update shape for next layer
            B, C_in, H_in, W_in = current_shape
            C_out = module.out_features if hasattr(module, 'out_features') else module.weight.shape[0]
            H_out = (H_in - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0] + 1
            W_out = (W_in - module.kernel_size[1] + 2 * module.padding[1]) // module.stride[1] + 1
            current_shape = (B, C_out, H_out, W_out)

        elif isinstance(module, nn.Linear):
            macs = compute_linear_macs(module, current_shape[:2])
            macs_dict[name] = macs

            # Update shape for next layer
            B, in_features = current_shape[:2]
            out_features = module.out_features
            current_shape = (B, out_features)

    return macs_dict
