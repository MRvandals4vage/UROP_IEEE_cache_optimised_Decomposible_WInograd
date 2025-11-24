"""Utility functions for the Lightweight CNN project."""
import torch
import torch.nn as nn
import random
import numpy as np

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in the model.

    Args:
        model: The PyTorch model to count parameters for.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
