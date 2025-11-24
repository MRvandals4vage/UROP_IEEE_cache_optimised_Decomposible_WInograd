"""Test parameter counting utility."""
import torch
from ..lightweight_cnn.model import build_lightweight_cnn
from ..lightweight_cnn.utils import count_parameters

def test_count_parameters():
    """Test that parameter counting works correctly."""
    model = build_lightweight_cnn()

    # Count parameters
    total_params = count_parameters(model)

    # Should have more than 0 parameters
    assert total_params > 0, "Model should have trainable parameters"

    # Should have reasonable number of parameters (less than 10M for a lightweight model)
    assert total_params < 10_000_000, "Model should be lightweight (<10M parameters)"

    # Check that we can count parameters for individual layers
    conv_params = sum(p.numel() for name, p in model.named_parameters() if 'conv' in name and p.requires_grad)
    assert conv_params > 0, "Should have convolutional parameters"

    print(f"Total parameters: {total_params:,}")

if __name__ == '__main__':
    test_count_parameters()
    print("Parameter counting test passed!")
