"""Test forward pass shapes for Lightweight CNN."""
import torch
import pytest
from ..lightweight_cnn.model import build_lightweight_cnn

def test_forward_shape():
    """Test that the model produces the correct output shape."""
    model = build_lightweight_cnn(num_classes=100)

    # Create dummy input
    dummy_input = torch.randn(4, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape
    expected_shape = torch.Size([4, 100])
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_layer_shapes():
    """Test that each layer produces the expected intermediate shapes."""
    from ..lightweight_cnn.layers.conv_stem1 import ConvStem1
    from ..lightweight_cnn.layers.conv_stem2 import ConvStem2
    from ..lightweight_cnn.layers.pool_to_7 import PoolTo7
    from ..lightweight_cnn.layers.conv_block1 import ConvBlock1
    from ..lightweight_cnn.layers.conv_block2 import ConvBlock2
    from ..lightweight_cnn.layers.conv_block3 import ConvBlock3
    from ..lightweight_cnn.layers.global_avg_pool import GlobalAvgPool
    from ..lightweight_cnn.layers.classifier import Classifier

    x = torch.randn(2, 3, 32, 32)

    # Test stem layers
    stem1 = ConvStem1()
    x = stem1(x)
    assert x.shape == torch.Size([2, 16, 16, 16])

    stem2 = ConvStem2()
    x = stem2(x)
    assert x.shape == torch.Size([2, 32, 8, 8])

    pool = PoolTo7()
    x = pool(x)
    assert x.shape == torch.Size([2, 32, 7, 7])

    # Test core layers
    block1 = ConvBlock1()
    x = block1(x)
    assert x.shape == torch.Size([2, 64, 7, 7])

    block2 = ConvBlock2()
    x = block2(x)
    assert x.shape == torch.Size([2, 128, 7, 7])

    block3 = ConvBlock3()
    x = block3(x)
    assert x.shape == torch.Size([2, 128, 7, 7])

    # Test head layers
    gap = GlobalAvgPool()
    x = gap(x)
    assert x.shape == torch.Size([2, 128, 1, 1])

    classifier = Classifier(100)
    x = classifier(x)
    assert x.shape == torch.Size([2, 100])

if __name__ == '__main__':
    test_forward_shape()
    test_layer_shapes()
    print("All shape tests passed!")
