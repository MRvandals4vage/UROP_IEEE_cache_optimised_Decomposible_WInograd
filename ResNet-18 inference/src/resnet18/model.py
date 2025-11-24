"""ResNet-18 model for CIFAR-100 classification.

This implementation is adapted for CIFAR-100's 32×32 images.
Unlike the ImageNet version, we use a smaller initial conv layer
and no max pooling to preserve spatial resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34.
    
    Structure:
        3×3 conv -> BN -> ReLU -> 3×3 conv -> BN -> (+shortcut) -> ReLU
    """
    expansion = 1  # Output channels = input channels × expansion
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        
        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (identity or projection)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet18(nn.Module):
    """ResNet-18 architecture adapted for CIFAR-100.
    
    Architecture:
        - Initial conv: 3×3, 64 channels (no 7×7 conv or max pool)
        - Stage 1: 2 blocks, 64 channels
        - Stage 2: 2 blocks, 128 channels, stride=2
        - Stage 3: 2 blocks, 256 channels, stride=2
        - Stage 4: 2 blocks, 512 channels, stride=2
        - Global average pooling
        - Fully connected layer (512 -> num_classes)
    """
    
    def __init__(self, num_classes: int = 100):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution (adapted for CIFAR-100's 32×32 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual stages
        self.layer1 = self._make_layer(64, 2, stride=1)   # 32×32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 16×16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 8×8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 4×4
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a residual stage with multiple blocks.
        
        Args:
            out_channels: Number of output channels
            num_blocks: Number of residual blocks in this stage
            stride: Stride for the first block (for downsampling)
        
        Returns:
            Sequential container of residual blocks
        """
        downsample = None
        
        # Create projection shortcut if dimensions change
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18.
        
        Args:
            x: Input tensor of shape (B, 3, 32, 32)
        
        Returns:
            Output logits of shape (B, num_classes)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def build_resnet18(num_classes: int = 100) -> nn.Module:
    """Factory function to build ResNet-18 model.
    
    Args:
        num_classes: Number of output classes (default: 100 for CIFAR-100)
    
    Returns:
        ResNet-18 model
    """
    return ResNet18(num_classes)


if __name__ == "__main__":
    # Demo: Build model and test forward pass
    model = build_resnet18()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    # Dummy forward pass
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-100 images
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: torch.Size([4, 100])")
        print(f"Output shape matches: {output.shape == torch.Size([4, 100])}")
