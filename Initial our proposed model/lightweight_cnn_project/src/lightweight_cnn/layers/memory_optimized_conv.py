"""Memory-optimized convolution with tiling and on-chip buffering."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MemoryOptimizedConv2d(nn.Module):
    """
    Memory-optimized 2D convolution with tiling and on-chip buffering.
    
    Implements two key optimizations:
    1. Tiling with overlap: Processes input in smaller tiles with overlap to reduce
       off-chip memory accesses while maintaining accuracy.
    2. On-chip feature map buffering: Caches frequently accessed feature maps in on-chip memory.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel (default: 3)
        stride: Stride of the convolution (default: 1)
        padding: Padding added to all sides (default: 1)
        bias: If True, adds a learnable bias (default: False)
        tile_size: Size of the processing tile (default: 32)
        buffer_size: Size of the on-chip buffer in number of channels (default: 32)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        tile_size: int = 32,
        buffer_size: int = 32
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tile_size = tile_size
        self.buffer_size = buffer_size
        
        # Standard convolution parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
        # Buffer for on-chip feature maps (simulated with register buffer)
        self.register_buffer('feature_buffer', None)
        self.current_buffer_channels = 0
        
    def _tiled_conv2d(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        tile_size: int,
        overlap: int = 2
    ) -> torch.Tensor:
        """
        Apply convolution with tiling and overlap.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            weight: Convolution weights (out_channels, in_channels, kH, kW)
            bias: Optional bias tensor (out_channels,)
            tile_size: Size of the processing tile
            overlap: Overlap between tiles to handle boundary effects
            
        Returns:
            Output tensor after tiled convolution
        """
        batch_size, in_channels, height, width = x.size()
        out_channels = weight.size(0)
        
        # Calculate output dimensions
        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output tensor
        output = x.new_zeros((batch_size, out_channels, out_h, out_w))
        
        # Calculate number of tiles in each dimension
        h_tiles = (height + tile_size - 1) // tile_size
        w_tiles = (width + tile_size - 1) // tile_size
        
        # Pad input to handle border cases
        padded_x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        for h in range(h_tiles):
            for w in range(w_tiles):
                # Calculate tile boundaries with overlap
                h_start = h * tile_size
                w_start = w * tile_size
                h_end = min(h_start + tile_size + 2 * overlap, height)
                w_end = min(w_start + tile_size + 2 * overlap, width)
                
                # Extract tile from input
                tile = padded_x[:, :, h_start:h_end, w_start:w_end]
                
                # Apply convolution to tile
                tile_out = F.conv2d(
                    tile, weight, bias=bias,
                    stride=self.stride, padding=0, dilation=1, groups=1
                )
                
                # Calculate output boundaries
                out_h_start = (h * tile_size) // self.stride
                out_w_start = (w * tile_size) // self.stride
                out_h_end = out_h_start + tile_out.size(2)
                out_w_end = out_w_start + tile_out.size(3)
                
                # Add tile output to corresponding position in output
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = tile_out

        return output

    def _update_feature_buffer(self, x: torch.Tensor) -> None:
        """
        Update the on-chip feature map buffer with new data.

        Args:
            x: Input tensor to update the buffer with
        """
        # This is a simplified simulation - in hardware, this would be on-chip SRAM
        if self.feature_buffer is None or self.feature_buffer.size(0) != x.size(0):
            self.feature_buffer = x.new_zeros((x.size(0), self.buffer_size, x.size(2), x.size(3)))
            self.current_buffer_channels = 0

        # If we have space in the buffer, add the new features
        remaining_space = self.buffer_size - self.current_buffer_channels
        if remaining_space > 0:
            add_channels = min(remaining_space, x.size(1))
            self.feature_buffer[:, self.current_buffer_channels:self.current_buffer_channels+add_channels] = x[:, :add_channels]
            self.current_buffer_channels += add_channels

    def _get_buffered_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from the on-chip buffer if available, otherwise use input.
        """
        if self.feature_buffer is not None and self.current_buffer_channels > 0:
            # Use buffered features if available
            return self.feature_buffer[:, :self.current_buffer_channels]
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update feature buffer with input
        self._update_feature_buffer(x)
        
        # Get features from buffer (or input if buffer is empty)
        features = self._get_buffered_features(x)
        
        # Apply tiled convolution
        output = self._tiled_conv2d(
            features, self.weight, self.bias,
            tile_size=self.tile_size
        )
        
        # Reset buffer if this is the last layer in the block
        self.current_buffer_channels = 0
        
        return output
        
    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, padding={padding}')
        if self.bias is None:
            s += ', bias=False'
        s += ', tile_size={tile_size}, buffer_size={buffer_size}'
        return s.format(**self.__dict__)
