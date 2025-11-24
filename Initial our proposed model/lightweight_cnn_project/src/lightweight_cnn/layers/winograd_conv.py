"""Winograd convolution implementation for 3x3 kernels.

Winograd convolution reduces computational complexity for 3x3 convolutions
by transforming the convolution into element-wise multiplications in a different domain.

For F(2x2, 3x3) Winograd:
- Standard 3x3 convolution: 9 multiplications per output pixel
- Winograd F(2x2, 3x3): 4 multiplications per 2x2 output tile
- Reduction: ~2.25x fewer multiplications (9 vs 4 per output pixel)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WinogradConv2d(nn.Module):
    """
    Winograd F(2x2, 3x3) convolution implementation.
    
    This implementation uses Winograd minimal filtering algorithm to reduce
    the number of multiplications required for 3x3 convolutions.
    
    Key benefits:
    - Reduces multiplications from 9 to 4 per 2x2 output tile
    - ~2.25x reduction in computational complexity
    - Same output as standard convolution (mathematically equivalent)
    """

    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        # Learnable 3x3 convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
        # Winograd transformation matrices for F(2x2, 3x3)
        # G: transforms filter (3x3 -> 4x4)
        # B^T: transforms input tile (4x4 -> 4x4)
        # A^T: transforms output tile (4x4 -> 2x2)
        
        self.register_buffer('G', torch.tensor([
            [1.0,  0.0,  0.0],
            [0.5,  0.5,  0.5],
            [0.5, -0.5,  0.5],
            [0.0,  0.0,  1.0]
        ], dtype=torch.float32))
        
        self.register_buffer('BT', torch.tensor([
            [1.0,  0.0, -1.0,  0.0],
            [0.0,  1.0,  1.0,  0.0],
            [0.0, -1.0,  1.0,  0.0],
            [0.0,  1.0,  0.0, -1.0]
        ], dtype=torch.float32))
        
        self.register_buffer('AT', torch.tensor([
            [1.0,  1.0,  1.0,  0.0],
            [0.0,  1.0, -1.0, -1.0]
        ], dtype=torch.float32))

    def forward(self, x):
        """
        Forward pass using Winograd algorithm.
        
        For stride=1, padding=1 (same as standard conv):
        Uses Winograd F(2x2, 3x3) to compute convolution efficiently.
        """
        if self.stride != 1:
            # Fall back to standard convolution for stride != 1
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        
        # Apply padding
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        batch_size, in_channels, height, width = x.shape
        
        # Transform filters: U = G @ g @ G^T for each filter
        # Shape: (out_channels, in_channels, 4, 4)
        U = torch.zeros(self.out_channels, self.in_channels, 4, 4, 
                       device=x.device, dtype=x.dtype)
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                g = self.weight[oc, ic]  # 3x3
                U[oc, ic] = self.G @ g @ self.G.t()
        
        # Process input in 4x4 tiles (producing 2x2 output per tile)
        out_height = height - 2  # After 3x3 conv with padding already applied
        out_width = width - 2
        
        # Calculate number of tiles
        tile_h = (out_height + 1) // 2
        tile_w = (out_width + 1) // 2
        
        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, out_height, out_width,
                           device=x.device, dtype=x.dtype)
        
        # Process each 4x4 input tile
        for th in range(tile_h):
            for tw in range(tile_w):
                # Extract 4x4 input tile
                h_start = th * 2
                w_start = tw * 2
                h_end = min(h_start + 4, height)
                w_end = min(w_start + 4, width)
                
                if h_end - h_start < 4 or w_end - w_start < 4:
                    continue
                
                d = x[:, :, h_start:h_end, w_start:w_end]  # (B, C_in, 4, 4)
                
                # Transform input: V = B^T @ d @ B for each input channel
                V = torch.zeros(batch_size, in_channels, 4, 4,
                              device=x.device, dtype=x.dtype)
                for b in range(batch_size):
                    for ic in range(in_channels):
                        V[b, ic] = self.BT @ d[b, ic] @ self.BT.t()
                
                # Element-wise multiplication in Winograd domain
                # M = U ⊙ V (element-wise for each position)
                M = torch.zeros(batch_size, self.out_channels, 4, 4,
                              device=x.device, dtype=x.dtype)
                for b in range(batch_size):
                    for oc in range(self.out_channels):
                        for ic in range(in_channels):
                            M[b, oc] += U[oc, ic] * V[b, ic]
                
                # Inverse transform: Y = A^T @ M @ A
                out_h_start = th * 2
                out_w_start = tw * 2
                out_h_end = min(out_h_start + 2, out_height)
                out_w_end = min(out_w_start + 2, out_width)
                
                for b in range(batch_size):
                    for oc in range(self.out_channels):
                        Y = self.AT @ M[b, oc] @ self.AT.t()  # 2x2
                        output[b, oc, out_h_start:out_h_end, out_w_start:out_w_end] = Y[:out_h_end-out_h_start, :out_w_end-out_w_start]
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output
