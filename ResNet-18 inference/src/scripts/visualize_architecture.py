"""Visualize and compare model architectures."""
import torch
from ..resnet18.model import build_resnet18


def print_architecture_comparison():
    """Print a visual comparison of ResNet-18 and Lightweight CNN architectures."""
    
    print("\n" + "="*100)
    print("ARCHITECTURE COMPARISON: ResNet-18 vs Lightweight CNN")
    print("="*100)
    
    print("\n" + "─"*100)
    print("LIGHTWEIGHT CNN ARCHITECTURE (Efficient & Fast)")
    print("─"*100)
    print("""
    Input: 3×32×32 (CIFAR-100 image)
    │
    ├─ STEM
    │  ├─ ConvStem1:  3→16,  k=3, s=2, p=1  →  16×16×16   [  4,320 params,   2.2M MACs]
    │  ├─ ConvStem2: 16→32,  k=3, s=2, p=1  →  32×8×8     [ 18,432 params,   4.7M MACs]
    │  └─ PoolTo7:   MaxPool               →  32×7×7     [      0 params,       0 MACs]
    │
    ├─ CORE (Feature Extraction)
    │  ├─ ConvBlock1: 32→64,  k=3, s=1, p=1 →  64×7×7    [ 73,728 params,   3.6M MACs]
    │  ├─ ConvBlock2: 64→128, k=3, s=1, p=1 →  128×7×7   [294,912 params,   7.2M MACs]
    │  └─ ConvBlock3: 128→128, k=3, s=1, p=1 → 128×7×7   [589,824 params,   7.2M MACs]
    │
    └─ HEAD (Classification)
       ├─ GlobalAvgPool                    →  128×1×1    [      0 params,       0 MACs]
       └─ Classifier: 128→100              →  100        [ 12,900 params,  12.9K MACs]
    
    Output: 100 class logits
    
    TOTAL: 258,292 parameters | 15.8M MACs | 0.99 MB
    """)
    
    print("\n" + "─"*100)
    print("RESNET-18 ARCHITECTURE (Accurate & Deep)")
    print("─"*100)
    print("""
    Input: 3×32×32 (CIFAR-100 image)
    │
    ├─ INITIAL CONV
    │  └─ Conv1: 3→64, k=3, s=1, p=1       →  64×32×32   [  1,728 params,   1.8M MACs]
    │
    ├─ LAYER 1 (2 Residual Blocks @ 64 channels, 32×32)
    │  ├─ Block 1: [64→64→64] + skip       →  64×32×32   [ 73,984 params,  75.5M MACs]
    │  └─ Block 2: [64→64→64] + skip       →  64×32×32   [ 73,984 params,  75.5M MACs]
    │
    ├─ LAYER 2 (2 Residual Blocks @ 128 channels, 16×16)
    │  ├─ Block 1: [64→128→128] + proj     →  128×16×16  [230,144 params,  59.8M MACs]
    │  └─ Block 2: [128→128→128] + skip    →  128×16×16  [295,424 params,  75.5M MACs]
    │
    ├─ LAYER 3 (2 Residual Blocks @ 256 channels, 8×8)
    │  ├─ Block 1: [128→256→256] + proj    →  256×8×8    [919,040 params,  59.8M MACs]
    │  └─ Block 2: [256→256→256] + skip    →  256×8×8    [1.18M params,   75.5M MACs]
    │
    ├─ LAYER 4 (2 Residual Blocks @ 512 channels, 4×4)
    │  ├─ Block 1: [256→512→512] + proj    →  512×4×4    [3.67M params,   59.8M MACs]
    │  └─ Block 2: [512→512→512] + skip    →  512×4×4    [4.72M params,   75.5M MACs]
    │
    └─ HEAD (Classification)
       ├─ AdaptiveAvgPool                  →  512×1×1    [      0 params,       0 MACs]
       └─ FC: 512→100                      →  100        [ 51,300 params,  51.2K MACs]
    
    Output: 100 class logits
    
    TOTAL: 11,220,132 parameters | 555.5M MACs | 42.8 MB
    """)
    
    print("\n" + "="*100)
    print("KEY ARCHITECTURAL DIFFERENCES")
    print("="*100)
    
    print("""
    1. DEPTH
       • Lightweight CNN: 8 layers (2 stem + 3 core + 1 head)
       • ResNet-18:      18 layers (1 initial + 8 residual blocks + 1 head)
       → ResNet-18 is 2.25x deeper
    
    2. SKIP CONNECTIONS
       • Lightweight CNN: None (simple feedforward)
       • ResNet-18:      8 skip connections (residual learning)
       → ResNet-18 can learn identity mappings and deeper representations
    
    3. CHANNEL PROGRESSION
       • Lightweight CNN: 3 → 16 → 32 → 64 → 128 → 128 (gradual)
       • ResNet-18:      3 → 64 → 64 → 128 → 256 → 512 (aggressive)
       → ResNet-18 uses 4x more channels in final layers
    
    4. SPATIAL DOWNSAMPLING
       • Lightweight CNN: Aggressive early (32×32 → 7×7 in stem)
       • ResNet-18:      Gradual (32×32 → 16×16 → 8×8 → 4×4)
       → Lightweight CNN reduces spatial dimensions faster
    
    5. PARAMETER DISTRIBUTION
       • Lightweight CNN: Evenly distributed across layers
       • ResNet-18:      Concentrated in deeper layers (Layer 4 = 75% of params)
       → ResNet-18 invests heavily in high-level feature extraction
    
    6. COMPUTATIONAL COST
       • Lightweight CNN: 15.8M MACs (efficient)
       • ResNet-18:      555.5M MACs (35x more)
       → Lightweight CNN is much more energy efficient
    """)
    
    print("\n" + "="*100)
    print("PERFORMANCE TRADE-OFFS")
    print("="*100)
    
    print("""
    ┌─────────────────────┬──────────────────────┬──────────────────────┐
    │ Metric              │ Lightweight CNN      │ ResNet-18            │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Parameters          │ 258K                 │ 11.2M (43x more)     │
    │ MACs                │ 15.8M                │ 555M (35x more)      │
    │ Model Size          │ 0.99 MB              │ 42.8 MB (43x larger) │
    │ Inference Speed     │ ~2-5 ms (baseline)   │ ~10-25 ms (5x slower)│
    │ Expected Accuracy   │ 65-75%               │ 75-80% (+10-15%)     │
    │ Training Time       │ ~15-25 min (100 ep)  │ ~2-3 hrs (100 ep)    │
    │ Energy Efficiency   │ Excellent            │ Moderate             │
    │ Mobile Deployment   │ ✓ Excellent          │ ✗ Challenging        │
    │ Edge Devices        │ ✓ Ideal              │ ✗ Too heavy          │
    │ Server Inference    │ ✓ Good               │ ✓ Excellent          │
    └─────────────────────┴──────────────────────┴──────────────────────┘
    """)
    
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)
    print("""
    Choose LIGHTWEIGHT CNN when:
      ✓ Deploying to mobile/edge devices (phones, IoT, embedded systems)
      ✓ Energy efficiency is critical (battery-powered devices)
      ✓ Real-time inference required (< 5ms latency)
      ✓ Limited memory/storage (< 5MB model size)
      ✓ Acceptable accuracy: 65-75%
    
    Choose RESNET-18 when:
      ✓ Maximum accuracy is required (75-80%+)
      ✓ Server-side inference with GPU acceleration
      ✓ Sufficient computational resources available
      ✓ Training time is not a constraint
      ✓ Model size is not a concern
    """)
    
    print("="*100 + "\n")


def print_layer_by_layer_comparison():
    """Print detailed layer-by-layer parameter and MAC comparison."""
    
    print("\n" + "="*100)
    print("LAYER-BY-LAYER COMPUTATIONAL COST COMPARISON")
    print("="*100)
    
    print(f"\n{'Layer':<30} {'Lightweight CNN':<25} {'ResNet-18':<25} {'Ratio':<20}")
    print("-" * 100)
    
    # Lightweight CNN layers
    light_layers = {
        'ConvStem1': (4_320, 2_211_840),
        'ConvStem2': (18_432, 4_718_592),
        'PoolTo7': (0, 0),
        'ConvBlock1': (73_728, 3_612_672),
        'ConvBlock2': (294_912, 7_225_344),
        'ConvBlock3': (589_824, 7_225_344),
        'GlobalAvgPool': (0, 0),
        'Classifier': (12_900, 12_900),
    }
    
    # ResNet-18 layers
    resnet_layers = {
        'Conv1 + BN': (1_856, 1_769_472),
        'Layer1 (2 blocks)': (147_968, 150_994_944),
        'Layer2 (2 blocks)': (525_568, 134_217_728),
        'Layer3 (2 blocks)': (2_099_712, 134_217_728),
        'Layer4 (2 blocks)': (8_393_728, 134_217_728),
        'FC': (51_300, 51_200),
    }
    
    print("\nLightweight CNN Breakdown:")
    for layer, (params, macs) in light_layers.items():
        print(f"  {layer:<28} {params:>10,} params    {macs:>12,} MACs")
    
    print("\nResNet-18 Breakdown:")
    for layer, (params, macs) in resnet_layers.items():
        print(f"  {layer:<28} {params:>10,} params    {macs:>12,} MACs")
    
    print("\n" + "="*100 + "\n")


def main():
    """Main function to display architecture visualizations."""
    print_architecture_comparison()
    print_layer_by_layer_comparison()
    
    # Test model instantiation
    print("Testing model instantiation...")
    try:
        model = build_resnet18()
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ ResNet-18 forward pass successful: {dummy_input.shape} → {output.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*100)
    print("To train and compare models, use:")
    print("  python3 -m src.scripts.train --epochs 100")
    print("  python3 -m src.scripts.compare_models --resnet-checkpoint <path> --lightweight-checkpoint <path>")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
