"""Script to measure model parameters and MACs."""
from ..lightweight_cnn.model import build_lightweight_cnn
from ..lightweight_cnn.utils import count_parameters
from ..lightweight_cnn.macs import compute_macs

def main():
    # Build model
    model = build_lightweight_cnn()

    # Count parameters
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")

    # Compute MACs
    macs_dict = compute_macs(model)
    total_macs = sum(macs_dict.values())
    print(f"Total MACs: {total_macs:,}")

    # Print per-layer breakdown
    print("\nPer-layer breakdown:")
    print(f"{{'Layer':<20}} {{'Parameters':<12}} {{'MACs':<12}}")
    print("-" * 50)

    for name, macs in macs_dict.items():
        # Count parameters for this layer
        layer_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith(name) and p.requires_grad)
        print(f"{{name:<20}} {{layer_params:<12}} {{macs:<12}}")

    # Total row
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_params:<12} {total_macs:<12}")

if __name__ == '__main__':
    main()
