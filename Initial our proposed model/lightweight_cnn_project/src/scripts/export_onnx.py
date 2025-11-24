"""Export script for Lightweight CNN to ONNX and TorchScript."""
import torch
import argparse

def export_model(model: torch.nn.Module, checkpoint_path: str = None, output_dir: str = '.'):
    """Export model to ONNX and TorchScript formats.

    Args:
        model: The model to export.
        checkpoint_path: Path to model checkpoint (optional).
        output_dir: Directory to save exported models.
    """
    # Load checkpoint if provided
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded checkpoint from {checkpoint_path}")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32)

    # Export to ONNX
    onnx_path = f"{output_dir}/lightweight_cnn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to ONNX: {onnx_path}")

    # Export to TorchScript
    try:
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        torchscript_path = f"{output_dir}/lightweight_cnn.pt"
        traced_model.save(torchscript_path)
        print(f"Model exported to TorchScript: {torchscript_path}")
    except Exception as e:
        print(f"Failed to export to TorchScript: {e}")

        # Try scripting instead
        try:
            scripted_model = torch.jit.script(model)
            torchscript_path = f"{output_dir}/lightweight_cnn_scripted.pt"
            scripted_model.save(torchscript_path)
            print(f"Model exported to scripted TorchScript: {torchscript_path}")
        except Exception as e2:
            print(f"Failed to export to scripted TorchScript: {e2}")

def main():
    parser = argparse.ArgumentParser(description='Export Lightweight CNN to ONNX and TorchScript')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (optional)')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    # Build model
    from ..lightweight_cnn.model import build_lightweight_cnn
    model = build_lightweight_cnn()

    # Export model
    export_model(model, args.checkpoint, args.output_dir)

if __name__ == '__main__':
    main()
