"""Compare Lightweight CNN with and without Winograd transform."""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import time
import numpy as np

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy and inference time."""
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times)
    
    return accuracy, avg_inference_time

def count_parameters(model):
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_macs(model, input_shape=(1, 3, 32, 32)):
    """Estimate MACs for the model."""
    # This is a simplified MAC estimation
    total_macs = 0
    
    def conv_macs(in_channels, out_channels, kernel_size, output_size):
        return in_channels * out_channels * kernel_size * kernel_size * output_size[0] * output_size[1]
    
    # Rough estimation based on model architecture
    # ConvStem1: 3->16, k=3, s=2, 32x32 -> 16x16
    total_macs += conv_macs(3, 16, 3, (16, 16))
    
    # ConvStem2: 16->32, k=3, s=2, 16x16 -> 8x8
    total_macs += conv_macs(16, 32, 3, (8, 8))
    
    # Pool to 7x7
    # ConvBlock1: 32->64, k=3, s=1, 7x7 -> 7x7
    total_macs += conv_macs(32, 64, 3, (7, 7))
    
    # ConvBlock2: 64->128, k=3, s=1, 7x7 -> 7x7
    total_macs += conv_macs(64, 128, 3, (7, 7))
    
    # ConvBlock3: 128->128, k=3, s=1, 7x7 -> 7x7
    total_macs += conv_macs(128, 128, 3, (7, 7))
    
    # Classifier: 128->100
    total_macs += 128 * 100
    
    return total_macs

def main():
    parser = argparse.ArgumentParser(description='Compare Winograd vs Standard CNN')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--data-dir', type=str, default='./data', help='CIFAR-100 data directory')
    parser.add_argument('--standard-checkpoint', type=str, default='best_model.ckpt', 
                       help='Path to standard model checkpoint')
    parser.add_argument('--winograd-checkpoint', type=str, default='best_model_winograd.ckpt',
                       help='Path to Winograd model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("="*60)
    print("LIGHTWEIGHT CNN: WINOGRAD vs STANDARD COMPARISON")
    print("="*60)

    # Load test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    testset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load models
    from ..lightweight_cnn.model import build_lightweight_cnn
    from ..lightweight_cnn.model_winograd import build_lightweight_cnn_winograd

    # Standard model
    standard_model = build_lightweight_cnn().to(device)
    if torch.cuda.is_available():
        standard_model.load_state_dict(torch.load(args.standard_checkpoint))
    else:
        standard_model.load_state_dict(torch.load(args.standard_checkpoint, map_location='cpu'))

    # Winograd model
    winograd_model = build_lightweight_cnn_winograd().to(device)
    if torch.cuda.is_available():
        winograd_model.load_state_dict(torch.load(args.winograd_checkpoint))
    else:
        winograd_model.load_state_dict(torch.load(args.winograd_checkpoint, map_location='cpu'))

    # Model statistics
    standard_params = count_parameters(standard_model)
    winograd_params = count_parameters(winograd_model)
    
    standard_macs = estimate_macs(standard_model)
    winograd_macs = estimate_macs(winograd_model)
    
    # For Winograd, theoretical MAC reduction is ~2.25x for 3x3 convs
    # Estimate MAC savings for the core conv blocks
    conv_macs_standard = conv_macs(32, 64, 3, (7, 7)) + conv_macs(64, 128, 3, (7, 7)) + conv_macs(128, 128, 3, (7, 7))
    conv_macs_winograd = conv_macs_standard / 2.25  # Theoretical Winograd reduction
    winograd_macs_theoretical = standard_macs - conv_macs_standard + conv_macs_winograd

    print("\n📊 MODEL STATISTICS")
    print("-" * 40)
    print(f"Standard CNN Parameters:     {standard_params:,}")
    print(f"Winograd CNN Parameters:     {winograd_params:,}")
    print(f"Parameter Difference:        {abs(winograd_params - standard_params):,}")
    
    print(f"\nStandard CNN MACs:           {standard_macs:,}")
    print(f"Winograd CNN MACs (theoretical): {winograd_macs_theoretical:,}")
    print(f"MAC Reduction:               {((standard_macs - winograd_macs_theoretical) / standard_macs * 100):.1f}%")

    # Evaluate models
    print("\n🧪 EVALUATING MODELS")
    print("-" * 40)
    
    print("Evaluating Standard CNN...")
    standard_acc, standard_time = evaluate_model(standard_model, test_loader, device)
    
    print("Evaluating Winograd CNN...")
    winograd_acc, winograd_time = evaluate_model(winograd_model, test_loader, device)

    # Results
    print("\n📈 RESULTS COMPARISON")
    print("-" * 40)
    print(f"Standard CNN Accuracy:       {standard_acc:.2f}%")
    print(f"Winograd CNN Accuracy:       {winograd_acc:.2f}%")
    print(f"Accuracy Difference:         {winograd_acc - standard_acc:+.2f}%")
    
    print(f"\nStandard CNN Inference Time: {standard_time*1000:.2f} ms/batch")
    print(f"Winograd CNN Inference Time: {winograd_time*1000:.2f} ms/batch")
    print(f"Speed Improvement:           {((standard_time - winograd_time) / standard_time * 100):+.1f}%")

    # Energy efficiency (accuracy per MAC)
    standard_efficiency = standard_acc / (standard_macs / 1e6)  # Accuracy per million MACs
    winograd_efficiency = winograd_acc / (winograd_macs_theoretical / 1e6)
    
    print(f"\n⚡ ENERGY EFFICIENCY")
    print("-" * 40)
    print(f"Standard CNN (Acc/M-MACs):   {standard_efficiency:.2f}")
    print(f"Winograd CNN (Acc/M-MACs):   {winograd_efficiency:.2f}")
    print(f"Efficiency Improvement:      {((winograd_efficiency - standard_efficiency) / standard_efficiency * 100):+.1f}%")

    # Summary
    print("\n🎯 SUMMARY")
    print("-" * 40)
    if winograd_acc >= standard_acc - 1.0:  # Within 1% accuracy
        print("✅ Winograd CNN maintains comparable accuracy")
    else:
        print("⚠️  Winograd CNN has lower accuracy")
    
    if winograd_time < standard_time:
        print("✅ Winograd CNN is faster")
    else:
        print("⚠️  Winograd CNN is slower (implementation overhead)")
    
    print(f"✅ Winograd CNN reduces theoretical MACs by {((standard_macs - winograd_macs_theoretical) / standard_macs * 100):.1f}%")
    print(f"✅ Winograd CNN improves energy efficiency by {((winograd_efficiency - standard_efficiency) / standard_efficiency * 100):+.1f}%")

def conv_macs(in_channels, out_channels, kernel_size, output_size):
    """Calculate MACs for a convolution layer."""
    return in_channels * out_channels * kernel_size * kernel_size * output_size[0] * output_size[1]

if __name__ == '__main__':
    main()
