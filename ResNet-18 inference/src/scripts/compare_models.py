"""Compare ResNet-18 and Lightweight CNN performance on CIFAR-100."""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import time
import sys
import os


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, 
                   model_name: str) -> dict:
    """Evaluate a model and return comprehensive metrics.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        model_name: Name of the model for display
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    print(f"\nEvaluating {model_name}...")
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_top1 += predicted.eq(targets).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            targets_expanded = targets.view(-1, 1).expand_as(top5_pred)
            correct_top5 += top5_pred.eq(targets_expanded).sum().item()
    
    end_time = time.time()
    
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total
    total_time = end_time - start_time
    time_per_image = total_time / total * 1000  # ms per image
    
    return {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'total_time': total_time,
        'time_per_image': time_per_image,
        'total_samples': total
    }


def main():
    parser = argparse.ArgumentParser(description='Compare ResNet-18 and Lightweight CNN')
    parser.add_argument('--resnet-checkpoint', type=str, required=True, 
                       help='Path to ResNet-18 checkpoint')
    parser.add_argument('--lightweight-checkpoint', type=str, required=True,
                       help='Path to Lightweight CNN checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='./data', help='CIFAR-100 data directory')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("Model Comparison: ResNet-18 vs Lightweight CNN")
    print("="*80)
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data transforms
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Load test dataset
    print("\nLoading CIFAR-100 test dataset...")
    testset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(testset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test samples: {len(testset)}")
    
    # Load ResNet-18
    print("\n" + "-"*80)
    print("Loading ResNet-18...")
    from ..resnet18.model import build_resnet18
    from ..resnet18.utils import count_parameters as count_params_resnet
    from ..resnet18.macs import compute_macs as compute_macs_resnet
    
    resnet_model = build_resnet18().to(device)
    checkpoint = torch.load(args.resnet_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        resnet_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        resnet_model.load_state_dict(checkpoint)
    
    resnet_params = count_params_resnet(resnet_model)
    resnet_macs = sum(compute_macs_resnet(resnet_model).values())
    print(f"ResNet-18 parameters: {resnet_params:,}")
    print(f"ResNet-18 MACs: {resnet_macs:,}")
    
    # Load Lightweight CNN
    print("\n" + "-"*80)
    print("Loading Lightweight CNN...")
    
    # Add the lightweight CNN path to sys.path
    lightweight_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 
                     '../../../Initial our proposed model/lightweight_cnn_project/src')
    )
    sys.path.insert(0, lightweight_path)
    
    from lightweight_cnn.model import build_lightweight_cnn
    from lightweight_cnn.utils import count_parameters as count_params_light
    from lightweight_cnn.macs import compute_macs as compute_macs_light
    
    lightweight_model = build_lightweight_cnn().to(device)
    lightweight_model.load_state_dict(
        torch.load(args.lightweight_checkpoint, map_location=device)
    )
    
    lightweight_params = count_params_light(lightweight_model)
    lightweight_macs = sum(compute_macs_light(lightweight_model).values())
    print(f"Lightweight CNN parameters: {lightweight_params:,}")
    print(f"Lightweight CNN MACs: {lightweight_macs:,}")
    
    # Evaluate both models
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    resnet_results = evaluate_model(resnet_model, test_loader, device, "ResNet-18")
    lightweight_results = evaluate_model(lightweight_model, test_loader, device, "Lightweight CNN")
    
    # Print comparison table
    print("\n" + "="*80)
    print("Detailed Comparison")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Lightweight CNN':<25} {'ResNet-18':<25} {'Difference':<20}")
    print("-" * 100)
    
    # Parameters
    param_ratio = resnet_params / lightweight_params
    print(f"{'Parameters':<30} {f'{lightweight_params:,}':<25} {f'{resnet_params:,}':<25} {f'{param_ratio:.1f}x more':<20}")
    
    # MACs
    macs_ratio = resnet_macs / lightweight_macs
    print(f"{'MACs':<30} {f'{lightweight_macs:,}':<25} {f'{resnet_macs:,}':<25} {f'{macs_ratio:.1f}x more':<20}")
    
    # Model size
    light_size = (lightweight_params * 4) / (1024 * 1024)
    resnet_size = (resnet_params * 4) / (1024 * 1024)
    size_ratio = resnet_size / light_size
    print(f"{'Model Size (MB)':<30} {f'{light_size:.2f}':<25} {f'{resnet_size:.2f}':<25} {f'{size_ratio:.1f}x larger':<20}")
    
    print()
    
    # Top-1 Accuracy
    acc_diff = resnet_results['top1_acc'] - lightweight_results['top1_acc']
    print(f"{'Top-1 Accuracy (%)':<30} {f'{lightweight_results["top1_acc"]:.2f}':<25} {f'{resnet_results["top1_acc"]:.2f}':<25} {f'{acc_diff:+.2f}%':<20}")
    
    # Top-5 Accuracy
    acc5_diff = resnet_results['top5_acc'] - lightweight_results['top5_acc']
    print(f"{'Top-5 Accuracy (%)':<30} {f'{lightweight_results["top5_acc"]:.2f}':<25} {f'{resnet_results["top5_acc"]:.2f}':<25} {f'{acc5_diff:+.2f}%':<20}")
    
    # Inference time
    time_ratio = resnet_results['time_per_image'] / lightweight_results['time_per_image']
    print(f"{'Inference Time (ms/img)':<30} {f'{lightweight_results["time_per_image"]:.2f}':<25} {f'{resnet_results["time_per_image"]:.2f}':<25} {f'{time_ratio:.1f}x slower':<20}")
    
    # Total inference time
    print(f"{'Total Time (s)':<30} {f'{lightweight_results["total_time"]:.2f}':<25} {f'{resnet_results["total_time"]:.2f}':<25} {'':<20}")
    
    # Energy efficiency metric (MACs per accuracy point)
    light_efficiency = lightweight_macs / lightweight_results['top1_acc']
    resnet_efficiency = resnet_macs / resnet_results['top1_acc']
    efficiency_ratio = resnet_efficiency / light_efficiency
    print(f"{'MACs per Accuracy Point':<30} {f'{light_efficiency:,.0f}':<25} {f'{resnet_efficiency:,.0f}':<25} {f'{efficiency_ratio:.1f}x worse':<20}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("\n✓ Lightweight CNN Advantages:")
    print(f"  • {param_ratio:.1f}x fewer parameters ({lightweight_params:,} vs {resnet_params:,})")
    print(f"  • {macs_ratio:.1f}x fewer MACs (computational operations)")
    print(f"  • {time_ratio:.1f}x faster inference ({lightweight_results['time_per_image']:.2f} ms vs {resnet_results['time_per_image']:.2f} ms)")
    print(f"  • {size_ratio:.1f}x smaller model size ({light_size:.2f} MB vs {resnet_size:.2f} MB)")
    print(f"  • {efficiency_ratio:.1f}x better energy efficiency (MACs per accuracy point)")
    
    print("\n✓ ResNet-18 Advantages:")
    print(f"  • {acc_diff:.2f}% higher top-1 accuracy ({resnet_results['top1_acc']:.2f}% vs {lightweight_results['top1_acc']:.2f}%)")
    print(f"  • {acc5_diff:.2f}% higher top-5 accuracy ({resnet_results['top5_acc']:.2f}% vs {lightweight_results['top5_acc']:.2f}%)")
    print(f"  • Better feature representation capacity")
    
    print("\n✓ Use Case Recommendations:")
    print("  • Lightweight CNN: Edge devices, mobile apps, real-time systems, energy-constrained environments")
    print("  • ResNet-18: Server-side inference, applications requiring maximum accuracy, GPU-accelerated systems")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
