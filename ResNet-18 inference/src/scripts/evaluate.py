"""Evaluation script for ResNet-18 on CIFAR-100."""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import time


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple:
    """Evaluate the model on the test set.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Tuple of (top-1 accuracy, top-5 accuracy, inference time per image)
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
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
    inference_time = (end_time - start_time) / total * 1000  # ms per image
    
    return top1_acc, top5_acc, inference_time


def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 on CIFAR-100')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='./data', help='CIFAR-100 data directory')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    
    # Load model
    print("\nLoading ResNet-18 model...")
    from ..resnet18.model import build_resnet18
    from ..resnet18.utils import count_parameters
    
    model = build_resnet18().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    
    # Evaluate
    print("\n" + "="*80)
    print("Evaluating on CIFAR-100 test set...")
    print("="*80)
    
    top1_acc, top5_acc, inference_time = evaluate(model, test_loader, device)
    
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  Inference time: {inference_time:.2f} ms/image")
    print("="*80)
    
    # Comparison with Lightweight CNN (reference values)
    print("\nComparison with Lightweight CNN:")
    print(f"  {'Model':<20} {'Parameters':<15} {'Top-1 Acc':<15} {'Inference Time':<15}")
    print("-" * 70)
    print(f"  {'Lightweight CNN':<20} {'258,292':<15} {'~65-75%':<15} {'~2-5 ms':<15}")
    print(f"  {'ResNet-18':<20} {f'{total_params:,}':<15} {f'{top1_acc:.2f}%':<15} {f'{inference_time:.2f} ms':<15}")
    print("="*80)


if __name__ == '__main__':
    main()
