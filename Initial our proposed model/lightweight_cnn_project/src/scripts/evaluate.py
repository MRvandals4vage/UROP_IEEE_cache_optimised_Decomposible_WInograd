"""Evaluation script for Lightweight CNN on CIFAR-100."""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the model on the test set.

    Args:
        model: The model to evaluate.
        test_loader: Test data loader.
        device: Device to evaluate on.

    Returns:
        Top-1 accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate Lightweight CNN on CIFAR-100')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='./data', help='CIFAR-100 data directory')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load test dataset
    testset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load model
    from ..lightweight_cnn.model import build_lightweight_cnn
    model = build_lightweight_cnn().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Evaluate
    accuracy = evaluate(model, test_loader, device)
    print(f'Top-1 Accuracy on CIFAR-100 test set: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
