"""Train both Standard and Winograd models and compare results."""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import time
import os
from typing import Tuple, Dict

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_name: str
) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'  [{model_name}] Batch {batch_idx + 1}/{len(train_loader)}, '
                  f'Loss: {running_loss / (batch_idx + 1):.3f}, '
                  f'Acc: {100. * correct / total:.2f}%')

    train_loss = running_loss / len(train_loader)
    return train_loss

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def train_model(model, model_name, train_loader, val_loader, device, args):
    """Train a single model and return best accuracy."""
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    save_path = f"best_model_{model_name.lower().replace(' ', '_')}.ckpt"
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, model_name)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        print(f'[{model_name}] Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'[{model_name}] Saved best model with accuracy: {best_acc:.2f}%')

    return best_acc, save_path

def count_parameters(model):
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_theoretical_macs(model_type):
    """Estimate theoretical MACs for each model type."""
    # Base MACs for stem layers and classifier
    base_macs = 0
    
    # ConvStem1: 3->16, k=3, s=2, 32x32 -> 16x16
    base_macs += 3 * 16 * 3 * 3 * 16 * 16
    
    # ConvStem2: 16->32, k=3, s=2, 16x16 -> 8x8  
    base_macs += 16 * 32 * 3 * 3 * 8 * 8
    
    # Classifier: 128->100
    base_macs += 128 * 100
    
    # Core convolution MACs
    # ConvBlock1: 32->64, k=3, s=1, 7x7 -> 7x7
    conv1_macs = 32 * 64 * 3 * 3 * 7 * 7
    
    # ConvBlock2: 64->128, k=3, s=1, 7x7 -> 7x7
    conv2_macs = 64 * 128 * 3 * 3 * 7 * 7
    
    # ConvBlock3: 128->128, k=3, s=1, 7x7 -> 7x7
    conv3_macs = 128 * 128 * 3 * 3 * 7 * 7
    
    total_conv_macs = conv1_macs + conv2_macs + conv3_macs
    
    if model_type == "standard":
        return base_macs + total_conv_macs
    elif model_type == "winograd":
        # Winograd reduces 3x3 conv MACs by ~2.25x
        return base_macs + (total_conv_macs / 2.25)
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Train and Compare Standard vs Winograd CNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='CIFAR-100 data directory')
    parser.add_argument('--train-both', action='store_true', help='Train both models')
    parser.add_argument('--compare-only', action='store_true', help='Only compare existing models')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("🚀 LIGHTWEIGHT CNN: WINOGRAD vs STANDARD EXPERIMENT")

    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )

    # Split train into train and validation
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load models
    from ..lightweight_cnn.model_standard import build_lightweight_cnn_standard
    from ..lightweight_cnn.model_winograd import build_lightweight_cnn_winograd

    standard_model = build_lightweight_cnn_standard().to(device)
    winograd_model = build_lightweight_cnn_winograd().to(device)

    # Model statistics
    standard_params = count_parameters(standard_model)
    winograd_params = count_parameters(winograd_model)
    standard_macs = estimate_theoretical_macs("standard")
    winograd_macs = estimate_theoretical_macs("winograd")

    print(f"\n📊 MODEL ARCHITECTURE COMPARISON")
    print("-" * 50)
    print(f"Standard CNN Parameters:     {standard_params:,}")
    print(f"Winograd CNN Parameters:     {winograd_params:,}")
    print(f"Parameter Difference:        {abs(winograd_params - standard_params):,}")
    
    print(f"\nStandard CNN MACs:           {standard_macs:,}")
    print(f"Winograd CNN MACs:           {winograd_macs:,}")
    print(f"MAC Reduction:               {((standard_macs - winograd_macs) / standard_macs * 100):.1f}%")

    results = {}

    if not args.compare_only:
        # Train models
        if args.train_both:
            print("\n🏋️ TRAINING BOTH MODELS")
            
            # Train standard model
            standard_acc, standard_path = train_model(
                standard_model, "Standard CNN", train_loader, val_loader, device, args
            )
            results['standard'] = {'accuracy': standard_acc, 'path': standard_path}
            
            # Train Winograd model  
            winograd_acc, winograd_path = train_model(
                winograd_model, "Winograd CNN", train_loader, val_loader, device, args
            )
            results['winograd'] = {'accuracy': winograd_acc, 'path': winograd_path}
        else:
            print("\n🏋️ TRAINING SINGLE MODEL (use --train-both for both)")
            # Default to training standard model
            standard_acc, standard_path = train_model(
                standard_model, "Standard CNN", train_loader, val_loader, device, args
            )
            results['standard'] = {'accuracy': standard_acc, 'path': standard_path}

    # Final comparison
    print(f"\n🎯 FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Model Architecture Efficiency:")
    print(f"  Standard CNN: {standard_params:,} params, {standard_macs:,} MACs")
    print(f"  Winograd CNN: {winograd_params:,} params, {winograd_macs:,} MACs")
    print(f"  MAC Reduction: {((standard_macs - winograd_macs) / standard_macs * 100):.1f}%")
    
    if results:
        print(f"\nTraining Results:")
        for model_type, result in results.items():
            print(f"  {model_type.title()} CNN: {result['accuracy']:.2f}% accuracy")
    
    print(f"\n✅ Experiment completed!")
    print(f"📁 Model checkpoints saved in current directory")

if __name__ == '__main__':
    main()
