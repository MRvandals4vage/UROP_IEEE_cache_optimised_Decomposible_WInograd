"""Training script for Improved Lightweight CNN on CIFAR-100."""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
from typing import Tuple

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
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

        total += targets.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 99:
            print(f'Epoch {epoch}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.3f}, Acc: {100. * correct / total:.2f}%')

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f'Saved improved best model with accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
