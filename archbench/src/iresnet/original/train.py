"""
Unified Training Script for IResNet Assignment
===============================================
Trains either the from-scratch or official iResNet on CIFAR-10.

Usage:
    python train.py --model scratch   --epochs 20
    python train.py --model official   --epochs 20
    python train.py --model scratch   --epochs 1 --subset   # quick smoke test
"""

import argparse
import sys
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_cifar10_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='IResNet Training on CIFAR-10')
    parser.add_argument('--model', type=str, default='scratch',
                        choices=['scratch', 'official'],
                        help='Which model: scratch (our impl) or official (repo)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--subset', action='store_true',
                        help='Use a tiny subset for quick smoke testing')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader worker threads')
    return parser.parse_args()


def build_scratch_model(num_classes=10):
    """Build our from-scratch iResNet-18 for CIFAR-10."""
    from scratch_version.model import iresnet18_cifar
    return iresnet18_cifar(num_classes=num_classes)


def build_official_model(num_classes=10):
    """
    Build the official iResNet-18 from the cloned repo, adapted for CIFAR-10.

    The official model uses a 7×7 stride-2 stem designed for 224×224 ImageNet.
    We monkey-patch it in-memory to use a 3×3 stride-1 stem for 32×32 CIFAR-10
    and adjust layer1's stride, keeping the official repo code untouched.
    """
    # Add the official repo to the Python path so its imports resolve
    repo_dir = os.path.join(os.path.dirname(__file__), 'iresnet')
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from models.iresnet import iresnet18

    model = iresnet18(pretrained=False, num_classes=num_classes)

    # --- Patch the stem for CIFAR-10 (32×32) ---
    # Replace 7×7 stride-2 conv with 3×3 stride-1 conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Re-initialize the patched conv
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

    # Fix layer1: the official iResNet uses stride=2 for layer1 (since there's
    # no MaxPool in stem). For CIFAR-10 we need stride=1 at layer1 to keep
    # spatial resolution at 32×32 through the first stage.
    # We rebuild layer1 with stride=1 by calling _make_layer again.
    model.inplanes = 64  # Reset inplanes before rebuilding
    from models.iresnet import BasicBlock
    model.layer1 = model._make_layer(BasicBlock, 64, 2, stride=1)

    return model


def train_one_epoch(model, trainloader, criterion, optimizer, scaler, device):
    """Train for one epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(trainloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, testloader, criterion, device):
    """Evaluate on test set. Returns (avg_loss, accuracy%)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(testloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ---- Build model ----
    if args.model == 'scratch':
        print("Using FROM-SCRATCH iResNet-18 (our implementation)")
        net = build_scratch_model(num_classes=10)
    else:
        print("Using OFFICIAL iResNet-18 (repo, adapted for CIFAR-10)")
        net = build_official_model(num_classes=10)

    net = net.to(device)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # ---- Data ----
    trainloader, testloader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_subset=args.subset,
    )

    # ---- Training setup ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ---- Training loop ----
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    peak_vram = 0.0
    total_start = time.time()

    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            net, trainloader, criterion, optimizer, scaler, device)
        test_loss, test_acc = evaluate(net, testloader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_vram = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                      if torch.cuda.is_available() else 0)
        peak_vram = max(peak_vram, epoch_vram)

        print(f"Epoch: {epoch + 1:02d}/{args.epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Max VRAM: {epoch_vram:.0f} MB")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

    total_time = time.time() - total_start
    print(f"\nTraining Complete in {total_time / 60:.2f} mins.")

    # ---- Save metrics ----
    history['total_params'] = num_params
    history['total_time_seconds'] = total_time
    history['peak_vram_mb'] = peak_vram

    log_filename = f"history_{args.model}.json"
    with open(log_filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Metrics saved to {log_filename}")


if __name__ == '__main__':
    main()
