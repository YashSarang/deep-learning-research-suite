"""
CIFAR-10 Data Loaders for IResNet Assignment
=============================================
Standard CIFAR-10 transforms with data augmentation for training.
Supports a --subset mode for quick smoke-testing.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def get_cifar10_dataloaders(batch_size=64, num_workers=4, use_subset=False):
    """
    Returns train and test DataLoaders for CIFAR-10.

    Train augmentation: RandomCrop(32, pad=4) + RandomHorizontalFlip + Normalize
    Test:               Normalize only

    Args:
        batch_size:  mini-batch size
        num_workers: DataLoader worker threads
        use_subset:  if True, use 1000 train / 200 test images for fast verification

    Returns:
        (trainloader, testloader)
    """

    # Standard CIFAR-10 normalization (per-channel mean/std)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    if use_subset:
        trainset = Subset(trainset, list(range(1000)))
        testset = Subset(testset, list(range(200)))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return trainloader, testloader
