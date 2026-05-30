import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

def get_cifar10_dataloaders(batch_size=64, num_workers=4, use_subset=False):
    """
    Returns train and test dataloaders for CIFAR-10.
    Includes data augmentation for training.
    """
    
    # Standard DenseNet CIFAR transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # If use_subset is True, reduce the dataset size drastically for fast testing
    if use_subset:
        train_indices = list(range(0, 1000))  # 1000 images for fast training test
        test_indices = list(range(0, 200))    # 200 images for fast testing
        
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True)

    return trainloader, testloader
