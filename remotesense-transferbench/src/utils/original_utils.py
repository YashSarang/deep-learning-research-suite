import numpy as np
import torch
import torchvision.transforms as T      # For image transformation functions
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Subset, random_split   # Data handling

'''
Function name
    get_dataloaders
Description
    This code builds a reproducible train-val data split, applies appropriate 
    preprocessing on the images and returns DataLoaders and class labels.
Inputs
    data_dir => Path to the dataset. str datatype.
    batch_size => Size of image batch. int32 datatype with default value of 32.
    seed => For reproducibility. int32 datatype with default value of 42.
Outputs
    train_loader => A torch DataLoader for the training dataset.
    val_loader => A torch DataLoader for the validation dataset.
    dataset.classes => Image labels. A list datatype with str entries.
'''
def get_dataloaders(data_dir, batch_size=32, seed=42):

    # Training transformation
    # Images are re-szied, horizontally flipped, converted to tensors & normalized with ImageNet mean and std
    transform_train = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    # Validation transformation
    # Images are re-szied, converted to tensors & normalized with ImageNet mean and std
    transform_val = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    # Load dataset and apply training transformation.
    dataset = datasets.ImageFolder(data_dir, transform=transform_train)

    # Train-validation split.
    # Uses a seeded random generator for reproducibility.
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Applying validation transformation
    val_set.dataset.transform = transform_val

    # Creates and returns the data loaders.
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    return train_loader, val_loader, dataset.classes

def get_datasubset(data_dir, fraction, batch_size=32, seed=42):

    # Training transformation
        # Multiple transformations chained using "T.Compose"
        # Images are re-sized as per ResNet-50.
        # Images are randomly horizontally flipped (data augmentation)
        # PIL images are converted to tensors.
        # Normalizes each channel with ImageNet mean and std
    transform_train = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    # Validation transformation
        # Multiple transformations chained using "T.Compose"
        # Images are re-sized as per ResNet-50.
        # PIL images are converted to tensors.
        # Normalizes each channel with ImageNet mean and std
    transform_val = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    # Load dataset and apply training transformation.
    dataset = datasets.ImageFolder(data_dir, transform=transform_train)

    # Train-validation split.
    # Uses a seeded random generator for reproducibility.
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Applying validation transformation
    val_set.dataset.transform = transform_val

    num_samples = int(len(train_set) * fraction)
    indices = np.random.choice(len(train_set), num_samples, replace=False)
    subset = Subset(train_set, indices)
    # Creates and returns the data loaders.
    sub_train_loader = DataLoader(
        subset, 
        batch_size=64, 
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    return sub_train_loader, val_loader, dataset.classes


class CorruptedDataset(Dataset):
    def __init__(self, dataset, corruption="gaussian", sigma=None):
        self.dataset = dataset
        self.corruption = corruption
        self.sigma = sigma

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.corruption == "gaussian":
            img = gaussian_noise(img, self.sigma)
        elif self.corruption == "motion_blur":
            img = motion_blur(img)
        elif self.corruption == "brightness":
            img = brightness(img)

        return img, label

def gaussian_noise(img, sigma):
    noise = torch.randn_like(img) * sigma
    return torch.clamp(img + noise, 0, 1)

def motion_blur(img):
    blur = T.GaussianBlur(kernel_size=9, sigma=3)
    return blur(img)

def brightness(img):
    bright = T.ColorJitter(brightness=0.5)
    return bright(img)

def extract_layer(model, images, layer):
    probe_features = model.module.forward_probe(images)
    feat = probe_features[layer]

    # Global average pooling
    feat = torch.mean(feat, dim=[2,3])
    return feat


def train_probe(feature_extractor, model, layer, probe, dataloader, optimizer, criterion, device):
    probe.train()
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            features = feature_extractor(model, images, layer)

        outputs = probe(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return total_loss / len(dataloader), acc


def evaluate_probe(feature_extractor, model, layer, probe, dataloader, device):
    probe.eval()
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            features = feature_extractor(model, images, layer)

            outputs = probe(features)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def compute_feature_norms(feature_extractor, model, layer, dataloader, device):
    norms = []
    model.eval()

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feats = feature_extractor(model, images, layer)
            norm = torch.norm(feats, dim=1)
            norms.extend(norm.cpu().numpy())

    return np.mean(norms), np.std(norms)


def collect_balanced_pca_features(feature_extractor, model, layer, dataloader, device, num_classes, samples_per_class):
    class_counts = {i:0 for i in range(num_classes)}
    features, labels = [], []

    with torch.no_grad():
        for images, y in dataloader:
            images = images.to(device)
            feats = feature_extractor(model, images, layer).cpu()

            for i in range(len(y)):
                cls = y[i].item()
                if class_counts[cls] < samples_per_class:
                    features.append(feats[i])
                    labels.append(cls)
                    class_counts[cls] += 1

            # Stop when ALL classes have enough samples
            if all(count >= samples_per_class for count in class_counts.values()):
                break

    return torch.stack(features).numpy(), torch.tensor(labels).numpy()