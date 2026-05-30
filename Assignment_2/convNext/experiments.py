import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import logging

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from tqdm import tqdm
from utils import *
from ConvNeXtTiny_LP import *
from plots import *

# Setting logger
logging.basicConfig(
    filename="convNext.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

def lin_probe_convnext_tiny(device, train_loader, val_loader, classes, plot_path, lr=1e-3, num_epochs=30):    
    num_classes = len(classes)

    # Model
    model = ConvNeXtTiny_LP(num_classes)

    # Wrap model for multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.module.classifier.parameters(), lr)

    # Training Loop
    train_acc_list = []
    val_acc_list = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        correct = 0
        total = 0
        features_list = []
        labels_list = []
        all_preds = []
        all_labels = []


        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, features = model(images)
                _, predicted = outputs.max(1)

                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        features_all = np.concatenate(features_list)
        labels_all = np.concatenate(labels_list)
        val_acc = correct / total
        val_acc_list.append(val_acc)

        # Log metrics
        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] \t | \t Train Acc: {train_acc:.4f} \t | \t Val Acc: {val_acc:.4f}"
        )
 
    # Plots
    accuracy_plot(train_acc_list, val_acc_list, plot_path)  # Accuracy plot
    plot_cm(all_labels, all_preds, plot_path)   # Confusion matrix
    pca_visualization(features_all, labels_all, classes, plot_path)     # PCA feature visualization
    tsne_plot(features_all, labels_all, classes, plot_path)     # t-SNE plot

###############################################################################
###############################################################################
###############################################################################

def fine_tune_convnext_tiny(device, train_loader, val_loader, classes, strategy, lr, num_epochs, plot_path):
    num_classes = len(classes)
    model = ConvNeXtTiny_LP(num_classes)
    model = nn.DataParallel(model).to(device)

    # Freeze all backbone parameters initially
    for param in model.module.backbone.parameters():
        param.requires_grad = False

    # Strategy-specific unfreezing
    if strategy == "linear_probe":
        # Only classifier is trainable
        trainable_params = model.module.classifier.parameters()

    elif strategy == "last_block":
        # Unfreeze last ConvNeXt stage + classifier
        for param in model.module.backbone.stages[3].parameters():
            param.requires_grad = True
        trainable_params = list(model.module.backbone.stages[3].parameters()) + list(model.module.classifier.parameters())

    elif strategy == "full_finetune":
        # Unfreeze entire backbone
        for param in model.module.backbone.parameters():
            param.requires_grad = True
        trainable_params = model.parameters()

    elif strategy == "selective_unfreeze":
        total_params = sum(p.numel() for p in model.module.backbone.parameters())
        target_params = int(0.2 * total_params)

        unfrozen = 0
        trainable_params =[]
        
        # ConvNeXt uses stages.2 and stages.3 instead of layer3 and layer4
        for layer in[model.module.backbone.stages[2], model.module.backbone.stages[3]]:
            for param in layer.parameters():
                if unfrozen < target_params:
                    param.requires_grad = True
                    trainable_params.append(param)
                    unfrozen += param.numel()
                else:
                    break
        trainable_params += list(model.module.classifier.parameters())

    else:
        raise ValueError("Unknown strategy")

    strat_perc_map = {
        "linear_probe": 0,
        "last_block": 15,
        "full_finetune": 100,
        "selective_unfreeze": 20
    }
    optimizer = optim.Adam(trainable_params, lr)
    criterion = nn.CrossEntropyLoss()

    train_acc_list, val_acc_list, grad_norms = [], [], []
    train_loss_list = []

    logging.info(
        f"Strategy used - {strategy} \n ================================================================="
    )

    for epoch in tqdm(range(num_epochs)):
        model.train()
        correct, total = 0, 0
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient norm statistics
            norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
            grad_norms.append(norms)

            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss /= len(train_loader)
        train_loss_list.append(epoch_loss)
        train_acc_list.append(correct / total)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        val_acc_list.append(correct / total)

        # Log info
        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc_list[-1]:.4f}, Val Acc: {val_acc_list[-1]:.4f}"
        )

    # Plots
    plot_accuracy_vs_unfrozen(
        strategy=strategy, 
        train_acc_list=train_acc_list, 
        val_acc_list=val_acc_list, 
        percentages=strat_perc_map,
        plot_path=plot_path
    )
    plot_gradient_norms(grad_norms, strategy, plot_path)
    plot_convergence(train_loss_list, strategy, plot_path)


###############################################################################
###############################################################################
###############################################################################

def run_data_efficiency_experiment(device, data_dir, subset_batch_size, lr, num_epochs, plot_path):
    # Define regimes
    regimes = {"100%": 1.0, "20%": 0.2, "5%": 0.05}
    results = {}

    for regime_name, fraction in regimes.items():
        if regime_name == "100%":
            subset_batch_size = 1024
        elif regime_name == "100%":
            subset_batch_size = 512
        else:
            subset_batch_size = 128
        # Subset training data
        train_loader, val_loader, classes = get_datasubset(
            data_dir=data_dir,
            fraction=fraction,
            batch_size=subset_batch_size
        )
        num_classes = len(classes)

        # Model setup
        model = ConvNeXtTiny_LP(num_classes)
        model = nn.DataParallel(model).to(device)
        
        for param in model.module.backbone.parameters():
            param.requires_grad = False
        trainable_params = model.module.classifier.parameters()
        optimizer = optim.Adam(trainable_params, lr)
        criterion = nn.CrossEntropyLoss()

        train_acc_list, val_acc_list, train_loss_list = [], [], []
        for epoch in tqdm(range(num_epochs)):
            model.train()
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc_list.append(correct / total)
            train_loss_list.append(epoch_loss / len(train_loader))

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            val_acc_list.append(correct / total)

            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc_list[-1]:.4f}, Val Acc: {val_acc_list[-1]:.4f}"
            )

        results[regime_name] = {
            "train_acc": train_acc_list,
            "val_acc": val_acc_list,
            "train_loss": train_loss_list,
            "final_val_acc": val_acc_list[-1]
        }

    # Relative performance drop
    acc_100 = results["100%"]["final_val_acc"]
    acc_5 = results["5%"]["final_val_acc"]
    rel_drop = (acc_100 - acc_5) / acc_100

    # Training–validation gap
    gaps = {regime: np.mean(np.array(results[regime]["train_acc"]) - np.array(results[regime]["val_acc"]))
            for regime in regimes.keys()}
    
    plot_val_accuracy_bar(results, plot_path)
    plot_train_val_gap(gaps, plot_path)
    return rel_drop


###############################################################################
###############################################################################
###############################################################################

def load_corrupted_dataset(data_dir, batch_size, seed, corruption, sigma):
    transform_train = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    transform_val = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform_train)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    val_set.dataset.transform = transform_val

    corr_val_set = CorruptedDataset(
        val_set,
        corruption=corruption,
        sigma=sigma
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    corr_val_loader = DataLoader(
        corr_val_set, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    return val_loader, corr_val_loader, dataset.classes

def evaluate_with_corruption(data_dir, device, batch_size, seed, corruption, sigma):
    val_loader, corr_val_loader, classes = load_corrupted_dataset(
        data_dir,
        batch_size, 
        seed, 
        corruption, 
        sigma
    )
    num_classes = len(classes)
    model = ConvNeXtTiny_LP(num_classes)
    model = nn.DataParallel(model).to(device)
    model.eval()

    correct_corr, total_corr, correct_base, total_base = 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Eval baseline"):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = outputs.max(1)
            correct_base += predicted.eq(labels).sum().item()
            total_base += labels.size(0)

        for images, labels in tqdm(corr_val_loader, desc=f"Eval {corruption}"):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = outputs.max(1)
            correct_corr += predicted.eq(labels).sum().item()
            total_corr += labels.size(0)

    base_acc = correct_base / total_base
    corr_acc = correct_corr / total_corr
    error = 1 - corr_acc
    rr = corr_acc / base_acc
    return (corr_acc, error, rr)


###############################################################################
###############################################################################
###############################################################################

def layer_wise(data_dir, device, batch_size, seed, epochs, lr, plot_path):
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size, seed)
    num_classes = len(classes)
    
    model = ConvNeXtTiny_LP(num_classes=num_classes, enable_probe=True).to(device)
    model = torch.nn.DataParallel(model)
    layer_dims = {
        "stages.0":96,
        "stages.2":384,
        "stages.3":768,
    }
    results = {}

    for layer in layer_dims:
        print("Training probe for:", layer)
        probe = LinearProbe(layer_dims[layer], num_classes).to(device)
        optimizer = optim.Adam(probe.parameters(), lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in tqdm(range(epochs)):
            model.eval()

            loss, acc = train_probe(
                extract_layer,
                model,
                layer,
                probe,
                train_loader,
                optimizer,
                criterion,
                device
            )

            logging.info(
                f"Epoch [{epoch+1}/{epochs}] Loss: {loss:.4f}, Training Acc: {acc:.4f}"
            )

        val_acc = evaluate_probe(
            extract_layer,
            model,
            layer,
            probe,
            val_loader,
            device
        )

        logging.info(
            f"Val accuracy : {val_acc}"
        )
        results[layer] = val_acc

    logging.info(
        "Layer-wise accuracy \n -------------------------------------"
    )
    for k,v in results.items():
        logging.info(
            f"Layer : {k} \t | \t Accuracy : {v}"
        )
    
    # Feature norm statistics
    for layer in layer_dims:
        mean_norm, std_norm = compute_feature_norms(
            extract_layer,
            model,
            layer,
            val_loader,
            device
        )
    
        logging.info(
            f"Layer : {layer} \t|\t Feature Norm Mean: {mean_norm} \t|\t Std: {std_norm}"
        )

        features, labels = collect_balanced_pca_features(
            extract_layer,
            model,
            layer,
            val_loader,
            device,
            num_classes=30,
            samples_per_class=30
        )

        plot_pca(
            features,
            labels,
            title=f"PCA Visualization - {layer}",
            plot_path=plot_path
        )