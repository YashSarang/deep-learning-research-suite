import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
import logging

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T      # For image transformation functions
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split   # Data handling

from collections import defaultdict
from thop import profile    # Used to calculate MACs and FLOPs
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import *
from ConvNeXtTiny_LP import *
from experiments import *

# Setting logger
logging.basicConfig(
    filename="convNext.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

def main(args):
    # Setting seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    # Loading data
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )

    model_convnext_tiny = ConvNeXtTiny_LP(num_classes=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_convnext_tiny.to(device)

    # Getting params, macs and flops
    dummy = torch.randn(1,3,224,224).to(device)

    # Get the flops and params using thop.profile 
    macs, params = profile(model_convnext_tiny, inputs=(dummy,), verbose=False)
    flops = 2 * macs
    logging.info(
        f"Number of params={params} \t | \t MACs = {macs} \t | \t FLOPS={flops}"
    )
    
    # Exp 4.1
    logging.info(
        "Linear Probe Transfer \n =============================================="
    )
    lin_probe_convnext_tiny(
        device=device, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        classes=classes,
        plot_path = args.plot_path,
        lr=args.lr,
        num_epochs=args.epochs
    )
    torch.cuda.empty_cache()

    # Exp 4.2
    logging.info(
        "Fine-Tuning Strategies \n =============================================="
    )
    strats =["linear_probe", "last_block", "full_finetune", "selective_unfreeze"]
    for strategy in strats:
        fine_tune_convnext_tiny(device, train_loader, val_loader, classes, strategy, args.lr, args.epochs, args.plot_path)
        torch.cuda.empty_cache()
    
    # Exp 4.3
    logging.info(
        "Few-Shot Learning \n =============================================="
    )
    rel_drop = run_data_efficiency_experiment(device, args.data_dir, args.subset_batch_size, args.lr, args.small_epochs, args.plot_path)
    logging.info(
        f"Relative Performance Drop: {rel_drop:.4f}"
    )
    torch.cuda.empty_cache()
    
    # Exp 4.4
    logging.info(
        "Corruption Robustness \n =============================================="
    )
    corruptions = ["gaussian_0.05", "gaussian_0.1", "gaussian_0.2", 'motion_blur', 'brightness']
    corruption_results = defaultdict(lambda: defaultdict(float))
    for corr in corruptions:
        if (corr == 'gaussian_0.05'):
            c = 'gaussian'
            sigma = 0.05
        elif (corr == 'gaussian_0.1'):
            c = 'gaussian'
            sigma = 0.1
        elif (corr == 'gaussian_0.2'):
            c = 'gaussian'
            sigam = 0.2
        else:
            c = corr
            sigma = None
        
        corr_acc, error, rr = evaluate_with_corruption(args.data_dir,
            device, 
            batch_size=args.subset_batch_size, 
            seed=args.seed, 
            corruption=c, 
            sigma=sigma
        )

        logging.info(
            f"Corruption = {corr} \t | \t Val accuracy = {corr_acc} \t | \t Corruption error = {error} \t | \t Robustness = {rr}"
        )
        corruption_results[c]['accuracy'] = corr_acc
        corruption_results[c]['error'] = error
        corruption_results[c]['robustness'] = rr
    
    plot_corruption_metrics(corruption_results, args.plot_path)
    torch.cuda.empty_cache()

    # Exp 4.5
    layer_wise(args.data_dir, device, args.subset_batch_size, args.seed, args.epochs, args.lr, args.plot_path)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../train_data", help="Relative path to the dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for fine-tuning")
    parser.add_argument("--small_epochs", type=int, default=20, help="Number of epochs for few-shot analysis")
    parser.add_argument("--batch_size", type=int, default=512, help="Size of the batch")
    parser.add_argument("--subset_batch_size", type=int, default=64, help="Size of the batch for subset sampling")
    parser.add_argument("--plot_path", type=str, default="../Figures/resnet_50/", help="Path to store plots")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    args = parser.parse_args()

    main(args)