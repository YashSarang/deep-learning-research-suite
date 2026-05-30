import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def accuracy_plot(train_acc_list, val_acc_list, plot_path):
    path = plot_path + "accuracy_curve.pdf"
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_list, label="Train Accuracy", color="#1f77b4", linewidth=2.5)
    plt.plot(val_acc_list, label="Validation Accuracy", color="#ff7f0e", linewidth=2.5, linestyle="--")
    plt.xlabel("Epoch", fontsize=14, labelpad=10)
    plt.ylabel("Accuracy", fontsize=14, labelpad=10)
    plt.title("Linear Probe Accuracy", fontsize=18, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="lower right", frameon=True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_cm(all_labels, all_preds, plot_path):
    path = plot_path + "confusion_matrix.pdf"
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, square=True)
    plt.title("Confusion Matrix", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def pca_visualization(features_all, labels_all, classes, plot_path, n_components=2):
    path = plot_path + "pca_features.pdf"
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_all)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        features_pca[:, 0], 
        features_pca[:, 1],
        c=labels_all, 
        cmap="tab20", 
        s=25,
        alpha=0.8,
        edgecolors="k"
    )
    handles, _ = scatter.legend_elements()
    plt.legend(
        handles, 
        classes, 
        title="Classes", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=10
    )
    plt.title("PCA of Frozen ResNet50 Features", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def tsne_plot(features_all, labels_all, classes, plot_path, n_components=2, perplexity=10, random_state=42):
    path = plot_path + "tsne_features.pdf"
    tsne = TSNE(
        n_components=n_components, 
        perplexity=perplexity, 
        random_state=random_state
    )
    features_tsne = tsne.fit_transform(features_all[:3000])

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        features_tsne[:, 0], 
        features_tsne[:, 1],
        c=labels_all[:3000], 
        cmap="tab20", 
        s=25, 
        alpha=0.8, 
        edgecolors="k"
    )

    handles, _ = scatter.legend_elements()
    plt.legend(
        handles, 
        classes, 
        title="Classes", 
        bbox_to_anchor=(1.05, 1), 
        loc='best',
        fontsize=10,
        frameon=True
    )
    plt.title("t-SNE of Frozen ResNet50 Features", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Dim 1", fontsize=14)
    plt.ylabel("Dim 2", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

###############################################################################
###############################################################################
###############################################################################

def plot_accuracy_vs_unfrozen(strategy, train_acc_list, val_acc_list, percentages, plot_path):
    path = plot_path + f"acc_unfrozen_{strategy}.pdf"
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(train_acc_list)), train_acc_list, label=f"{strategy} Train ({percentages[strategy]}%)")
    plt.plot(range(len(val_acc_list)), val_acc_list, linestyle="--", label=f"{strategy} Val ({percentages[strategy]}%)")
    plt.title(f"Training and Val Accuracy vs % Unfrozen Parameters - {strategy}", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(fontsize=12, loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_gradient_norms(grad_norms, strategy, plot_path):
    path = plot_path + f"grad_norm_{strategy}.pdf"
    layer_names = list(grad_norms[0].keys())
    norms_per_layer = {layer: [] for layer in layer_names}
    for epoch_norms in grad_norms:
        for layer, val in epoch_norms.items():
            norms_per_layer[layer].append(val)

    plt.figure(figsize=(10, 7))
    for layer, values in norms_per_layer.items():
        plt.plot(range(len(values)), values, label=layer)
    plt.title(f"Gradient Norm Statistics Across Layers ({strategy})", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Gradient Norm", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_convergence(train_loss_list, strategy, plot_path):
    path = plot_path + f"conv_stability_{strategy}.pdf"
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(train_loss_list)), train_loss_list, label=strategy)
    plt.title(f"Training Loss vs Epoch-{strategy}", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Training Loss", fontsize=14)
    plt.legend(fontsize=10, loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

###############################################################################
###############################################################################
###############################################################################

def plot_val_accuracy_bar(results, plot_path):
    path = plot_path + f"val_acc_bar.pdf"
    regimes = list(results.keys())
    accuracies = [results[r]["final_val_acc"] for r in regimes]

    plt.figure(figsize=(8,6))
    plt.bar(regimes, accuracies, color=["#4c72b0","#55a868","#c44e52"])
    plt.title("Validation Accuracy under Different Data Fractions", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Training Data Fraction", fontsize=14)
    plt.ylabel("Validation Accuracy", fontsize=14)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc+0.01, f"{acc:.2f}", ha="center", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_train_val_gap(gaps, plot_path):
    path = plot_path + f"val_gap.pdf"
    regimes = list(gaps.keys())
    gap_values = [gaps[r] for r in regimes]

    plt.figure(figsize=(8,6))
    plt.plot(regimes, gap_values, marker="o", linewidth=2.5, color="#dd8452")
    plt.title("Training-Validation Gap vs Data Fraction", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Training Data Fraction", fontsize=14)
    plt.ylabel("Train-Validation Gap", fontsize=14)
    for i, gap in enumerate(gap_values):
        plt.text(i, gap+0.005, f"{gap:.2f}", ha="center", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

###############################################################################
###############################################################################
###############################################################################

def plot_corruption_metrics(corruption_results, plot_path):
    path = plot_path +  "corrupted_plots.pdf"
    corruptions = list(corruption_results.keys())
    accuracies = [corruption_results[c]["accuracy"] for c in corruptions]
    errors = [corruption_results[c]["error"] for c in corruptions]
    robustness = [corruption_results[c]["robustness"] for c in corruptions]

    x = np.arange(len(corruptions))
    width = 0.25

    plt.figure(figsize=(12,7))
    bars1 = plt.bar(x - width, accuracies, width, color="red", label="Validation Accuracy")
    bars2 = plt.bar(x, errors, width, color="blue", label="Corruption Error")
    bars3 = plt.bar(x + width, robustness, width, color="green", label="Robustness Sensitivity")

    plt.title("Model Robustness under Different Corruptions", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Corruption Type", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.xticks(x, corruptions, rotation=30, ha="right")
    plt.legend(fontsize=12, loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Add value labels above bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                     f"{height:.2f}", ha="center", va="bottom", fontsize=10)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

###############################################################################
###############################################################################
###############################################################################

def plot_pca(features, labels, title, plot_path):
    path = plot_path + "pca_feat_labels.pdf"
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(7,7))
    scatter = plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=labels,
        cmap="tab20",
        s=15
    )

    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    plt.title(title, fontsize=18, fontweight="bold", pad=15)
    plt.colorbar(scatter)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()