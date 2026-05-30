'''
pca.py
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_pca(features, labels, title):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=labels,
        cmap="tab20",
        s=10
    )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter)
    plt.savefig("../Figures/resnet_50/")