# GNR-638: Deep Learning Coursework - Assignment 3

![Course](https://img.shields.io/badge/Course-GNR_638-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)

This repository contains the complete implementation and evaluation for **Assignment 3** of the GNR-638 Deep Learning course. The project explores two major architectural advancements in Convolutional Neural Networks: **DenseNet** and **iResNet**.

## Project Structure

The repository is organized into two primary implementation suites, each focusing on a specific architecture:

1.  **[Assignment_3-DenseNet](./Assignment_3-DenseNet/)**: A rigorous replication of the "Social Butterfly of AI" (DenseNet), comparing custom implementations against official PyTorch and legacy Lua baselines.
2.  **[Assignment_3-IResNet](./Assignment_3-IResNet/)**: An implementation of "Fixing the Cracks in ResNet" (iResNet), focusing on improving signal flow and architectural efficiency.

---

## 🏗️ Implementations Overview

### 1. DenseNet (Densely Connected Convolutional Networks)
This suite replicates the CVPR 2017 Best Paper. It features a memory-efficient "from-scratch" implementation that utilizes checkpointing to handle deep networks on limited hardware.

*   **Key Features**: Custom DenseBlock implementation, memory-efficient training, and comparison with official baselines.
*   **Compatibility Note**: As the official implementation of DenseNet was originally written in **Torch7 (Lua)**, it requires a specialized **Docker** environment (`nagadomi/torch7`) to run. Due to the deprecation of Torch7 and complex dependency chains, this implementation encountered significant friction on Linux-based environments.

### 2. iResNet (Improved Residual Networks) — *Extra Implementation*
To ensure a seamless and modern evaluation experience, we have included **iResNet** as an extra implementation. Unlike the legacy requirements of the DenseNet suite, iResNet is built entirely on modern PyTorch standards.

*   **Why iResNet?**: We went ahead with iResNet because it **works perfectly fine without the requirement of Docker or legacy runtimes**. It "fixes the cracks" in traditional ResNet architectures (Post-Activation ReLU, Strided Conv, and Aggressive Stem issues) and provides a robust, high-performance baseline for CIFAR-10.
*   **Key Benefits**: 100% native PyTorch, easy to install via `requirements.txt`, and produces high-quality academic reports and visualizations out-of-the-box.

---

## Getting Started

Each directory contains its own detailed `README.md` and `run_all.ps1` script for automated execution.

### For iResNet (Recommended for Quick Setup):
```bash
cd Assignment_3-IResNet
pip install -r requirements.txt
./run_all.ps1
```

### For DenseNet:
```bash
cd Assignment_3-DenseNet
# Follow the detailed Instructions.md inside for Docker setup
./run_all.ps1
```

## Authors & Research
- **DenseNet Blog**: [The "Social Butterfly" of AI](https://medium.com/@yashsarang.com/the-social-butterfly-of-ai-dfed0e1ee88e)
- **iResNet Blog**: [Fixing the Cracks in ResNet](https://medium.com/@sarvesh260500/fixing-the-cracks-in-resnet-6c803821f9b9)

---
*Developed as part of the GNR-638 Deep Learning Coursework.*
