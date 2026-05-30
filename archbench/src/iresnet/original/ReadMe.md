# iResNet Assignment - ReadMe

This repository contains the implementation of **Improved Residual Networks (iResNet)** for Assignment 3. It includes a from-scratch implementation, the official repository for comparison, and a unified pipeline for training and evaluation on CIFAR-10.

---

## 🚀 Research Foundation: "Fixing the Cracks in ResNet"

This implementation is based on the architectural refinements proposed in the iResNet paper (Duta et al., 2020). For a deep dive into the motivation and technical breakdown of these improvements, refer to the following research blog:

**Medium Article:** [**Fixing the Cracks in ResNet**](https://medium.com/@sarvesh260500/fixing-the-cracks-in-resnet-6c803821f9b9)  
*Author: Sarvesh Shashidhar*

### Key "Cracks" Fixed in iResNet:
1.  **The Post-Activation ReLU Crack**: Standard ResNet's `Add → ReLU` pattern clips negative gradients. iResNet restructures the block to maintain identity signal flow.
2.  **The Strided Conv Crack**: Replaces information-losing strided 1x1 convolutions in the projection shortcut with a **3x3 MaxPool** followed by a 1x1 Conv.
3.  **The Aggressive Stem Crack**: Optimizes the network's initial layers to preserve more spatial detail, especially beneficial for smaller datasets like CIFAR-10.

---

## 1. Prerequisites

Ensure you have a Python environment (3.8+) with CUDA support for efficient training.

### Install Dependencies
Run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 2. Running the Pipeline

You can run the entire evaluation pipeline (training both models and generating the report) using the provided PowerShell script.

### Unified Execution (Recommended)
This script handles the full workflow:
1. Trains the from-scratch iResNet.
2. Trains the official iResNet.
3. Generates the comparative report and plots.

```powershell
./run_all.ps1
```

---

## 3. Manual Execution

If you wish to run individual steps, use the following commands:

### Training the Custom Model (From-Scratch)
```bash
python train.py --model scratch --epochs 100
```

### Training the Official Model
```bash
python train.py --model official --epochs 100
```

### Quick Smoke Test
To verify the setup quickly with a subset of data and 1 epoch:
```bash
python train.py --model scratch --epochs 1 --subset
```

### Generating the Report
Once training is complete and `history_scratch.json` and `history_official.json` are present, generate the final report:
```bash
python generate_report.py
```

---

## 4. Output Artifacts

- **`Report.md`**: The final descriptive report summarizing the findings.
- **`iresnet_metrics.png`**: Comparison plots for Loss and Accuracy.
- **`history_*.json`**: Raw training logs for both models.

---

## 5. Directory Structure

- `scratch_version/`: Contains our from-scratch implementation of iResNet.
- `iresnet/`: The official iResNet implementation (cloned repository).
- `train.py`: Unified training script with model patching for CIFAR-10.
- `dataset.py`: CIFAR-10 data loaders and augmentations.
- `generate_report.py`: Script to visualize results and generate the Markdown report.
