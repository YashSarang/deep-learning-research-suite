# Assignment 3

In this programming assignment, 
you will implement the paper that you have written a blog on, 
from scratch, and compare it against the official implementation of the paper.

### My Blog : The "Social Butterfly" of AI (How DenseNet Reimagined Convolution Neural Networks)
### Link : https://medium.com/@yashsarang.com/the-social-butterfly-of-ai-dfed0e1ee88e

### Official Implementation : https://github.com/liuzhuang13/DenseNet
### BEst Paper CVPR 2017

For the selected paper, you should run your implementation on a toy dataset or a small sample of the dataset and compare its performance with that of the official implementation. 
The aim is to be able to reproduce the results of the research paper from scratch.


## Assignment Guidelines:
You are required to submit the report.

---

## ✅ Implementation Status & Instructions

This assignment has been fully implemented.

### What's included:
- `model.py`: From-scratch PyTorch implementation of the `DenseNet-BC` architecture (L=100, k=12, optimized for CIFAR).
- `dataset.py`: Automatic `torchvision` loader for CIFAR-10, with identical augmentations as the paper.
- `train.py`: Unified, GPU-optimized training script that calculates memory footprint, leverages mixed-precision (AMP), and records history to JSON logs.
- `generate_report.py`: Script to parse training logs and automatically plot Loss and Accuracy graphs using `matplotlib`.
- `run_all.ps1`: The master PowerShell orchestrator. 

### How to reproduce results:
Simply execute the following on your Windows terminal:
```powershell
.\run_all.ps1
```
This script will automatically run training for both the from-scratch and the official implementations and immediately generate the evaluation plots (`assignment_3_metrics.png`) for your final report.
