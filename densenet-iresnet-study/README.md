# ArchBench-DenseNet-iResNet — Architectural Deep Dive

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Rigorous replications of **DenseNet** (CVPR 2017 Best Paper) and **iResNet** architectural improvements. Features memory-efficient training, checkpointing strategies, and comprehensive ablation studies.

---

## 🎯 What's Inside

### 1. DenseNet: "The Social Butterfly of AI"
- ✅ Custom DenseNet implementation with gradient checkpointing
- ✅ Memory-efficient training (vs. torchvision baseline)
- ✅ Comparison: Custom vs. Official PyTorch vs. Legacy Torch7/Lua
- ✅ Ablation: Impact of growth rate, compression, depth

**Blog:** [The "Social Butterfly" of AI](https://medium.com/@yashsarang.com/the-social-butterfly-of-ai-dfed0e1ee88e)

### 2. iResNet: "Fixing the Cracks in ResNet"
- ✅ Fixes 3 architectural issues: Post-Activation ReLU, Strided Conv, Aggressive Stem
- ✅ 100% native PyTorch (no Docker/legacy runtimes)
- ✅ High-quality academic reports & visualizations
- ✅ CIFAR-10/100 benchmarks

**Blog:** [Fixing the Cracks in ResNet](https://medium.com/@sarvesh260500/fixing-the-cracks-in-resnet-6c803821f9b9)

---

## 📦 Installation

```bash
cd densenet-iresnet-study

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download CIFAR-10/100 (automatic on first run)
python src/download_datasets.py
```

---

## 🚀 Quick Start

### DenseNet Experiments

```bash
# Train custom DenseNet-121 on CIFAR-10
python src/densenet/train.py --model densenet121 --dataset cifar10 --epochs 300

# Compare with official baseline
python src/densenet/compare_baselines.py

# Run ablation studies
bash scripts/densenet_ablations.sh
```

### iResNet Experiments

```bash
# Train iResNet-50 on CIFAR-10
python src/iresnet/train.py --model iresnet50 --dataset cifar10 --epochs 200

# Run full evaluation suite
bash scripts/run_iresnet_suite.sh
```

---

## 📊 Key Results

### DenseNet Memory Efficiency

| Model | Torchvision (GB) | Custom + Checkpointing (GB) | Reduction |
|-------|------------------|----------------------------|-----------|
| DenseNet-121 | 4.2 | **2.1** | 50% |
| DenseNet-169 | 6.8 | **3.2** | 53% |
| DenseNet-201 | 9.5 | **4.1** | 57% |

### iResNet CIFAR-10 Accuracy

| Model | Baseline ResNet | iResNet (Ours) | Improvement |
|-------|-----------------|----------------|-------------|
| ResNet-50 | 94.2% | **95.7%** | +1.5% |
| ResNet-101 | 94.8% | **96.1%** | +1.3% |
| ResNet-152 | 95.1% | **96.4%** | +1.3% |

---

## 🏗️ Architecture

```
densenet-iresnet-study/
├── src/
│   ├── densenet/
│   │   ├── model.py              ← Custom DenseNet implementation
│   │   ├── train.py              ← Training script
│   │   ├── compare_baselines.py  ← Compare with official
│   │   └── ablations/            ← Growth rate, compression studies
│   │
│   ├── iresnet/
│   │   ├── model.py              ← iResNet implementation
│   │   ├── train.py              ← Training script
│   │   └── analyze_fixes.py      ← Ablate each fix
│   │
│   ├── utils/
│   │   ├── memory_profiler.py    ← GPU memory tracking
│   │   ├── visualization.py      ← Feature maps, gradients
│   │   └── checkpointing.py      ← Gradient checkpointing utils
│   │
│   └── data/
│       ├── cifar.py              ← CIFAR-10/100 loaders
│       └── augmentations.py      ← AutoAugment, Cutout
│
├── notebooks/
│   ├── densenet_analysis.ipynb   ← Feature visualization
│   ├── iresnet_ablations.ipynb   ← Ablation analysis
│   └── memory_comparison.ipynb   ← Memory profiling
│
├── scripts/
│   ├── densenet_ablations.sh     ← Run all DenseNet experiments
│   ├── run_iresnet_suite.sh      ← Run all iResNet experiments
│   └── generate_report.py        ← PDF report generator
│
├── tests/
│   ├── test_densenet.py
│   └── test_iresnet.py
│
├── results/
│   ├── densenet/                 ← Checkpoints, logs, plots
│   └── iresnet/
│
└── README.md                     ← This file
```

---

## 📝 Key Insights

### DenseNet
1. **Memory bottleneck:** Dense connections grow quadratically
2. **Solution:** Gradient checkpointing trades compute for memory (2x slower, 50% less VRAM)
3. **Compression:** Reducing feature maps by 50% barely hurts accuracy (<0.5%)

### iResNet
1. **Post-Activation ReLU:** Causes gradient issues in deep networks
2. **Strided Conv:** First convolution loses spatial detail
3. **Aggressive Stem:** 7×7 conv too aggressive for small images (CIFAR)

**Full analysis:** [results/architectural_study.pdf](./results/architectural_study.pdf)

---

## 🤝 Contributing

Priority improvements:
- [ ] Add DenseNet variants (BC, Efficient)
- [ ] iResNet for ImageNet
- [ ] Compare with modern architectures (ConvNeXt, EfficientNet)
- [ ] Distributed training support

---

## 📄 Citation

```bibtex
@misc{sarang2024archbench,
  author = {Sarang, Yash},
  title = {ArchBench: DenseNet and iResNet Architectural Study},
  year = {2024},
  url = {https://github.com/YashSarang/deep-learning-research-suite}
}
```

---

## 📚 Related Work

- **DenseNet Paper:** [Densely Connected Convolutional Networks (Huang et al., 2017)](https://arxiv.org/abs/1608.06993)
- **iResNet Paper:** [Improved Residual Networks (He et al., 2016)](https://arxiv.org/abs/1603.05027)
- **Gradient Checkpointing:** [Training Deep Nets with Sublinear Memory Cost (Chen et al., 2016)](https://arxiv.org/abs/1604.06174)

---

**⭐ Star if you found the architectural insights useful!**
