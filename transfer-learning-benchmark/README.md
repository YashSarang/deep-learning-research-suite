# RemoteSense-TransferBench — Transfer Learning Benchmark Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Systematic evaluation of **pre-trained CNN backbones** for transfer learning on **remote sensing imagery**. Implements 5 experimental scenarios across 3 ImageNet-pretrained models on the Aerial Image Dataset (AID, 30 classes).

---

## 🎯 Features

✅ **3 Pre-trained Backbones** — ResNet-50, DenseNet-121, ConvNeXt-Tiny (via `timm`)  
✅ **5 Experimental Scenarios** — Linear Probe, Fine-Tuning, Few-Shot, Corruption Robustness, Layer-Wise Probing  
✅ **Unified Configuration** — Hydra-based config system (no more duplicate code)  
✅ **Reproducible Experiments** — Deterministic training with seed control  
✅ **MLflow Tracking** — Automatic experiment logging and visualization  
✅ **Rich Visualizations** — Confusion matrices, t-SNE, attention maps, layer-wise features

---

## 📊 Experimental Scenarios

| Scenario | Description | Key Insight |
|----------|-------------|-------------|
| **S1: Linear Probe** | Freeze backbone, train only classifier | Pre-trained feature quality |
| **S2: Fine-Tuning Strategies** | Compare 4 unfreezing strategies | Optimal adaptation depth |
| **S3: Few-Shot Learning** | Train with 100% / 20% / 5% data | Data efficiency |
| **S4: Corruption Robustness** | Test on noise/blur/weather corruptions | Generalization under distribution shift |
| **S5: Layer-Wise Probing** | Train linear probes at each layer | Semantic abstraction hierarchy |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM, 8GB+ VRAM recommended

### Setup

```bash
# Navigate to project directory
cd transfer-learning-benchmark

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download AID dataset (7GB)
python scripts/download_aid_dataset.py --output-dir ./data/AID
```

---

## 🚀 Quick Start

### Run All Scenarios (Single Model)

```bash
# ResNet-50 (default)
python main.py model=resnet50

# DenseNet-121
python main.py model=densenet121

# ConvNeXt-Tiny
python main.py model=convnext_tiny
```

### Run Specific Scenario

```bash
# Linear Probe only
python main.py model=resnet50 scenarios=[linear_probe]

# Fine-Tuning with custom config
python main.py model=densenet121 scenarios=[fine_tuning] \
    training.epochs=50 training.batch_size=64
```

### Run All Models (Benchmark Suite)

```bash
bash scripts/run_all_models.sh
```

**Output:**
- Logs: `outputs/<model>/<scenario>/<timestamp>/`
- Checkpoints: `checkpoints/<model>/<scenario>/best_model.pth`
- Figures: `figures/<model>/<scenario>/`
- MLflow UI: `mlflow ui --port 5000`

---

## 📖 Configuration

**Main config:** `configs/config.yaml`

```yaml
model: resnet50  # resnet50 | densenet121 | convnext_tiny

scenarios:
  - linear_probe
  - fine_tuning
  - few_shot
  - corruption_robustness
  - layer_wise_probing

training:
  epochs: 30
  batch_size: 128
  learning_rate: 0.001
  optimizer: adam
  weight_decay: 1e-4
  lr_scheduler: cosine

data:
  dataset: AID
  num_classes: 30
  image_size: 224
  num_workers: 8
  augmentation: true

system:
  seed: 42
  device: cuda
  mixed_precision: true
  mlflow_tracking: true
```

**Model-specific overrides:** `configs/model/resnet50.yaml`, `densenet121.yaml`, `convnext_tiny.yaml`

---

## 📊 Results (on AID Dataset)

**Hardware:** NVIDIA RTX A6000 (48GB VRAM)

### Scenario 1: Linear Probe Transfer

| Model | Top-1 Acc | Top-5 Acc | Training Time |
|-------|-----------|-----------|---------------|
| ResNet-50 | 87.3% | 96.8% | 12 min |
| DenseNet-121 | 88.1% | 97.2% | 14 min |
| ConvNeXt-Tiny | **89.6%** | **97.9%** | 16 min |

### Scenario 3: Few-Shot Learning

| Model | 100% Data | 20% Data | 5% Data |
|-------|-----------|----------|---------|
| ResNet-50 | 92.4% | 78.6% | 58.3% |
| DenseNet-121 | 93.1% | 80.2% | 61.7% |
| ConvNeXt-Tiny | **94.2%** | **82.4%** | **64.9%** |

**Key Findings:**
- ConvNeXt-Tiny consistently outperforms ResNet-50 and DenseNet-121
- Fine-tuning last 2 blocks optimal (vs. full fine-tuning)
- 20% data sufficient for 80%+ of full-data accuracy
- Weather corruptions (fog, snow) most challenging

Full results: [results/benchmark_report.pdf](./results/benchmark_report.pdf)

---

## 🏗️ Architecture

```
transfer-learning-benchmark/
├── configs/                    ← Hydra configuration files
│   ├── config.yaml             ← Main experiment config
│   ├── model/                  ← Model-specific configs
│   │   ├── resnet50.yaml
│   │   ├── densenet121.yaml
│   │   └── convnext_tiny.yaml
│   └── scenario/               ← Scenario-specific configs
│       ├── linear_probe.yaml
│       ├── fine_tuning.yaml
│       └── ...
│
├── src/                        ← Source code
│   ├── models/                 ← Model wrappers
│   │   ├── base_model.py       ← Abstract base class
│   │   ├── resnet.py           ← ResNet-50 wrapper
│   │   ├── densenet.py         ← DenseNet-121 wrapper
│   │   └── convnext.py         ← ConvNeXt-Tiny wrapper
│   │
│   ├── scenarios/              ← Experiment implementations
│   │   ├── linear_probe.py
│   │   ├── fine_tuning.py
│   │   ├── few_shot.py
│   │   ├── corruption_robustness.py
│   │   └── layer_wise_probing.py
│   │
│   ├── data/                   ← Dataset & data loading
│   │   ├── aid_dataset.py      ← AID dataset wrapper
│   │   ├── transforms.py       ← Augmentations
│   │   └── corruptions.py      ← Synthetic corruptions
│   │
│   ├── utils/                  ← Utilities
│   │   ├── metrics.py          ← Accuracy, confusion matrix
│   │   ├── visualization.py    ← Plotting functions
│   │   ├── logging.py          ← MLflow integration
│   │   └── reproducibility.py  ← Seed setting, determinism
│   │
│   └── training/               ← Training loop
│       ├── trainer.py          ← Main trainer class
│       └── callbacks.py        ← Early stopping, checkpointing
│
├── scripts/                    ← Automation scripts
│   ├── download_aid_dataset.py ← Dataset downloader
│   ├── run_all_models.sh       ← Benchmark all models
│   ├── generate_report.py      ← PDF report generator
│   └── export_to_hf.py         ← Upload to Hugging Face
│
├── tests/                      ← Unit tests
│   ├── test_models.py
│   ├── test_scenarios.py
│   └── test_data.py
│
├── notebooks/                  ← Interactive analysis
│   ├── scenario_analysis.ipynb ← Explore results
│   └── feature_visualization.ipynb
│
├── main.py                     ← Entry point
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Usage Examples

### Example 1: Custom Fine-Tuning Strategy

```python
from src.models import ResNet50Model
from src.scenarios import FineTuningScenario
from src.data import AIDDataset

# Load model
model = ResNet50Model(num_classes=30, pretrained=True)

# Configure fine-tuning
scenario = FineTuningScenario(
    model=model,
    freeze_until_layer='layer3',  # Freeze first 3 blocks
    learning_rate=1e-4,
    epochs=30
)

# Load dataset
dataset = AIDDataset(root='./data/AID', split='train')

# Run experiment
results = scenario.run(dataset)
print(f"Best accuracy: {results['best_val_acc']:.2f}%")
```

### Example 2: Generate Visualization Report

```python
from src.utils.visualization import generate_scenario_report

# Generate comprehensive report for all scenarios
generate_scenario_report(
    model='convnext_tiny',
    scenarios=['linear_probe', 'fine_tuning', 'few_shot'],
    output_dir='./reports/'
)
```

---

## 🤝 Contributing

We welcome contributions! Priority areas:
- [ ] Additional pre-trained backbones (ViT, Swin Transformer)
- [ ] More datasets (NWPU-RESISC45, UCM, WHU-RS19)
- [ ] Advanced augmentation strategies
- [ ] Distributed training support
- [ ] TensorBoard integration

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📄 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{sarang2024remotesense_transferbench,
  author = {Sarang, Yash},
  title = {RemoteSense-TransferBench: Systematic Evaluation of CNN Transfer Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YashSarang/deep-learning-research-suite}
}
```

---

## 📝 Related Work

- **Original AID Paper:** [Aerial Image Dataset (Xia et al., 2017)](https://captain-whu.github.io/AID/)
- **timm Library:** [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
- **Transfer Learning Survey:** [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792)

---

## 📄 License

MIT License — see [LICENSE](../LICENSE) for details.

---

**⭐ Star this repo if it helped your research!**
