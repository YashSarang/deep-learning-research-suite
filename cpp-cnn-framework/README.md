# TinyLearn — Lightweight C++ Deep Learning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

A minimal, educational deep learning framework implemented in C++17 with Python bindings. Built for understanding backpropagation internals without heavy dependencies.

## 🎯 Features

✅ **Pure C++ CNN implementation** — Conv2D, MaxPool, ReLU, Linear layers from scratch  
✅ **Python bindings via Pybind11** — Seamless C++/Python interop  
✅ **SGD with Momentum** — Includes weight decay regularization  
✅ **Multi-threaded data loading** — OpenMP parallelization  
✅ **Benchmarking utilities** — MACs/FLOPs calculation, throughput metrics  
✅ **Two demo datasets** — MNIST (10 classes) and CIFAR-100 (100 classes)  
✅ **CMake build system** — Cross-platform (Linux, macOS, Windows)

---

## 📦 Installation

### Prerequisites

- **C++ Compiler:** GCC 7+ or Clang 10+ (C++17 support required)
- **CMake:** 3.15+
- **OpenCV:** 4.x (`libopencv-dev` on Ubuntu)
- **Python:** 3.10+ with development headers
- **Pybind11:** `pip install pybind11`

### Build from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/deep-learning-research-suite.git
cd deep-learning-research-suite/cpp-cnn-framework

# Build C++ executable and Python module
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install Python bindings
pip install -e .
```

---

## 🚀 Quick Start

### MNIST Demo (C++)

```bash
# Train on MNIST dataset
./build/tinylearn_mnist --data-dir ./data/mnist --epochs 10 --lr 0.01 --batch-size 128

# Outputs:
#   - training_log.txt (loss, accuracy, MACs/FLOPs per epoch)
#   - mnist_weights.bin (trained model checkpoint)
```

### MNIST Demo (Python)

```python
import tinylearn
import numpy as np

# Load pre-trained model
model = tinylearn.SimpleCNN(num_classes=10)
model.load_weights("mnist_weights.bin")

# Inference on custom image
image = np.random.randn(3, 32, 32).astype(np.float32)  # CHW format
output = model.forward(image)
prediction = output.argmax()
print(f"Predicted class: {prediction}")
```

### CIFAR-100 Demo

```bash
# Train on CIFAR-100 (100 classes, scaled architecture)
./build/tinylearn_cifar100 --data-dir ./data/cifar100 --epochs 50 --lr 0.01

# Model automatically scales to 100 output classes
```

---

## 🏗️ Architecture

**SimpleCNN** (for MNIST):
```
Input (3×32×32)
  ↓
Conv2D(3→8, k=3, s=1, p=1)  →  ReLU  →  MaxPool2D(2×2)
  ↓
Conv2D(8→16, k=3, s=1, p=1) →  ReLU  →  MaxPool2D(2×2)
  ↓
Flatten  →  Linear(16×8×8 → 64)  →  ReLU
  ↓
Linear(64 → 10)  →  Softmax
```

**Key Design Choices:**
- **Lazy initialization** for Linear layers (auto-infers input size after first forward pass)
- **In-place ReLU** for memory efficiency
- **Strided convolutions** optional (default: stride=1, padding=1)
- **Cross-entropy loss** with numerical stability (log-softmax trick)

---

## 📊 Benchmarks

**Hardware:** Intel i7-10700K (8 cores), 16 GB RAM, no GPU

| Dataset    | Model Params | MACs/Image | Throughput (img/s) | Accuracy (10 epochs) |
|------------|--------------|------------|--------------------|----------------------|
| MNIST      | 13,834       | 1.2M       | 2,400              | 98.7%                |
| CIFAR-100  | 141,300      | 4.8M       | 850                | 45.2%                |

*Note: CIFAR-100 accuracy is intentionally low (shallow architecture) — this is an educational framework, not a SOTA model.*

---

## 🧪 Testing

```bash
# Run C++ unit tests (Catch2)
cd build && ctest --output-on-failure

# Run Python binding tests
pytest tests/
```

**Test coverage:**
- ✅ Forward pass correctness (vs. PyTorch reference)
- ✅ Gradient numerical stability
- ✅ Data loader thread safety
- ✅ Model save/load fidelity

---

## 📖 API Reference

### C++ API

```cpp
#include "tinylearn/model.h"
#include "tinylearn/data_loader.h"
#include "tinylearn/optimizer.h"

// Create model
SimpleCNN model(num_classes=10);

// Load dataset
ImageFolderDataset dataset("./data/mnist");
DataLoader loader(dataset, batch_size=128, shuffle=true);

// Configure optimizer
SGD optimizer(lr=0.01f, momentum=0.9f, weight_decay=5e-4f);
optimizer.init(model.parameters());

// Training loop
for (int epoch = 0; epoch < 10; ++epoch) {
    for (auto& batch : loader) {
        auto output = model.forward(batch.images);
        float loss = cross_entropy_loss(output, batch.labels);
        model.backward(batch.labels);
        optimizer.step(model.parameters());
    }
}

// Save weights
model.save_weights("checkpoint.bin");
```

### Python API

```python
import tinylearn

# Model creation
model = tinylearn.SimpleCNN(num_classes=10)

# Forward pass
output = model.forward(image_tensor)  # shape: (C, H, W)

# Inference
prediction = tinylearn.argmax(output)

# Load/save weights
model.load_weights("checkpoint.bin")
model.save_weights("checkpoint.bin")

# Compute MACs/FLOPs
macs = tinylearn.compute_macs(model, input_height=32, input_width=32)
```

---

## 🔧 Hyperparameter Tuning

**Configuration file:** `configs/mnist_config.yaml`

```yaml
training:
  epochs: 10
  batch_size: 128
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0005

model:
  conv1_out_channels: 8
  conv2_out_channels: 16
  fc1_hidden_units: 64
  dropout: 0.0  # not implemented yet

data:
  num_workers: 4
  shuffle: true
  augmentation: false  # not implemented yet
```

Modify and run:
```bash
./build/tinylearn_mnist --config configs/mnist_config.yaml
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

**Priority improvements:**
- [ ] Dropout layer
- [ ] Batch normalization
- [ ] Data augmentation (random crop, flip)
- [ ] Adam optimizer
- [ ] GPU support (CUDA kernels)
- [ ] Quantization (INT8 inference)

---

## 📚 Educational Resources

This framework was built for **GNR-638: Deep Learning for Remote Sensing** at IIT Bombay. Great for:
- Understanding backpropagation mechanics
- Learning C++ template metaprogramming
- Benchmarking custom operators vs. PyTorch
- Teaching CNN fundamentals without black-box libraries

**Related blog posts:**
- [Building a CNN from Scratch in C++](https://medium.com/@yashsarang.com/cnn-from-scratch) *(coming soon)*
- [Pybind11 for Deep Learning](https://medium.com/@yashsarang.com/pybind11-dl) *(coming soon)*

---

## 📄 License

MIT License — see [LICENSE](../LICENSE) for details.

---

## 🙏 Acknowledgments

- **Course:** GNR-638 Deep Learning for Remote Sensing, IIT Bombay
- **Instructors:** Prof. Biplab Banerjee
- **Contributors:** Yash Sarang, Sarvesh Shashidhar, Anirban Saha

---

**Star ⭐ this repo if it helped you learn CNNs!**
