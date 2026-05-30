# Deep Learning Research Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

A collection of **production-ready deep learning projects** spanning CNN architectures, transfer learning, and vision-language models. Developed as part of advanced research in deep learning and computer vision.

---

## 🗂️ Projects

### 1. [TinyLearn](./cpp-cnn-framework/) — C++ Deep Learning Framework
<parameter name="content">
**Minimal educational CNN framework in C++17 with Python bindings**

- ✅ Conv2D, MaxPool, ReLU, Linear layers from scratch
- ✅ SGD with momentum & weight decay
- ✅ Multi-threaded data loading (OpenMP)
- ✅ MACs/FLOPs benchmarking
- ✅ MNIST & CIFAR-100 demos

**Tech Stack:** C++17, Pybind11, OpenCV, CMake  
**Use Case:** Understanding backpropagation internals, teaching CNNs without black-box libraries

[📖 Documentation](./cpp-cnn-framework/README.md) | [🚀 Quick Start](./cpp-cnn-framework/README.md#quick-start)

---

### 2. [RemoteSense-TransferBench](./transfer-learning-benchmark/) — Transfer Learning Benchmark
**Systematic evaluation of pre-trained CNN backbones on aerial imagery**

- ✅ 3 models: ResNet-50, DenseNet-121, ConvNeXt-Tiny
- ✅ 5 scenarios: Linear Probe, Fine-Tuning, Few-Shot, Corruption Robustness, Layer-Wise Probing
- ✅ Aerial Image Dataset (AID, 30 classes)
- ✅ Reproducible experiments with MLflow tracking

**Tech Stack:** PyTorch, timm, Hydra, MLflow  
**Use Case:** Benchmarking transfer learning strategies, domain adaptation research

[📖 Documentation](./transfer-learning-benchmark/README.md) | [📊 Results](./transfer-learning-benchmark/results/)

---

### 3. [ArchBench-DenseNet-iResNet](./densenet-iresnet-study/) — Architectural Deep Dive
**Rigorous replications of DenseNet (CVPR 2017) and iResNet improvements**

- ✅ Memory-efficient DenseNet with gradient checkpointing
- ✅ iResNet: fixes for Post-Activation ReLU, Strided Conv, Aggressive Stem
- ✅ Custom vs. Official baseline comparisons
- ✅ Ablation studies + interactive notebooks

**Tech Stack:** PyTorch, Jupyter, CIFAR-10/100  
**Use Case:** Understanding architectural innovations, academic replications

[📖 Documentation](./densenet-iresnet-study/README.md) | [📝 Blog: DenseNet](https://medium.com/@yashsarang.com/the-social-butterfly-of-ai-dfed0e1ee88e) | [📝 Blog: iResNet](https://medium.com/@sarvesh260500/fixing-the-cracks-in-resnet-6c803821f9b9)

---

### 4. [VLM-Examiner](./vlm-mcq-solver/) — Vision-Language MCQ Solver
**Multi-stage VLM pipeline for academic multiple-choice questions (95.4% accuracy)**

- ✅ Qwen2.5-VL-72B (4-bit quantization) + fine-tuned 7B (LoRA)
- ✅ Confidence-based cascading (3 stages + fallback solvers)
- ✅ VRAM-efficient model swapping
- ✅ FastAPI server + Streamlit dashboard

**Tech Stack:** Qwen2.5-VL, LoRA, FAISS, SymPy, FastAPI  
**Use Case:** Automated exam grading, educational AI, VLM research

[📖 Documentation](./vlm-mcq-solver/README.md) | [🎯 Competition Results](./vlm-mcq-solver/README.md#results)

---

## 🚀 Quick Start

### Clone the Repository
```bash
git clone https://github.com/yourusername/deep-learning-research-suite.git
cd deep-learning-research-suite
```

### Option 1: Pick a Specific Project
```bash
# C++ CNN Framework
cd cpp-cnn-framework && mkdir build && cd build && cmake .. && make

# Transfer Learning Benchmark
cd transfer-learning-benchmark && pip install -r requirements.txt

# DenseNet/iResNet Study
cd densenet-iresnet-study && pip install -r requirements.txt

# VLM MCQ Solver
cd vlm-mcq-solver && bash setup.sh  # Requires 48GB VRAM
```

### Option 2: Install All Python Projects
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt  # Root requirements (coming soon)
```

---

## 📊 Project Comparison

| Project | Language | LOC | Complexity | Use Case | Audience |
|---------|----------|-----|------------|----------|----------|
| TinyLearn | C++17 | ~2,500 | High | Education | Students, Researchers |
| RemoteSense-TransferBench | Python | ~3,000 | Medium | Research | ML Practitioners |
| ArchBench | Python | ~4,500 | High | Academic | Researchers, PhD students |
| VLM-Examiner | Python | ~2,500 | Very High | Production | Industry, Education |

---

## 📚 Documentation

- **[Installation Guide](./docs/installation.md)** — System requirements, dependencies
- **[Contributing Guide](./CONTRIBUTING.md)** — How to contribute
- **[License](./LICENSE)** — MIT License
- **[Citation](./CITATION.cff)** — How to cite this work

---

## 🎓 Background

This suite represents advanced coursework and research in deep learning. Each project has been transformed from research code into production-ready tools with:

✅ Professional documentation & READMEs  
✅ CI/CD pipelines (GitHub Actions)  
✅ Unit tests & benchmarks  
✅ Reproducibility scripts  
✅ Clean architecture & code quality  

**Author:**
- Yash Sarang — [GitHub](https://github.com/YashSarang) | [LinkedIn](https://linkedin.com/in/yash-sarang)

---

## 🤝 Contributing

We welcome contributions! Whether it's:
- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Additional tests
- 💡 Research ideas

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📄 License

All projects are licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgments

- **IIT Bombay** for computational resources
- **PyTorch, timm, Hugging Face** communities for excellent tooling
- **Open-source contributors** whose libraries made this possible

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/deep-learning-research-suite&type=Date)](https://star-history.com/#yourusername/deep-learning-research-suite&Date)

---

**⭐ Star this repo if it helped your research or learning!**
