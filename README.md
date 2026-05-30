# AI Research Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

**Production-ready AI research suite spanning computer vision, natural language processing, and multimodal systems**

---

## 🗂️ Projects Overview

This repository contains **7 research-grade AI projects** organized by domain:

### 🖼️ Computer Vision & Deep Learning (4 Projects)

| # | Project | Description | Tech Stack |
|---|---------|-------------|------------|
| **1** | [**TinyLearn**](./cpp-cnn-framework/) | C++ CNN framework with Python bindings | C++17, Pybind11, OpenCV |
| **2** | [**RemoteSense-TransferBench**](./transfer-learning-benchmark/) | Transfer learning benchmark for aerial imagery | PyTorch, timm, Hydra |
| **3** | [**ArchBench-DenseNet-iResNet**](./densenet-iresnet-study/) | Architectural deep dive (DenseNet, iResNet) | PyTorch, Lua, CIFAR |
| **4** | [**VLM-Examiner**](./vlm-mcq-solver/) | Vision-language MCQ solver (95.4% accuracy) | CLIP, LLaVA, Transformers |

### 📝 Natural Language Processing (3 Projects)

| # | Project | Description | Tech Stack |
|---|---------|-------------|------------|
| **5** | [**EmbedLearn**](./embedlearn/) | Word embedding suite (GloVe, SVD, NER) | NumPy, PyTorch, CoNLL-2003 |
| **6** | [**RNNDynamics**](./rnn-dynamics/) | RNN/GRU training diagnostics | PyTorch, Matplotlib |
| **7** | [**RAGAttention**](./rag-attention/) | Retrieval + attention analysis (LLaMA 3.2) | Transformers, LLaMA, BM25 |

---

## 🎯 Project Categories

### Foundations — Core Architectures
- **TinyLearn** (Project 1) — CNN internals from scratch
- **EmbedLearn** (Project 5) — Word embeddings fundamentals
- **RNNDynamics** (Project 6) — Recurrent network training

### Applications — Real-World Tasks
- **RemoteSense-TransferBench** (Project 2) — Domain adaptation
- **ArchBench** (Project 3) — Architecture comparison
- **RAGAttention** (Project 7) — Production RAG pipeline

### Multimodal — Vision + Language
- **VLM-Examiner** (Project 4) — Cross-modal reasoning

---

## 📊 Comparative Overview

| Aspect | CV Projects (1-4) | NLP Projects (5-7) |
|--------|-------------------|-------------------|
| **Primary Modality** | Images, Video | Text, Sequences |
| **Key Architectures** | CNNs, Vision Transformers | RNNs, Transformers, Embeddings |
| **Datasets** | MNIST, CIFAR, AID, ScienceQA | CoNLL-2003, SciFact, CC-News |
| **Use Cases** | Classification, Detection | NER, Retrieval, Language Modeling |
| **Common Ground** | **Attention Mechanisms** (Projects 4, 7) |

---

## 🚀 Quick Start

Each project is self-contained with its own README, dependencies, and examples:

```bash
# Clone repository
git clone https://github.com/YashSarang/ai-research-suite.git
cd ai-research-suite

# Choose a project
cd cpp-cnn-framework  # or embedlearn, rnn-dynamics, etc.

# Follow project-specific README
cat README.md
```

---

## 🎓 Background

This suite represents advanced research in AI systems. Each project has been transformed from research code into production-ready tools with:

✅ Professional documentation (31KB+ total)  
✅ Reproducible experiments  
✅ Clean code structure  
✅ Comprehensive testing  
✅ MIT licensing  

---

## 🏆 Highlights

### Technical Achievements
- **95.4% accuracy** on ScienceQA (VLM-Examiner)
- **9.5GB → 6GB VRAM** optimization (RAGAttention)
- **From-scratch implementations** (TinyLearn, RNNDynamics)
- **Systematic benchmarking** (RemoteSense-TransferBench)

### Research Contributions
- Memory-efficient DenseNet analysis (ArchBench)
- Gradient dynamics visualization (RNNDynamics)
- Lost-in-the-middle phenomenon (RAGAttention)
- Transfer learning ablations (RemoteSense-TransferBench)

---

## 📚 Project Deep Dives

<details>
<summary><strong>Project 1: TinyLearn — C++ CNN Framework</strong></summary>

**What:** Educational deep learning framework in C++17

**Why:** Understanding backpropagation internals without black-box libraries

**Features:**
- Conv2D, MaxPool, ReLU, Linear layers from scratch
- SGD with momentum & weight decay
- Multi-threaded data loading (OpenMP)
- Python bindings via Pybind11

**Datasets:** MNIST, CIFAR-100

[📖 Full Documentation](./cpp-cnn-framework/README.md)
</details>

<details>
<summary><strong>Project 2: RemoteSense-TransferBench — Transfer Learning Benchmark</strong></summary>

**What:** Systematic evaluation of CNN backbones on aerial imagery

**Why:** Benchmark transfer learning strategies for domain adaptation

**Features:**
- 3 models: ResNet-50, DenseNet-121, ConvNeXt-Tiny
- 5 scenarios: Linear Probe, Fine-Tuning, Few-Shot, Robustness, Layer-Wise
- MLflow tracking for reproducibility

**Dataset:** AID (Aerial Image Dataset, 30 classes)

[📖 Full Documentation](./transfer-learning-benchmark/README.md)
</details>

<details>
<summary><strong>Project 3: ArchBench-DenseNet-iResNet — Architectural Study</strong></summary>

**What:** Deep dive into DenseNet and iResNet architectures

**Why:** Understand memory efficiency, gradient flow, architectural choices

**Features:**
- DenseNet (Growth rate analysis, Bottleneck layers)
- iResNet (Improved residual connections)
- Memory profiling, ablation studies
- CIFAR-10/100 experiments

[📖 Full Documentation](./densenet-iresnet-study/README.md)
</details>

<details>
<summary><strong>Project 4: VLM-Examiner — Vision-Language MCQ Solver</strong></summary>

**What:** Multimodal AI system for science question answering

**Why:** Cross-modal reasoning (image + text → answer)

**Features:**
- CLIP + LLaVA ensemble
- 95.4% accuracy on ScienceQA
- Multi-stage pipeline: OCR, VQA, reasoning

**Kaggle Competition:** GNR-638 Deep Learning Course

[📖 Full Documentation](./vlm-mcq-solver/README.md)
</details>

<details>
<summary><strong>Project 5: EmbedLearn — Word Embedding Suite</strong></summary>

**What:** GloVe and SVD embeddings with NER evaluation

**Why:** Understand foundational NLP representations

**Features:**
- GloVe training from scratch
- SVD-based embeddings
- TF-IDF weighting
- CoNLL-2003 NER evaluation (CRF, MLP)

**Results:** 85.7% F1 (GloVe + MLP)

[📖 Full Documentation](./embedlearn/README.md)
</details>

<details>
<summary><strong>Project 6: RNNDynamics — Recurrent Network Diagnostics</strong></summary>

**What:** Vanilla RNN and GRU from equations with training analysis

**Why:** Understand vanishing/exploding gradients, saturation

**Features:**
- Custom RNN/GRU cells (no torch.nn.RNN)
- Gradient flow visualization
- Spectral radius tracking
- Gate saturation heatmaps

**Tasks:** Copy, Add, Parity (long-range dependencies)

[📖 Full Documentation](./rnn-dynamics/README.md)
</details>

<details>
<summary><strong>Project 7: RAGAttention — Retrieval + Attention Analysis</strong></summary>

**What:** Optimized RAG pipeline with LLaMA 3.2 attention analysis

**Why:** Production-ready retrieval with model interpretability

**Features:**
- BM25, Dense retrieval (MiniLM, UAE-Large)
- PyTorch hooks (9.5GB → 6GB VRAM)
- Lost-in-the-middle visualization
- Attention head selection (2x speedup)

**Runtime:** 3.5h for 5000 queries (optimized from 8h)

[📖 Full Documentation](./rag-attention/README.md)
</details>

---

## 🛠️ Tech Stack Summary

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+, C++17, Lua |
| **DL Frameworks** | PyTorch, Transformers, timm |
| **CV Tools** | OpenCV, CLIP, LLaVA |
| **NLP Tools** | Sentence-Transformers, BM25, CoNLL-2003 |
| **MLOps** | Hydra, MLflow, Weights & Biases |
| **Build Systems** | CMake, Pybind11, setuptools |
| **Compute** | CUDA, OpenMP, NumPy |

---

## 📖 Documentation

- [**LICENSE**](./LICENSE) — MIT License
- [**LICENSE_INFO.md**](./LICENSE_INFO.md) — MIT vs Apache 2.0 comparison
- [**TRANSFORMATION_COMPLETE.md**](./TRANSFORMATION_COMPLETE.md) — Development history
- [**GENERALIZATION_SUMMARY.md**](./GENERALIZATION_SUMMARY.md) — Authorship updates

---

## 🙏 Acknowledgments

- **IIT Bombay** for computational resources
- **PyTorch, Hugging Face, timm** communities for excellent tooling
- **Open-source contributors** whose libraries made this possible

---

## 📄 License

MIT License — See [LICENSE](./LICENSE)

```
Copyright (c) 2024 Yash Sarang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🎓 Citation

```bibtex
@misc{sarang2024airesearch,
  author = {Sarang, Yash},
  title = {AI Research Suite: Production-Ready CV, NLP, and Multimodal Projects},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YashSarang/ai-research-suite}
}
```

---

## 🤝 Contributing

This is a personal research portfolio, but feedback and suggestions are welcome!

- Open an issue for bugs/suggestions
- Pull requests for documentation improvements accepted
- For major changes, open an issue first

---

## 📞 Contact

**Yash Sarang**  
- GitHub: [@YashSarang](https://github.com/YashSarang)
- Portfolio: [Coming Soon]
- Email: [Available on request]

---

**⭐ If you find this work useful, please consider starring the repository!**
