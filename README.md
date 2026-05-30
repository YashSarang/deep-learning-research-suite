# AI Research Suite

Production-ready deep learning research spanning computer vision, natural language processing, and multimodal systems.

## Overview

This repository contains 7 independent deep learning projects, each addressing a distinct research problem with reproducible implementations. All projects are Docker-containerized for consistent environments across development and deployment.

**Repository:** [github.com/YashSarang/deep-learning-research-suite](https://github.com/YashSarang/deep-learning-research-suite)  
**Author:** Yash Sarang  
**License:** MIT

## Projects

### Computer Vision

**1. TinyLearn - C++ CNN Framework**
- Custom autograd engine with gradient tracking
- Layers: Conv2D, MaxPool, Linear, BatchNorm, ReLU
- Python bindings via Pybind11
- Result: 98.2% accuracy on MNIST in 10 epochs

**2. RemoteSense-TransferBench - Transfer Learning Study**
- 5 transfer strategies for aerial imagery classification
- Datasets: EuroSAT, UC Merced, RESISC45
- Results: Fine-tuning achieves 95.8% accuracy (vs 87.2% training from scratch)

**3. ArchBench - Architecture Comparison**
- DenseNet-121 vs ResNet-50 vs iResNet on CIFAR-10
- Memory profiling and parameter efficiency analysis
- Results: DenseNet-121 (7M params) outperforms ResNet-50 (23M params)

### Multimodal AI

**4. VLM-Examiner - Vision-Language MCQ Solver**
- CLIP visual encoding + LLaVA-1.5-7B reasoning
- ScienceQA dataset: 21k science questions with images
- Results: 95.4% accuracy (vs 88.2% text-only baseline)

### Natural Language Processing

**5. EmbedLearn - Word Embeddings**
- Three approaches: GloVe, SVD, Skip-gram
- Downstream task: Named Entity Recognition on CoNLL-2003
- Results: 85.7% F1 score with GloVe embeddings

**6. RNNDynamics - Recurrent Network Analysis**
- Gradient flow diagnostics for RNN, LSTM, and GRU
- Visualization tools: gradient norms, weight saturation, activation histograms
- Results: LSTM (81% accuracy) vs RNN (59%) on sentiment classification

**7. RAGAttention - Retrieval-Augmented Generation**
- BM25 retrieval + LLaMA 3.2-1B generation
- Memory optimization: 9.5GB reduced to 6GB VRAM
- Results: 0.84 F1 on SciFact dataset, 610ms latency

## Quick Start

### Docker Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/YashSarang/deep-learning-research-suite.git
cd deep-learning-research-suite

# Run a specific project
docker-compose up tinylearn        # C++ CNN framework
docker-compose up embedlearn       # Word embeddings
docker-compose up rag-attention    # RAG pipeline

# Access Jupyter Lab at localhost:8001-8007
```

### Manual Setup

Each project has its own requirements and setup instructions in its subdirectory README.

## Documentation

- [DOCKER.md](./DOCKER.md) - Docker setup and usage guide
- [LICENSE_INFO.md](./LICENSE_INFO.md) - MIT License rationale and comparison

## Project Structure

```
deep-learning-research-suite/
├── tinylearn/              # C++ CNN framework
├── remotesense-transferbench/  # Transfer learning
├── densenet-iresnet-study/ # Architecture comparison
├── vlm-examiner/           # Vision-language model
├── embedlearn/             # Word embeddings
├── rnn-dynamics/           # RNN diagnostics
├── rag-attention/          # RAG optimization
├── docker/                 # Docker configurations
└── docker-compose.yml      # Container orchestration
```

## Use Cases

**Research:** Baseline implementations for comparative studies  
**Education:** Course projects with reproducible environments  
**Production:** Docker-ready deployments for cloud infrastructure

## Contributing

This is a research portfolio repository. For questions or collaboration opportunities, please open an issue.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{sarang2026airesearch,
  author = {Sarang, Yash},
  title = {AI Research Suite: Production-Ready Deep Learning Projects},
  year = {2026},
  url = {https://github.com/YashSarang/deep-learning-research-suite}
}
```

## License

MIT License. See [LICENSE](./LICENSE) for details.
