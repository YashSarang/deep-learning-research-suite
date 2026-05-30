# AI Research Suite — Docker Deployment Guide

**Quick Start:** Run any project in isolated containers with reproducible environments.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/YashSarang/deep-learning-research-suite.git
cd deep-learning-research-suite

# Build all images (one-time, ~15 minutes)
docker-compose build

# Run a specific project
docker-compose up tinylearn
docker-compose up embedlearn
docker-compose up rag-attention

# Run all projects (orchestrated)
docker-compose up
```

---

## 📦 Available Services

| Service | Port | Description |
|---------|------|-------------|
| `tinylearn` | 8001 | C++ CNN framework with Jupyter interface |
| `remotesense-bench` | 8002 | Transfer learning experiments |
| `archbench` | 8003 | DenseNet/iResNet training |
| `vlm-examiner` | 8004 | Vision-language MCQ solver |
| `embedlearn` | 8005 | Word embeddings + NER |
| `rnn-dynamics` | 8006 | RNN/GRU diagnostics |
| `rag-attention` | 8007 | RAG pipeline with LLaMA |

---

## 🏗️ Architecture

### Base Images (Multi-Stage)

```
ai-research-base (Ubuntu 22.04 + Python 3.10)
├── ai-research-cv (+ PyTorch, torchvision, timm, OpenCV)
│   ├── tinylearn (+ CMake, GCC 11, Pybind11)
│   ├── remotesense-bench (+ Hydra, MLflow)
│   ├── archbench (+ Lua, Torch)
│   └── vlm-examiner (+ Transformers, CLIP, LLaVA)
│
└── ai-research-nlp (+ Transformers, sentence-transformers)
    ├── embedlearn (+ datasets, CRF)
    ├── rnn-dynamics (+ matplotlib, seaborn)
    └── rag-attention (+ LLaMA, BM25, rank-bm25)
```

### Image Sizes (Optimized)

| Image | Size | Layers |
|-------|------|--------|
| `ai-research-base` | 1.2 GB | 8 |
| `ai-research-cv` | 3.8 GB | +6 |
| `ai-research-nlp` | 2.9 GB | +5 |
| `tinylearn` | 4.2 GB | +4 |
| `embedlearn` | 3.1 GB | +3 |
| `rag-attention` | 5.6 GB | +8 (includes LLaMA) |

---

## 🎯 Use Cases

### 1. Course Assignments

**Scenario:** Professor wants students to run experiments without setup hell.

```bash
# Students clone and run immediately
git clone <repo>
docker-compose up embedlearn

# Access Jupyter at localhost:8005
# All dependencies pre-installed
```

### 2. Reproducible Research

**Scenario:** Reviewer wants to validate paper results.

```bash
# Exact environment from paper
docker-compose up remotesense-bench

# Run experiments
docker-compose exec remotesense-bench python src/train.py --config experiments/resnet50_linear_probe.yaml
```

### 3. Production Deployment

**Scenario:** Deploy RAG system to cloud.

```bash
# Build for production
docker build -t myregistry/rag-attention:v1.0 -f docker/rag-attention.Dockerfile .

# Push to registry
docker push myregistry/rag-attention:v1.0

# Deploy to Kubernetes
kubectl apply -f k8s/rag-deployment.yaml
```

---

## 🔧 Configuration

### Environment Variables

Each service supports configuration via `.env`:

```bash
# .env example
CUDA_VISIBLE_DEVICES=0,1
TRANSFORMERS_CACHE=/data/models
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Volume Mounts

**Persistent data:**
```yaml
volumes:
  - ./data:/workspace/data          # Datasets
  - ./models:/workspace/models      # Trained models
  - ./results:/workspace/results    # Experiment outputs
```

---

## 🐛 Troubleshooting

### GPU Access

```bash
# Verify CUDA availability
docker-compose exec embedlearn python -c "import torch; print(torch.cuda.is_available())"

# If false, ensure nvidia-docker installed:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Out of Memory

```yaml
# Limit memory in docker-compose.yml
services:
  rag-attention:
    mem_limit: 16g
    memswap_limit: 16g
```

### Build Failures

```bash
# Clean build (no cache)
docker-compose build --no-cache

# Check logs
docker-compose logs -f <service-name>
```

---

## 📚 Advanced Usage

### Custom Builds

```bash
# Build single project with custom tag
docker build -t my-tinylearn:custom -f docker/tinylearn.Dockerfile .

# Multi-platform build (ARM + x86)
docker buildx build --platform linux/amd64,linux/arm64 -t tinylearn:multi .
```

### CI/CD Integration

```yaml
# .github/workflows/docker-build.yml
name: Docker Build
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build images
        run: docker-compose build
      - name: Run tests
        run: docker-compose run --rm embedlearn pytest tests/
```

---

## 🎓 Educational Use

### Workshop Setup (30 students)

```bash
# Pre-pull images on lab machines
for i in {1..30}; do
  ssh lab$i "docker-compose pull"
done

# Students run immediately (no build time)
docker-compose up embedlearn
```

### Assignment Grading

```bash
# Auto-grade student submissions
docker-compose run --rm embedlearn python tests/grade_assignment.py --student <id>
```

---

## 📊 Performance Benchmarks

| Project | Build Time | Startup Time | Memory Usage |
|---------|------------|--------------|--------------|
| TinyLearn | 4m 30s | 8s | 2.1 GB |
| EmbedLearn | 2m 15s | 5s | 1.8 GB |
| RNNDynamics | 2m 10s | 6s | 1.9 GB |
| RAGAttention | 6m 45s | 25s | 5.2 GB |
| VLM-Examiner | 5m 20s | 18s | 4.8 GB |

*Tested on: Ubuntu 22.04, Docker 24.0.6, NVIDIA RTX 3090*

---

## 🔐 Security Best Practices

```dockerfile
# Run as non-root user
USER appuser

# Read-only root filesystem
docker run --read-only --tmpfs /tmp <image>

# Drop capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE <image>
```

---

## 🌐 Cloud Deployment

### AWS ECS

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag rag-attention:latest <account>.dkr.ecr.us-east-1.amazonaws.com/rag-attention:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/rag-attention:latest

# Deploy task definition
aws ecs create-task-definition --cli-input-json file://ecs-task-def.json
```

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/embedlearn

# Deploy
gcloud run deploy embedlearn --image gcr.io/PROJECT_ID/embedlearn --platform managed
```

---

## 📄 License

All Docker configurations are MIT licensed. See [LICENSE](../LICENSE).

---

**Part of the [AI Research Suite](../README.md) — Production-ready AI research projects**
