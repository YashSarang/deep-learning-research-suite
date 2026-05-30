# Docker Deployment Guide

This guide covers Docker-based deployment for all projects in the AI Research Suite.

## Prerequisites

- Docker 20.10 or later
- Docker Compose 2.0 or later
- (Optional) NVIDIA Docker for GPU support

## Quick Start

```bash
# Clone repository
git clone https://github.com/YashSarang/deep-learning-research-suite.git
cd deep-learning-research-suite

# Build all images
docker-compose build

# Run a specific project
docker-compose up tinylearn

# Access Jupyter Lab at the assigned port (see Port Mapping below)
```

## Port Mapping

Each project runs Jupyter Lab on a dedicated port:

| Project | Port | Access URL |
|---------|------|------------|
| TinyLearn | 8001 | http://localhost:8001 |
| RemoteSense-TransferBench | 8002 | http://localhost:8002 |
| ArchBench | 8003 | http://localhost:8003 |
| VLM-Examiner | 8004 | http://localhost:8004 |
| EmbedLearn | 8005 | http://localhost:8005 |
| RNNDynamics | 8006 | http://localhost:8006 |
| RAGAttention | 8007 | http://localhost:8007 |

## Project-Specific Instructions

### TinyLearn (C++ CNN Framework)

```bash
docker-compose up tinylearn

# Inside container:
docker-compose exec tinylearn bash
cd /workspace
mkdir build && cd build
cmake ..
make
./tinylearn
```

### EmbedLearn (Word Embeddings)

```bash
docker-compose up embedlearn

# Training GloVe embeddings:
docker-compose exec embedlearn python src/train_glove.py --corpus data/cc_news.txt --dim 300

# Named Entity Recognition:
docker-compose exec embedlearn python src/ner.py --embeddings models/glove.txt
```

### RAGAttention (LLaMA + Retrieval)

```bash
docker-compose up rag-attention

# Requires ~6GB VRAM
# Run with GPU:
docker-compose up rag-attention

# Access API:
curl -X POST http://localhost:8007/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes diabetes?", "top_k": 5}'
```

## GPU Support

To enable GPU acceleration:

1. Install NVIDIA Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

2. Uncomment the GPU configuration in docker-compose.yml:

```yaml
services:
  rag-attention:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. Run with GPU:

```bash
docker-compose up rag-attention
```

## Volumes

Shared volumes persist data across container restarts:

- `./data` - Datasets (mounted to /workspace/data)
- `./models` - Pre-trained models (mounted to /workspace/models)
- `./results` - Experiment outputs (mounted to /workspace/results)

## Image Sizes

Optimized multi-stage builds reduce image sizes:

| Project | Image Size |
|---------|------------|
| TinyLearn | 4.2 GB |
| EmbedLearn | 3.1 GB |
| RNNDynamics | 3.2 GB |
| RAGAttention | 5.6 GB |
| RemoteSense-TransferBench | 3.8 GB |
| ArchBench | 3.9 GB |
| VLM-Examiner | 5.2 GB |

## Troubleshooting

### Out of Memory

Increase Docker memory allocation in Docker Desktop settings (Preferences > Resources > Memory). Recommended: 8GB minimum, 16GB for GPU projects.

### Port Already in Use

Change port mapping in docker-compose.yml:

```yaml
services:
  tinylearn:
    ports:
      - "9001:8888"  # Change 8001 to 9001
```

### GPU Not Detected

Verify NVIDIA Docker installation:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Production Deployment

For production use:

1. Build optimized images:

```bash
docker-compose -f docker-compose.prod.yml build
```

2. Use environment variables for configuration:

```bash
export LLAMA_MODEL_PATH=/models/llama-3.2-1b
docker-compose up rag-attention
```

3. Set up logging and monitoring:

```bash
docker-compose logs -f rag-attention
```

## Manual Cleanup

Remove all containers and volumes:

```bash
docker-compose down -v
```

Remove images:

```bash
docker-compose down --rmi all
```

## Support

For issues specific to Docker deployment, check the project-specific README files or open an issue on GitHub.
