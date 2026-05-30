# AI Research Suite — NLP Base Image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch (lighter runtime)
RUN pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install Transformers ecosystem
RUN pip3 install \
    transformers==4.36.0 \
    sentence-transformers==2.2.2 \
    datasets==2.15.0 \
    tokenizers==0.15.0 \
    huggingface-hub==0.19.4 \
    accelerate==0.25.0

# Install NLP utilities
RUN pip3 install \
    numpy==1.26.2 \
    pandas==2.1.4 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    jupyter==1.0.0 \
    jupyterlab==4.0.9

WORKDIR /workspace

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /workspace
USER appuser

CMD ["/bin/bash"]
