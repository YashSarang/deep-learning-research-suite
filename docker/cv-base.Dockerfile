# AI Research Suite — Computer Vision Base Image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    cmake \
    build-essential \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install CV dependencies
RUN pip3 install \
    timm==0.9.12 \
    opencv-python==4.8.1.78 \
    albumentations==1.3.1 \
    pillow==10.1.0 \
    scikit-image==0.22.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    jupyter==1.0.0 \
    jupyterlab==4.0.9 \
    ipywidgets==8.1.1

# Set working directory
WORKDIR /workspace

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /workspace
USER appuser

CMD ["/bin/bash"]
