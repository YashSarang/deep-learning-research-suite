#!/bin/bash
# GNR Project Setup Script — Cluster helper (Kalpana)
# For the official submission setup, use setup.bash instead.

echo "=========================================="
echo "Setting up Conda environment for GNR"
echo "=========================================="

# 1. Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Error: Miniconda not found at $HOME/miniconda3"
    exit 1
fi

# 2. Create environment
ENV_NAME="gnr_project_env"
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' with Python 3.11..."
    conda create -n $ENV_NAME python=3.11 -y
fi

# 3. Activate environment
conda activate $ENV_NAME

# 4. Install PyTorch with CUDA support
# Using CUDA 12.1 as a safe default for modern VLMs, or matching system if possible
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install GNR requirements
echo "Installing GNR requirements..."
cd /users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition
pip install -r requirements.txt

# 6. Install PEFT (needed for LoRA)
echo "Installing PEFT..."
pip install peft

# 7. Attempt Flash Attention installation (optional but recommended for Qwen2.5-VL)
echo "Attempting to install flash-attn (may take time)..."
# This often requires a GPU node to build properly, but we'll try
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed. Continuing without it..."

echo "=========================================="
echo "Setup complete!"
echo "Environment : $ENV_NAME"
echo "Activate    : conda activate $ENV_NAME"
echo "=========================================="
