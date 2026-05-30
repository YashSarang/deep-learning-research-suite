#!/bin/bash
# =============================================================================
# GNR638 — Project 2 Setup Script
#
# Group:   2  |  24M2152 · 24M2160 · 25D1598
# Course:  GNR638 — Deep Learning for Remote Sensing, IIT Bombay
#
# This script is run ONCE with internet access to:
#   1. Clone the project repository
#   2. Create the conda environment  (gnr_project_env, Python 3.11)
#   3. Install all Python dependencies
#   4. Download model weights (Qwen2.5-VL-72B-Instruct + all-MiniLM-L6-v2)
#   5. Build the FAISS retrieval index
#
# After this script completes:
#   conda activate gnr_project_env
#   python inference.py --test_dir <absolute_path_to_test_dir>
# =============================================================================

set -e   # Exit immediately on any error

REPO_URL="https://github.com/YashSarang/GNR638-Group2-24M2152-24M2160-25D1598.git"
REPO_DIR="GNR638-Group2-24M2152-24M2160-25D1598"
PROJECT_DIR="$REPO_DIR/Kaggle_Competition"
ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"

echo "============================================================"
echo " GNR638 Project 2 — Setup"
echo " $(date)"
echo "============================================================"

# ── Step 1: Clone repository ──────────────────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    echo "[Step 1] Repository already cloned at: $REPO_DIR"
    echo "         Pulling latest changes..."
    cd "$REPO_DIR"
    git pull origin main
    cd ..
else
    echo "[Step 1] Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
    echo "[Step 1] Clone complete."
fi

cd "$PROJECT_DIR"
echo "[Info] Working directory: $(pwd)"

# ── Step 2: Initialise conda ──────────────────────────────────────────────────
echo ""
echo "[Step 2] Initialising conda..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "[ERROR] Conda not found. Install Miniconda first."
    exit 1
fi

# ── Step 3: Create conda environment ─────────────────────────────────────────
echo ""
echo "[Step 3] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "         Environment '$ENV_NAME' already exists — skipping creation."
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo "[Step 3] Environment created."
fi

conda activate "$ENV_NAME"
echo "[Step 3] Activated: $CONDA_DEFAULT_ENV  |  Python: $(python --version)"

# ── Step 4: Install PyTorch (CUDA 12.6 for L40s) ─────────────────────────────
echo ""
echo "[Step 4] Installing PyTorch with CUDA 12.6 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --quiet

# ── Step 5: Install project dependencies ─────────────────────────────────────
echo ""
echo "[Step 5] Installing project requirements..."
pip install -r requirements.txt --quiet

# Flash Attention 2 (optional but recommended for speed on L40s)
echo "[Step 5] Attempting Flash Attention 2 installation..."
pip install flash-attn --no-build-isolation --quiet || \
    echo "[Step 5] Flash Attention not installed — will use SDPA (still fast on L40s)."

# ── Step 6: Download model weights ───────────────────────────────────────────
echo ""
echo "[Step 6] Downloading Qwen2.5-VL-72B-Instruct + embedding model..."
echo "         This may take 20–60 minutes depending on bandwidth (~140 GB)."
python scripts/download_hf_models.py

# ── Step 7: Build FAISS retrieval index ──────────────────────────────────────
echo ""
echo "[Step 7] Building FAISS knowledge-base index..."
python scripts/build_faiss.py

# ── Step 8: Merge LoRA adapter (if fine-tuned weights exist) ─────────────────
echo ""
if [ -d "Current_implementation/qwen_mcq_finetuned/final" ]; then
    echo "[Step 8] Merging LoRA adapter into base 7B model..."
    python scripts/merge_lora.py || echo "[Step 8] Merge failed — fine-tuned stage will be skipped."
else
    echo "[Step 8] No LoRA adapter found — fine-tuned stage will be skipped (pipeline continues with 72B only)."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!  $(date)"
echo ""
echo " To run inference:"
echo "   conda activate $ENV_NAME"
echo "   cd $PROJECT_DIR"
echo "   python inference.py --test_dir <absolute_path_to_test_dir>"
echo "============================================================"
