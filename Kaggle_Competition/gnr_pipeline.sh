#!/bin/bash
#SBATCH --job-name=gnr_pipeline
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition/logs/%x_%j.out
#SBATCH --error=/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition/logs/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# gnr_pipeline.sh — Full Pipeline: Merge → FAISS → Evaluate
#
# Run with:   sbatch gnr_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────

PROJECT=/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition
FULL_EVAL=1    # Change to 1 for the 2000-image full evaluation

cd $PROJECT
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate gnr_project_env

echo "============================================================"
echo " GNR Pipeline — $(date)"
echo " Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

# ── Step 1: Merge LoRA → standalone 7B model ─────────────────────────────────
if [ -d "models/finetuned_7b" ]; then
    echo "[STEP 1] Skipped — merged 7B model already exists at models/finetuned_7b/"
else
    echo "[STEP 1] Merging LoRA adapter into base 7B model..."
    echo "         Base   : Qwen/Qwen2.5-VL-7B-Instruct"
    echo "         LoRA   : Current_implementation/qwen_mcq_finetuned/final"
    echo "         Output : models/finetuned_7b/"
    python scripts/merge_lora.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] merge_lora.py failed — aborting."
        exit 1
    fi
    echo "[STEP 1] Merge complete."
fi

# ── Step 2: Build FAISS index ─────────────────────────────────────────────────
if [ -f "models/faiss/index.faiss" ]; then
    echo "[STEP 2] Skipped — FAISS index already exists at models/faiss/index.faiss"
else
    echo "[STEP 2] Building FAISS index from knowledge base..."
    python scripts/build_faiss.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] build_faiss.py failed — aborting."
        exit 1
    fi
    echo "[STEP 2] FAISS index built."
fi

# ── Step 3: Quick offline evaluation (50 images) ─────────────────────────────
echo ""
echo "[STEP 3] Running offline evaluation on 50 images (quick validation)..."
python scripts/evaluate_offline.py --n 50 --verbose --save
if [ $? -ne 0 ]; then
    echo "[WARN] evaluate_offline.py returned non-zero — check logs."
fi

# ── Step 4 (optional): Full 2000-image evaluation ────────────────────────────
if [ "$FULL_EVAL" -eq 1 ]; then
    echo ""
    echo "[STEP 4] Running FULL evaluation on 2000 images (~12 hr)..."
    python scripts/evaluate_offline.py --n 0 --save
    if [ $? -ne 0 ]; then
        echo "[WARN] Full evaluation returned non-zero — check logs."
    fi
else
    echo ""
    echo "[STEP 4] Skipped full 2000-image eval (set FULL_EVAL=1 to enable)."
fi

echo ""
echo "============================================================"
echo " GNR Pipeline DONE — $(date)"
echo "============================================================"
