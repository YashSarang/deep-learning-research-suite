#!/bin/bash
#SBATCH --job-name=gnr_finetune
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
# gnr_finetune.sh — LoRA Fine-Tuning of Qwen2.5-VL-7B
#
# Run with:   sbatch gnr_finetune.sh
# ─────────────────────────────────────────────────────────────────────────────

PROJECT=/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition

cd $PROJECT
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate gnr_project_env

echo "============================================================"
echo " GNR Fine-Tune — $(date)"
echo " Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

echo "[FINETUNE] Starting LoRA fine-tuning of Qwen2.5-VL-7B-Instruct..."
echo "           Dataset  : synthetic_data/synthetic_data/train.csv"
echo "           Output   : Current_implementation/qwen_mcq_finetuned/final/"
echo "           Epochs   : 3  |  LR: 2e-5  |  Batch: 2 (grad_accum=2)"
echo ""

python Current_implementation/fine_eval.py
if [ $? -ne 0 ]; then
    echo "[ERROR] fine_eval.py failed — check logs."
    exit 1
fi

echo ""
echo "[FINETUNE] Training complete."
echo "           LoRA adapter saved to: Current_implementation/qwen_mcq_finetuned/final/"
echo "           Next step: sbatch gnr_pipeline.sh  (to merge + evaluate)"
echo "============================================================"
echo " GNR Fine-Tune DONE — $(date)"
echo "============================================================"
