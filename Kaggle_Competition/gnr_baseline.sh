#!/bin/bash
#SBATCH --job-name=gnr_baseline
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
# gnr_baseline.sh — Zero-Shot Qwen2.5-VL-72B Baseline
#
# Run with:   sbatch gnr_baseline.sh
# ─────────────────────────────────────────────────────────────────────────────

PROJECT=/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition

cd $PROJECT
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate gnr_project_env

echo "============================================================"
echo " GNR Baseline — $(date)"
echo " Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"
echo "[BASELINE] Running zero-shot Qwen2.5-VL-72B on 2000 images..."
echo "           Known result: 88.75% (1775/2000 correct)"
echo ""

python Current_implementation/baseline.py
if [ $? -ne 0 ]; then
    echo "[ERROR] baseline.py failed — check logs."
    exit 1
fi

echo ""
echo "[BASELINE] Results saved to: Current_implementation/stage1_results.csv"
echo "============================================================"
echo " GNR Baseline DONE — $(date)"
echo "============================================================"
