#!/bin/bash
#SBATCH --job-name=gnr_test_inference
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "GNR Inference End-to-End Test"
echo "Start time: $(date)"
echo "=========================================="

cd /users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate gnr_638

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

python inference.py --test_dir data/data/sample_test_project_2/

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
