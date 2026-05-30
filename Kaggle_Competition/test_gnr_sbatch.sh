#!/bin/bash
#SBATCH --job-name=test_gnr_sbatch
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "Testing GNR SLURM Submission"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate gnr_project_env

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Check GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
