#!/bin/bash
#SBATCH --job-name=PointComplete
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Initialize Conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate completion_env

# Run your real Utonia script
python cluster_inference.py