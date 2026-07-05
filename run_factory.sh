#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -J scannet_factory
#SBATCH -o factory_run.log

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if your miniconda is elsewhere
conda activate utonia_clean

# Run the pipeline
python data_processing/data_factory2.py