#!/bin/bash
#SBATCH --job-name=ImplicitCompletion
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err

#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=besteffort

#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

#SBATCH --time=04:00:00

source $(conda info --base)/etc/profile.d/conda.sh

conda activate utonia_clean

echo "================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "================================="

cd /mnt/aiongpfs/users/zkittel/PointCompleteRobot/data_processing 

python data_factory_scannet.py

echo "================================="
echo "Finished: $(date)"
echo "================================="