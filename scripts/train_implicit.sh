#!/bin/bash
#SBATCH --job-name=ImplicitCompletion
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal  

#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=04:00:00

source $(conda info --base)/etc/profile.d/conda.sh

conda activate utonia_clean

echo "================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "================================="

cd /mnt/aiongpfs/users/zkittel/PointCompleteRobot/

python train_implicit.py

echo "================================="
echo "Finished: $(date)"
echo "================================="