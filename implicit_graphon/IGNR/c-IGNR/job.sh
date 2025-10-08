#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=out/out.out
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_h100        # Partition/queue
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=8          # CPU cores
#SBATCH --mem=32G                  # Memory

module load 2023
module load CUDA/12.1
module load Anaconda3/2023.03

source activate env

# Run training
python train_cIGNR_two_heads.py 
