#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/gin.out
#SBATCH --job-name=2

python -u main_pyg.py --gnn gin --step-size 8e-3 --name=gin