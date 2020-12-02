#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/gin_v_flag.out
#SBATCH --job-name=3

python -u main_pyg.py --gnn gin-virtual --step-size 5e-3 --name=gin_v_flag