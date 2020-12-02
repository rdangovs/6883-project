#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/gcn.out
#SBATCH --job-name=0

python -u main_pyg.py --gnn gcn --step-size 2e-3 --name=gcn