#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/gcn_v_flag.out
#SBATCH --job-name=1

python -u main_pyg.py --gnn gcn-virtual --step-size 5e-3 --name=gcn_v_flag