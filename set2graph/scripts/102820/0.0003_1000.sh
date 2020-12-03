#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/0.0003-1000.out
#SBATCH --job-name=0.0003-1000

python main_scripts/main_jets.py --method=lin2 -de 1 -he 1 -ex 10 --baseline=transformer -l 0.0003 --ma 1000