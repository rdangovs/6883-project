#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/1-1-no.out
#SBATCH --job-name=1-1-no

python main_scripts/main_jets.py --method=lin2 --baseline=transformer -de 1 -he 1