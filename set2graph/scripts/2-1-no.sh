#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/2-1-no.out
#SBATCH --job-name=2-1-no

python main_scripts/main_jets.py --method=lin2 --baseline=transformer -de 2 -he 1