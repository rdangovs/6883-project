#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/4-2-no.out
#SBATCH --job-name=4-2-no

python main_scripts/main_jets.py --method=lin2 --baseline=transformer -de 4 -he 2