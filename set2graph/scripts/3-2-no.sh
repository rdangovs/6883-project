#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/3-2-no.out
#SBATCH --job-name=3-2-no

python main_scripts/main_jets.py --method=lin2 --baseline=transformer -de 3 -he 2