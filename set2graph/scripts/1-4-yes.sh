#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o logs/1-4-yes.out
#SBATCH --job-name=1-4-yes

python main_scripts/main_jets.py --method=lin2 --baseline=transformer -de 1 -he 4 -sw