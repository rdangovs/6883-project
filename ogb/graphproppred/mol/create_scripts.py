NUM_NODES = int(12)

list_jobs = []
names = []

date = 12

vars_model = {"baseline": "",
              "transformers": "--transformers",
              "controller": "--controller"
              }

vars_task = {"gcn+flag": "python -u main_pyg.py --dataset ogbg-molpcba --gnn gcn --step-size 8e-3",
             "gcn+v+flag": "python -u main_pyg.py --dataset ogbg-molpcba --gnn gcn-virtual --step-size 8e-3",
             "gin+flag": "python -u main_pyg.py --dataset ogbg-molpcba --gnn gin --step-size 8e-3",
             "gin+v+flag": "python -u main_pyg.py --dataset ogbg-molpcba --gnn gin-virtual --step-size 8e-3"
}

for task in vars_task:
    for model in vars_model:
        names.append(f'{date}_{task}_{model}')
        job = f'{vars_task[task]} {vars_model[model]}\n'
        list_jobs.append(job)

for l in list_jobs:
    print(l)

for i in range(NUM_NODES):
    with open(f'scripts/{i}.sh', 'w') as file:
        preamble = f'#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=20\n#SBATCH -o logs/{names[i]}.out\n#SBATCH --job-name={names[i]}\n\n'
        file.write(preamble)
        file.write(list_jobs[i])
