#!/bin/bash -l
#SBATCH --job-name=stan_eval              # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Request one GPU
#SBATCH --cpus-per-task=6                 # Number of CPU cores per task
#SBATCH --mem=32GB
#SBATCH -o slurm/logs/slurm-%j.out

# If you want to do it in the terminal,
# salloc --job-name=stans_terminal --nodes=1 --gres=gpu:1 --cpus-per-task=6 --mem=32GB
# srun (command)

# Load any necessary modules or activate your virtual environment here
micromamba activate view

# Evaluate model
srun python -m src.scripts.model_eval \
    --exp_name exp_param_sweep-augment_smaller_crops-include_all \
    --dset "val" "test" "stanford" \
           "uiowa" "chop" \
           "stanford_non_seq" "sickkids_silent_trial" \
    --log_to_comet
