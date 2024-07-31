#!/bin/bash -l
#SBATCH --job-name=eval                   # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Request one GPU
#SBATCH --cpus-per-task=6                 # Number of CPU cores per task
#SBATCH --mem=8GB
#SBATCH -o slurm/logs/slurm-%j.out

# If you want to do it in the terminal,
# salloc --job-name=stans_terminal --nodes=1 --gres=gpu:1 --cpus-per-task=6 --mem=32GB
# srun (command)

# Load any necessary modules or activate your virtual environment here
micromamba activate view

EXP_NAME=exp_param_sweep-augment_smaller_crops-include_all
# EXP_NAME=exp_param_sweep-augment_smaller_crops-include_all-before_drop
# EXP_NAME=exp_gradcam_loss-ft_from_sup__orig_implement
# EXP_NAME=exp_gradcam_loss-ft_from_sup
DSET=("train")     # "val" "test" "stanford" "uiowa" "chop" "stanford_non_seq" "sickkids_silent_trial"

# Evaluate model
srun python -m src.scripts.model_eval \
    --exp_name $EXP_NAME \
    --dset $DSET \
    --log_to_comet

# Create GradCAM
# srun python -m src.data_viz.grad_cam \
#     --exp_name $EXP_NAME \
#     --dset "val" "sickkids_silent_trial"

# "val" "uiowa" "chop" \
#            "stanford_non_seq" "sickkids_silent_trial" \