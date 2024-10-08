#!/bin/bash -l
#SBATCH --job-name=embed                  # Job name
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

# Set experiment name
# EXP_NAME=exp_ssl_pretrain-byol_accum_with_swa_no_seed
# EXP_NAME=exp_ssl_pretrain-byol-sup_plane
# EXP_NAME=exp_ssl_pretrain-moco_vanilla-no_seed
# EXP_NAME=exp_ssl_pretrain-moco_supervised-no_seed
# EXP_NAME=exp_ssl_pretrain-moco_supervised-same_video-no_seed
EXP_NAME=imagenet

# Create embeddings
srun python -m src.scripts.embed --exp_name $EXP_NAME --dsets sickkids --splits "train" "val" "test"

# Create UMAP
srun python -m src.data_viz.plot_umap --exp_name $EXP_NAME --dsets sickkids --splits "all"