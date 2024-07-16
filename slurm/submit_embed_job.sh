#!/bin/bash -l
#SBATCH --job-name=stans_job              # Job name
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

# Create embeddings
srun python -m src.scripts.embed --exp_name exp_descent-augment --dset "train" "val"

# Create UMAP
srun python -m src.data_viz.plot_umap --exp_name exp_descent-augment --dset "train" "val"