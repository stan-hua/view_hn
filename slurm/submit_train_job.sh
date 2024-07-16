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

# Train model
# srun python -m src.scripts.model_training -c "param_sweep/imagenet_augmented.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class_upscale_loss.ini"
srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class_upscale_loss__orig_implement.ini"