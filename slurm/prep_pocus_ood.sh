#!/bin/bash -l
#SBATCH --job-name=prep_pocus_ood                   # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=32                # Number of CPU cores per task
#SBATCH --mem=16GB
#SBATCH --tmp=4GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=12:00:00

# If you want to do it in the terminal,
# salloc --job-name=stans_terminal --nodes=1 --cpus-per-task=32 --mem=16GB
# srun (command)


################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
micromamba activate view


################################################################################
#                                 Process Data                                 #
################################################################################
# srun python -m src.data_prep.scripts.prep_ood_us_dataset download_datasets pocus_atlas
srun python -m src.data_prep.scripts.prep_ood_us_dataset process_datasets pocus_atlas --overwrite