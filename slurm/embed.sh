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

################################################################################
#                                  Constants                                   #
################################################################################
# EXP_NAME=exp_param_sweep-supervised_baseline-only_beamform
# EXP_NAME=exp_from_imagenet-only_beamform
# EXP_NAME=finetuned_moco_supervised_large_crop-only_beamform
# EXP_NAME=finetuned_moco_large_crop-only_beamform

EXP_NAME=exp_ssl_pretrain-tcl-per_class_gmm-only_beamform-large_crop-long_warmup

# Checkpoint option
CKPT_OPTION="last"      # "best" or "last"


################################################################################
#                              Create Embeddings                               #
################################################################################
# Create embeddings
# srun python -m src.scripts.embed --exp_name $EXP_NAME \
#     --dsets sickkids_beamform \
#     --splits "train" "val" "test" \
#     --ckpt_option $CKPT_OPTION

# Create UMAP
# srun python -m src.data_viz.plot_umap --exp_name $EXP_NAME --dsets sickkids_beamform --splits "test"


################################################################################
#                              Attribute Encoding                              #
################################################################################
srun python -m src.data_viz.attr_encoding \
    --exp_name $EXP_NAME \
    --dset "sickkids_beamform" \
    --ckpt_option $CKPT_OPTION
