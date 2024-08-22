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

# EXP_NAME=exp_param_sweep-augment_smaller_crops-include_all
# EXP_NAME=exp_gradcam_loss-all_class-only_segmented_from_supervised_moco
# EXP_NAME=exp_gradcam_loss-all_class-only_segmented_from_supervised
EXP_NAME=exp_gradcam_loss-all_class-only_segmented_from_scratch
# EXP_NAME=exp_perf_drop-remove_sag_not_cluster
# EXP_NAME=exp_ssl_pretrain-moco_supervised-no_seed__linear__plane__finetuned__aug

# --dset = "train" "val" "test" "stanford" "uiowa" "chop" "stanford_image" "sickkids_silent_trial"

# Evaluate model
srun python -m src.scripts.model_eval \
    --exp_name $EXP_NAME \
    --dsets "sickkids" "sickkids_silent_trial" \
    --splits "val" "test" \
    --log_to_comet

# Evaluate DA transform
# srun python -m src.scripts.model_eval \
#     --exp_name $EXP_NAME \
#     --dset "sickkids_silent_trial" \
#     --da_transform_name "fda" \

################################################################################
#                                   GradCAM                                    #
################################################################################
# Create GradCAM
srun python -m src.data_viz.grad_cam \
    --exp_names $EXP_NAME \
    --dsets "sickkids" "sickkids_silent_trial" \
    --splits "val" "test"

# "val" "uiowa" "chop" \
#            "stanford_image" "sickkids_silent_trial" \