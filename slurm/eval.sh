#!/bin/bash -l
#SBATCH --job-name=eval                   # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Request one GPU
#SBATCH --cpus-per-task=6                 # Number of CPU cores per task
#SBATCH --mem=8GB
#SBATCH -o slurm/logs/slurm-eval-%j.out

# If you want to do it in the terminal,
# salloc --job-name=stans_terminal --nodes=1 --gres=gpu:1 --cpus-per-task=6 --mem=32GB
# srun (command)

# Load any necessary modules or activate your virtual environment here
micromamba activate view

# EXP_NAME=exp_param_sweep-supervised_baseline-with_zoomout
# EXP_NAME=exp_penalize_other-supervised_baseline-with_other-penalize_other
# EXP_NAME=exp_penalize_other-supervised_baseline-with_other-penalize_other-ent_loss

# EXP_NAME=exp_param_sweep-supervised_baseline-only_orig
# EXP_NAME=exp_param_sweep-supervised_baseline-only_beamform
# EXP_NAME=exp-renalview
EXP_NAME=exp-renalview-ema-long

# EXP_NAME=exp_from_imagenet-only_beamform
# EXP_NAME=exp_ssl_pretrain-tcl-per_class_gmm-only_beamform
# EXP_NAME=finetuned_moco_large_crop-only_beamform
# EXP_NAME=finetuned_moco_supervised_large_crop-only_beamform
# EXP_NAME=finetuned_byol-only_beamform.ini

# EXP_NAME=exp_ssl_pretrain-tcl-per_class_gmm-only_beamform-val_fix-large_crop
# EXP_NAME=exp_ssl_pretrain-tcl-per_class_gmm-only_beamform-large_crop-long_warmup

# Best/Last Checkpoint
CKPT_OPTION="best"

################################################################################
#                               Model Evaluation                               #
################################################################################
# --dset = "sickkids" "stanford" "sickkids_beamform" "stanford_beamform" "sickkids_image" "sickkids_silent_trial" "stanford_image" "uiowa" "chop"

# Evaluate model
# srun python -m src.scripts.model_eval \
#     --exp_name $EXP_NAME \
#     --dsets "sickkids_beamform" "sickkids_image"\
#     --label_blacklist "Other" \
#     --splits "val" \
#     --ckpt_option $CKPT_OPTION \
#     --log_to_comet
#     # --da_transform_name "clahe" \

# srun python -m src.scripts.model_eval \
#     --exp_name $EXP_NAME \
#     --dsets "sickkids_beamform" "stanford_beamform" \
#     --label_blacklist "Other" \
#     --splits "test" \
#     --ckpt_option $CKPT_OPTION \
#     --log_to_comet
#     # --da_transform_name "clahe" \

srun python -m src.scripts.model_eval \
    --exp_name $EXP_NAME \
    --dsets "sickkids_silent_trial" "stanford_image" "uiowa" "chop" \
    --label_blacklist "Other" \
    --splits "test" \
    --ckpt_option $CKPT_OPTION \
    --log_to_comet
    # --da_transform_name "clahe" \

################################################################################
#                                   GradCAM                                    #
################################################################################
# Create GradCAM
# srun python -m src.data_viz.grad_cam \
#     --exp_names $EXP_NAME \
#     --dsets "sickkids" "sickkids_beamform" "sickkids_image" \
#     --splits "val"

# "val" "uiowa" "chop" \
#            "stanford_image" "sickkids_silent_trial" \