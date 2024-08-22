#!/bin/bash -l
#SBATCH --job-name=ssl_eval               # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Request one GPU
#SBATCH --cpus-per-task=6                 # Number of CPU cores per task
#SBATCH --mem=32GB
#SBATCH -o slurm/logs/slurm-%j.out

# If you want to do it in the terminal,
# salloc --job-name=stans_terminal --nodes=1 --gres=gpu:1 --cpus-per-task=6 --mem=32GB
# srun (command)

################################################################################
#                                   Re-Queue                                   #
################################################################################
# send this batch script a SIGUSR1 60 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@60
# trap handler - resubmit ourselves
handler(){
    echo "function handler called at $(date)"
    # do whatever cleanup you want here;
    # checkpoint, sync, etc
    # sbatch "$0"
    scontrol requeue $SLURM_JOB_ID
}
# register signal handler
trap 'handler' SIGUSR1


################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
micromamba activate view


################################################################################
#                                    Script                                    #
################################################################################
# Evaluate model
# EXP_NAME=exp_ssl_pretrain-byol-sup_plane
# EXP_NAME=exp_ssl_pretrain-moco_vanilla-no_seed
EXP_NAME=exp_ssl_pretrain-moco_supervised-no_seed
# EXP_NAME=exp_ssl_pretrain-moco_supervised-same_video-no_seed

CONFIG=ssl_eval_default.ini
# CONFIG=ssl_eval-no_aug.ini
# CONFIG=ssl_eval-without_sag_cluster.ini
# CONFIG=ssl_eval-imb_sampler.ini

# Run script
srun python -m src.scripts.ssl_model_eval \
    -c $CONFIG \
    --exp_name $EXP_NAME \
    --dsets "sickkids" "stanford" "uiowa" "chop" "stanford_image" "sickkids_silent_trial" \
    --splits "test"
