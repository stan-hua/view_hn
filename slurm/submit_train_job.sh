#!/bin/bash -l
#SBATCH --job-name=train                  # Job name
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
#                                  Supervised                                  #
################################################################################
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-augment_smaller_crops-short_train_30_no_swa_final_epoch.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-augment_smaller_crops-short_train_60_no_swa_sgd.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-augment_smaller_crops-short_train_30_no_swa_fp32.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-augment_small-no_seed.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-augment_smaller_crops-include_all_before_drop.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-cutmix_aug.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-mixup_aug.ini"


# srun python -m src.scripts.model_training -c "exp_perf_drop/exp_perf_drop-remove_sag_cluster.ini"
srun python -m src.scripts.model_training -c "exp_perf_drop/exp_perf_drop-remove_sag_not_cluster.ini"

# srun python -m src.scripts.model_training -c "from_imagenet/exp_from_imagenet-augment.ini"

################################################################################
#                                 GradCAM Loss                                 #
################################################################################
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class_upscale_loss.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class_upscale_loss__orig_implement.ini"
# srun python -m src.scripts.model_training -c "finetuning/gradcam_ft-sup.ini"
# srun python -m src.scripts.model_training -c "finetuning/gradcam_ft-sup_frozen.ini"
# srun python -m src.scripts.model_training -c "finetuning/gradcam_ft-sup_orig_implement.ini"


################################################################################
#                               Self-Supervised                                #
################################################################################
# srun python -m src.scripts.model_training -c "contrastive_pretraining/byol_swa_no_seed.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/byol_supervised.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco_supervised.ini"