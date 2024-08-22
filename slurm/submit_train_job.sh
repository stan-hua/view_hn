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
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-augment_smaller_crops-include_all.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-with_zoomout.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-with_imbalanced_sampling.ini"

# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-without_swa.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-3_perc_train.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-5_perc_train.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-10_perc_train.ini"


# srun python -m src.scripts.model_training -c "exp_perf_drop/exp_perf_drop-remove_sag_cluster.ini"
# srun python -m src.scripts.model_training -c "exp_perf_drop/exp_perf_drop-remove_sag_not_cluster.ini"
# srun python -m src.scripts.model_training -c "exp_perf_drop/exp_perf_drop-remove_sag_cluster-grayscale-mixup.ini"

# srun python -m src.scripts.model_training -c "from_imagenet/exp_from_imagenet-augment.ini"

################################################################################
#                            Domain Generalization                             #
################################################################################
# srun python -m src.scripts.model_training -c "exp_ood_finetune/from_supervised-silent_trial.ini"
# srun python -m src.scripts.model_training -c "exp_ood_finetune/from_supervised-stanford_image.ini"
# srun python -m src.scripts.model_training -c "exp_ood_finetune/from_supervised-uiowa.ini"
# srun python -m src.scripts.model_training -c "exp_ood_finetune/from_supervised-chop.ini"


################################################################################
#                                 GradCAM Loss                                 #
################################################################################
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_supervised.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_ssl.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_scratch-imbalanced_sampler.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_scratch-remove_loss-imbalanced_sampler.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_scratch-remove_loss-using_file_exclusion.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_scratch-remove_loss-using_file_exclusion-no_aug.ini"

# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_scratch-orig.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_scratch-remove_loss.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_scratch-loss_weight=0.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-all_class-from_scratch-loss_weight=0.1.ini"

# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-pos_class-from_scratch.ini"
# srun python -m src.scripts.model_training -c "exp_gradcam_loss/gradcam_loss-pos_class-from_scratch-orig_implement.ini"


################################################################################
#                               Self-Supervised                                #
################################################################################
# srun python -m src.scripts.model_training -c "contrastive_pretraining/byol/byol_swa_no_seed.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/byol/byol_supervised.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/moco.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/moco_supervised.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/moco_supervised-same_video.ini"

# srun python -m src.scripts.model_training -c "contrastive_pretraining/tcl/tcl.ini"
srun python -m src.scripts.model_training -c "contrastive_pretraining/tcl/tcl-per_class_gmm.ini"
