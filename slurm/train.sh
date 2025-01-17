#!/bin/bash -l
#SBATCH --job-name=train                  # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Request one GPU
#SBATCH --cpus-per-task=6                 # Number of CPU cores per task
#SBATCH --mem=32GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=12:00:00

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
# srun python -m src.scripts.model_training -c "from_imagenet/exp_from_imagenet-augment.ini"
# srun python -m src.scripts.model_training -c "from_imagenet/exp_from_imagenet-only_beamform.ini"

# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-augment_smaller_crops-include_all.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-with_zoomout.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-with_imbalanced_sampling.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-with_other.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-with_other-mixup.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-with_other-imb_sampler.ini"
# srun python -m src.scripts.model_training -c "exp_penalize_other/exp_penalize_other-supervised_baseline-with_other-penalize_other.ini"
# srun python -m src.scripts.model_training -c "exp_penalize_other/exp_penalize_other-supervised_baseline-with_other-penalize_other-ent_loss.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-with_all_images-penalize_other.ini"
# srun python -m src.scripts.model_training -c "baseline/supervised_baseline.ini"
# srun python -m src.scripts.model_training -c "baseline/supervised_baseline-with_beamform_and_canonical.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-only_orig.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-only_beamform.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/exp_param_sweep-supervised_baseline-orig_and_beamform.ini"

# srun python -m src.scripts.model_training -c "exp_supervised_ablation/exp_supervised_beamform_ablation-without_imb_sampler.ini"
# srun python -m src.scripts.model_training -c "exp_supervised_ablation/exp_supervised_beamform_ablation-without_imb_sampler-mixup.ini"
# srun python -m src.scripts.model_training -c "exp_supervised_ablation/exp_supervised_beamform_ablation-without_imb_sampler-mixup-swa.ini"
# srun python -m src.scripts.model_training -c "exp_supervised_ablation/exp_supervised_beamform_ablation-without_imb_sampler-mixup-swa-aug.ini"

# srun python -m src.scripts.model_training -c "exp_param_sweep/sample_size/exp_param_sweep-supervised_baseline-only_beamform-3_perc_train.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/sample_size/exp_param_sweep-supervised_baseline-only_beamform-5_perc_train.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/sample_size/exp_param_sweep-supervised_baseline-only_beamform-10_perc_train.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/sample_size/exp_param_sweep-supervised_baseline-only_beamform-15_perc_train.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/sample_size/exp_param_sweep-supervised_baseline-only_beamform-20_perc_train.ini"

# srun python -m src.scripts.model_training -c "exp_adult_vs_child/exp_adult_vs_child-from_adult.ini"
# srun python -m src.scripts.model_training -c "exp_adult_vs_child/adult_param_sweep/exp_adult_vs_child-from_adult-imb_sampler.ini"
# srun python -m src.scripts.model_training -c "exp_adult_vs_child/exp_adult_vs_child-from_child_subset_1.ini"
# srun python -m src.scripts.model_training -c "exp_adult_vs_child/exp_adult_vs_child-from_child_subset_2.ini"
# srun python -m src.scripts.model_training -c "exp_adult_vs_child/exp_adult_vs_child-from_child_subset_3.ini"
# srun python -m src.scripts.model_training -c "exp_adult_vs_child/exp_adult_vs_child-from_child_subset_4.ini"
# srun python -m src.scripts.model_training -c "exp_adult_vs_child/exp_adult_vs_child-from_child_subset_5.ini"

# srun python -m src.scripts.model_training -c "model_capacity/train_sup_image_effnetb5.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/crop_size/exp_param_sweep-supervised_baseline-only_beamform-crop_scale=0.5.ini"
# srun python -m src.scripts.model_training -c "exp_param_sweep/crop_size/exp_param_sweep-supervised_baseline-only_beamform-crop_scale=0.75.ini"

# srun python -m src.scripts.model_training -c "from_imagenet/exp_from_imagenet-only_beamform-saft.ini"
# srun python -m src.scripts.model_training -c "from_imagenet/exp_from_imagenet-only_beamform-low_rank_linear.ini"
# srun python -m src.scripts.model_training -c "from_imagenet/exp_from_imagenet-only_beamform-saft-low_rank_linear.ini"

srun python -m src.scripts.model_training -c "baseline/exp-supervised_baseline.ini"

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
# srun python -m src.scripts.model_training -c "contrastive_pretraining/byol/only_beamform/byol-only_beamform.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/byol/only_beamform/byol_supervised-only_beamform.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/byol/finetune/finetuned_byol-only_beamform.ini"

# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/moco.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/moco_supervised.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/moco_supervised-same_video.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/moco_supervised-with_beamform_and_canonical.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/only_beamform/moco_supervised-only_beamform.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/only_beamform/moco_supervised-same_video-only_beamform.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/only_beamform/moco-only_beamform.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/only_beamform/moco-only_beamform-large_crop.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/only_beamform/moco_supervised-only_beamform-large_crop.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/finetune/finetuned_moco_large_crop-only_beamform.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/finetune/finetuned_moco_supervised_large_crop-only_beamform.ini"

# srun python -m src.scripts.model_training -c "baseline/same_label_moco-with_beamform_and_canonical.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/same_label_moco-simple_ft-only_beamform.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/same_label_moco-simple_ft-with_beamform_and_canonical.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/same_label_moco-simple_ft-with_beamform_and_canonical-only_pretrain.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/moco/same_label_moco-simple_ft"

# srun python -m src.scripts.model_training -c "contrastive_pretraining/tcl/tcl.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/tcl/tcl-per_class_gmm.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/tcl/only_beamform/tcl-per_class_gmm-only_beamform.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/tcl/only_beamform/tcl-per_class_gmm-only_beamform-val_fix-large_crop.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/tcl/only_beamform/tcl-per_class_gmm-only_beamform-large_crop-long_warmup.ini"
# srun python -m src.scripts.model_training -c "contrastive_pretraining/tcl/only_beamform/tcl-per_class_gmm-only_beamform-long_warmup.ini"