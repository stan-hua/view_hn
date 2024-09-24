"""
model_training.py

Description: Used to train PyTorch models.
"""

# Standard libraries
import argparse
import logging
import os
import random
import shutil
import sys
from datetime import datetime

# Non-standard libraries
import comet_ml             # NOTE: Recommended by Comet ML
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint, StochasticWeightAveraging, EarlyStopping
)
from lightning.pytorch.loggers import CometLogger

# Custom libraries
from src.data import constants
from src.scripts import load_data, load_model
from src.utils import config as config_utils
from src.utils.logging import FriendlyCSVLogger


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# Default random seed
SEED = None

# Comet-ML project name
COMET_PROJECT = "renal-view"


################################################################################
#                                Initialization                                #
################################################################################
def set_seed(seed=SEED, include_algos=False):
    """
    Set random seed for all models.

    Parameters
    ----------
    seed : int, optional
        Random seed. If None, don't set seed, by default SEED
    include_algos : bool, optional
        If True, forces usage of deterministic algorithms at the cost of
        performance, by default False.
    """
    # If seed is None, don't set seed
    if seed is None or seed < 0:
        LOGGER.warning(f"Random seed is not set!")
        return

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Force deterministic algorithms
    if include_algos:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    LOGGER.info(f"Success! Set random seed: {seed}")


################################################################################
#                           Training/Inference Flow                            #
################################################################################
def run(hparams, dm, results_dir=constants.DIR_RESULTS, fold=0):
    """
    Perform (1) model training, and/or (2) load model and perform testing.

    Parameters
    ----------
    hparams : dict
        Contains (data-related, model-related) setup parameters for training and
        testing
    dm : L.LightningDataModule
        Data module, which already called .setup()
    results_dir : str
        Path to directory containing trained model and/or test results
    fold : int, optional
        If performing cross-validation, supplies fold index. Index ranges
        between 0 to (num_folds - 1). If train-val-test split, remains 0, by
        default 0.
    """
    exp_name = hparams["exp_name"]

    # Create parent directory if not exists
    if not os.path.exists(f"{results_dir}/{exp_name}"):
        os.makedirs(f"{results_dir}/{exp_name}")

    # Directory for current experiment
    experiment_dir = f"{results_dir}/{exp_name}/{fold}"

    # Check if experiment run exists (i.e., resuming training / evaluation)
    run_exists = os.path.exists(experiment_dir)
    if run_exists and os.listdir(experiment_dir):
        LOGGER.info("Found pre-existing experiment directory! Resuming training/evaluation...")

    # Loggers
    loggers = []
    # If specified, use Comet ML for logging
    if hparams.get("use_comet_logger"):
        if hparams.get("debug"):
            LOGGER.info("Comet ML Logger disabled during debugging...")
            hparams["use_comet_logger"] = False
        elif not os.environ.get("COMET_API_KEY"):
            LOGGER.error(
                "Please set `COMET_API_KEY` environment variable before running! "
                "Or set `use_comet_logger` to false in config file..."
            )
        else:
            exp_key = None
            # If run exists, get stored experiment key to resume logging
            if run_exists:
                old_hparams = load_model.get_hyperparameters(
                    exp_name=hparams["exp_name"], on_error="raise")
                exp_key = old_hparams.get("comet_exp_key")

            # Set up LOGGER
            comet_logger = CometLogger(
                api_key=os.environ["COMET_API_KEY"],
                project_name=COMET_PROJECT,
                experiment_name=hparams["exp_name"],
                experiment_key=exp_key
            )

            # Store experiment key
            hparams["comet_exp_key"] = comet_logger.experiment.get_key()

            # Add tags
            tags = hparams.get("tags")
            if tags:
                comet_logger.experiment.add_tags(tags)
            loggers.append(comet_logger)
    # Add custom CSV logger
    loggers.append(FriendlyCSVLogger(results_dir, name=exp_name, version=str(fold)))

    # Flag for presence of validation set
    includes_val = (hparams["train_val_split"] < 1.0) or \
        (hparams["cross_val_folds"] > 1)

    # Callbacks
    callbacks = []
    # 1. Model checkpointing
    if hparams.get("checkpoint"):
        LOGGER.info("Storing model checkpoints...")
        callbacks.append(
            ModelCheckpoint(dirpath=experiment_dir, save_last=True,
                            monitor="val_loss" if includes_val else None))
    # 2. Stochastic Weight Averaging
    if hparams.get("swa"):
        LOGGER.info("Performing stochastic weight averaging (SWA)...")
        callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))
    # 3. Early stopping
    if hparams.get("early_stopping"):
        LOGGER.info("Performing early stopping on validation loss...")
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))

    # Initialize Trainer
    trainer = Trainer(default_root_dir=experiment_dir,
                      devices="auto", accelerator="auto",
                      num_sanity_val_steps=0,
                      log_every_n_steps=20,
                      accumulate_grad_batches=hparams["accum_batches"],
                      precision=hparams["precision"],
                      gradient_clip_val=hparams["grad_clip_norm"],
                      max_epochs=hparams["stop_epoch"],
                      enable_checkpointing=hparams["checkpoint"],
                      callbacks=callbacks,
                      logger=loggers,
                      fast_dev_run=hparams["debug"],
                      )

    # Show data stats
    LOGGER.info(f"[Training] Num Patients: {len(dm.get_patient_ids('train'))}")
    LOGGER.info(f"[Training] Num Images: {dm.size('train')}")
    if dm.size("val") > 0:
        LOGGER.info(f"[Validation] Num Patients: {len(dm.get_patient_ids('val'))}")

    # Create model (from scratch) or load pretrained
    model = load_model.load_model(hparams=hparams)

    # 1. Perform training
    if hparams["train"]:
        # If resuming training
        ckpt_path = None
        if run_exists:
            ckpt_path = "last"

        # Create dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader() if includes_val else None

        # Perform training
        try:
            trainer.fit(model, train_dataloaders=train_loader,
                        val_dataloaders=val_loader,
                        ckpt_path=ckpt_path)
        except KeyboardInterrupt:
            # Delete experiment directory
            if os.path.exists(experiment_dir):
                LOGGER.error("Caught keyboard interrupt! Deleting experiment directory")
                shutil.rmtree(experiment_dir)
            exit(1)

    # 2. Perform testing
    if hparams["test"]:
        trainer.test(model=model, dataloaders=dm.test_dataloader())


def main(conf):
    """
    Main method to run experiments

    Parameters
    ----------
    conf : configobj.ConfigObj
        Contains configurations needed to run experiments
    """
    # Process configuration parameters
    # Flatten nesting in configuration file
    hparams = config_utils.flatten_nested_dict(conf)

    # Overwrite number of classes
    if "num_classes" not in hparams:
        LOGGER.info("`num_classes` not provided! Providing defaults...")
        hparams["num_classes"] = len(constants.LABEL_PART_TO_CLASSES[hparams["label_part"]]["classes"]) - 1
    # Add default image size, if not specified
    if "img_size" not in hparams:
        hparams["img_size"] = constants.IMG_SIZE

    # 0. Set random seed
    set_seed(hparams.get("seed"))

    # May run out of memory on full videos, so accumulate over single batches instead
    if hparams["full_seq"]:
        hparams["accum_batches"] = hparams["batch_size"]
    else:
        hparams["accum_batches"] = hparams.get("accum_batches", 1)

    # If specified to use GradCAM loss, ensure segmentation masks are loaded
    if hparams.get("use_gradcam_loss"):
        hparams["load_seg_mask"] = True

    # 1. Set up data module
    dm = load_data.setup_data_module(hparams)

    # 2.1 Run experiment
    if hparams["cross_val_folds"] == 1:
        run(hparams, dm, constants.DIR_RESULTS)
    # 2.2 Run experiment w/ kfold cross-validation)
    else:
        for fold_idx in range(hparams["cross_val_folds"]):
            dm.set_kfold_index(fold_idx)
            run(hparams, dm, constants.DIR_RESULTS, fold=fold_idx)


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "-c", "--config",
        type=str,
        help=f"Name of configuration file under `{constants.DIR_CONFIG}`"
    )

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Load configurations
    CONF = config_utils.load_config(__file__, ARGS.config)
    LOGGER.debug("""
################################################################################
#                       Starting `model_training` Script                       #
################################################################################""")

    # 3. Run main
    main(CONF)
