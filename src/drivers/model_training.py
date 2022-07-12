"""
model_training.py

Description: Used to train PyTorch models.
"""

# Standard libraries
import argparse
import os
from datetime import datetime

# Non-standard libraries
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Custom libraries
from src.data import constants
from src.data_prep.dataloaders import UltrasoundDataModule
from src.models.efficientnet_pl import EfficientNetPL
from src.utilities.custom_logger import FriendlyCSVLogger


################################################################################
#                                  Constants                                   #
################################################################################
IMG_SIZE = (256, 256)


################################################################################
#                                Initialization                                #
################################################################################
def init(parser):
    """
    Initialize arguments for ArgumentParser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser object
    """
    # Help messages
    arg_help = {
        "exp_name": "Name of experiment",
        "train": "If flagged, run experiment to train model.",
        "test": "If flagged, run experiment to evaluate a trained model.",
        "train_test_split" : "Prop. of total data to leave for training (rest "
                             "for testing).",
        "train_val_split": "Prop. of remaining training data (after taking test"
                           " set) to save as training (rest for validation). "
                           "Cannot be specified with cross_val_folds!",
        "cross_val_folds": "If num. of folds (k) specified, runs kfold "
                           "validation, training k models. Cannot be specified "
                           "with train_val_split!",
        "checkpoint": "If flagged, save last model checkpoint during training.",
        "precision": "Number of bits for precision of floating point values.",
        "grad_clip_norm": "Value to normalize (clip) gradient to.",
        "stop_epoch": "Number of epochs to train model.",
        "debug": "If flagged, runs Trainer in dev mode. 1 quick batch is run."
    }

    # (General) Experiment related
    parser.add_argument("--train", action="store_true", help=arg_help["train"])
    parser.add_argument("--test", action="store_true", help=arg_help["test"])
    parser.add_argument("--checkpoint", type=bool, default=True,
                        help=arg_help["checkpoint"])
    parser.add_argument("--exp_name", help=arg_help["exp_name"],
                        default=datetime.now().strftime("%m-%d-%Y %H-%M"))
    parser.add_argument("--train_test_split", default=0.75,
                        help=arg_help["train_test_split"])
    parser.add_argument("--train_val_split", default=0.75,
                        help=arg_help["train_test_split"])
    parser.add_argument("--cross_val_folds", default=None, type=int,
                        help=arg_help["cross_val_folds"])

    # pl.Trainer related arguments
    parser.add_argument("--precision", default=32, type=int,
                        choices=[16, 32, 64], help=arg_help["precision"])
    parser.add_argument("--grad_clip_norm", default=1.0, type=float,
                        help=arg_help["grad_clip_norm"])
    parser.add_argument("--stop_epoch", default=50, type=int,
                        help=arg_help["stop_epoch"])
    parser.add_argument("--debug", action="store_true", help=arg_help["debug"])


################################################################################
#                           Training/Inference Flow                            #
################################################################################
def run(hparams, dm, results_dir, train=True, test=True, fold=0,
        checkpoint=True, version_name="1"):
    """
    Perform (1) model training, and/or (2) load model and perform testing.

    Parameters
    ----------
    hparams : dict
        Contains (data-related, model-related) setup parameters for training and
        testing
    dm : pytorch_lightning.LightningDataModule
        Data module, which already called .setup()
    results_dir : str
        Path to directory containing trained model and/or test results
    train : bool, optional
        If True, trains a model and saves it to results_dir, by default True
    test : bool, optional
        If True, performs evaluation on trained/existing model and saves results
        to results_dir, by default True
    fold : int, optional
        If performing cross-validation, supplies fold index. Index ranges
        between 0 to (num_folds - 1). If train-val-test split, remains 0, by
        default 0.
    checkpoint : bool, optional
        If True, saves model (last epoch) to checkpoint, by default True
    version_name : str, optional
        If provided, specifies current experiment number. Version will be shown
        in the logging folder, by default "1"
    """
    # Create parent directory if not exists
    if not os.path.exists(f"{results_dir}/{version_name}"):
        os.mkdir(f"{results_dir}/{version_name}")

    # Directory for current experiment
    experiment_dir = f"{results_dir}/{version_name}/{fold}"

    # Instantiate model
    model = EfficientNetPL(img_size=hparams["img_size"])

    # Loggers
    csv_logger = FriendlyCSVLogger(results_dir, name=version_name,
                                   version=str(fold))
    tensorboard_logger = TensorBoardLogger(results_dir, name=version_name,
                                           version=str(fold))

    # Callbacks (model checkpoint)
    callbacks = []
    if checkpoint:
        # save_path = f"{experiment_dir}/checkpoints"
        callbacks.append(
            ModelCheckpoint(dirpath=experiment_dir, monitor="val_loss",
                            save_last=True))

    # Initialize Trainer
    trainer = Trainer(default_root_dir=experiment_dir,
                      gpus=1,
                      num_sanity_val_steps=0,
                      # log_every_n_steps=100,
                      accumulate_grad_batches=None,
                      precision=hparams['precision'],
                      gradient_clip_val=hparams['grad_clip_norm'],
                      max_epochs=hparams['stop_epoch'],
                      enable_checkpointing=checkpoint,
                      # stochastic_weight_avg=True,
                      callbacks=callbacks,
                      logger=[csv_logger, tensorboard_logger],
                      fast_dev_run=hparams["debug"],
                      )

    # (1) Perform training
    if train:
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader() if hparams["train_val_split"] != 1.0 \
            else None
        trainer.fit(model, train_dataloader=train_loader,
                    val_dataloaders=val_loader)

    # (2) Perform testing
    if test:
        trainer.test(model=model, test_dataloaders=dm.test_dataloader())


def main(args):
    """
    Main method to run experiments

    Parameters
    ----------
    args : argparse.Namespace
        Contains arguments needed to run experiments
    """
    # 0. Set up hyperparameters
    hparams = {"img_size": IMG_SIZE}
    hparams.update(vars(args))

    # 0. Arguments for experiment
    experiment_hparams = {
        "train": hparams["train"],
        "test": hparams["test"],
        "checkpoint": hparams["checkpoint"],
        "version_name": hparams["exp_name"]
    }

    # 1. Get image paths and labels
    df = pd.read_csv(constants.METADATA_FILE)
    df = df.rename(columns={"IMG_FILE": "filename", "revised_labels": "label"})

    # 2. Instantiate data module
    dm = UltrasoundDataModule(df=df, dir=constants.DIR_IMAGES, **hparams)
    dm.setup()

    # 3.1 Run experiment (train-val split)
    if not hparams["cross_val_folds"]:
        run(hparams, dm, constants.DIR_RESULTS, **experiment_hparams)
    # 3.2 Run experiment (kfold cross-validation)
    else:
        for fold_idx in range(hparams["cross_val_folds"]):
            dm.set_kfold_index(fold_idx)
            run(hparams, dm, constants.DIR_RESULTS, fold=fold_idx,
                **experiment_hparams)


if __name__ == '__main__':
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Run main
    main(ARGS)
