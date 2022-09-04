"""
model_training.py

Description: Used to train PyTorch models.
"""

# Standard libraries
import argparse
import logging
import os
from datetime import datetime

# Non-standard libraries
import numpy as np 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Custom libraries
from src.data import constants
from src.data_prep.utils import load_metadata
from src.data_prep.dataset import UltrasoundDataModule
from src.models.efficientnet_lstm_pl import EfficientNetLSTM
from src.models.efficientnet_pl import EfficientNetPL
from src.utilities.custom_logger import FriendlyCSVLogger


# Set logging config
logging.basicConfig(level=logging.DEBUG)

################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)


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
        "full_seq": "If flagged, trains a CNN-LSTM model on full US sequences.",
        "relative_side": "If flagged, relabels side Left/Right to First/Second "
                         "based on which appeared first per sequence.",
        "adam": "If flagged, uses Adam optimizer during training. Otherwise, "
                "uses Stochastic Gradient Descent (SGD).",
        "lr": "Learning rate of optimizer",
        "momentum": "Optimizer momentum",
        "weight_decay": "Weight decay during training",
        "n_lstm_layers": "Number of LSTM layers",
        "hidden_dim": "Number of nodes in each LSTM hidden layer",
        "bidirectional": "If flagged, LSTM will be bidirectional",
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
        "batch_size": "Batch size during a training/validation/test step",
        "shuffle": "If flagged, shuffled data within batch during training.",
        "num_workers": "Number of CPU workers used to retrieve data during "
                       "training.",
        "pin_memory": "If flagged, pins tensors on GPU to speed data loading.",
        "checkpoint": "If flagged, save last model checkpoint during training.",
        "precision": "Number of bits for precision of floating point values.",
        "grad_clip_norm": "If specified, value to normalize gradient to.",
        "stop_epoch": "Number of epochs to train model.",
        "debug": "If flagged, runs Trainer in dev mode. 1 quick batch is run."
    }

    # General arguments
    parser.add_argument("--exp_name", help=arg_help["exp_name"],
                        default=datetime.now().strftime("%m-%d-%Y %H-%M"))

    # Model and data - related arguments
    parser.add_argument("--full_seq", action="store_true",
                        help=arg_help["full_seq"])
    parser.add_argument("--relative_side", action="store_true",
                        help=arg_help["relative_side"])

    # Model arguments
    parser.add_argument("--adam", action="store_true", help=arg_help["adam"])
    parser.add_argument("--lr", default=0.001, type=float, help=arg_help["lr"])
    parser.add_argument("--momentum", default=0.9, type=float,
                        help=arg_help["momentum"])
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                        help=arg_help["weight_decay"]) 
    parser.add_argument("--n_lstm_layers", default=1, type=int,
                        help=arg_help["n_lstm_layers"])
    parser.add_argument("--hidden_dim", default=512, type=int, 
                        help=arg_help["hidden_dim"])
    parser.add_argument("--bidirectional", action="store_true",
                        help=arg_help["bidirectional"])

    # Data arguments
    parser.add_argument("--train", action="store_true", help=arg_help["train"])
    parser.add_argument("--test", action="store_true", help=arg_help["test"])
    parser.add_argument("--train_test_split", default=1.0, type=float,
                        help=arg_help["train_test_split"])
    parser.add_argument("--train_val_split", default=1.0, type=float,
                        help=arg_help["train_test_split"])
    parser.add_argument("--cross_val_folds", default=1, type=int,
                        help=arg_help["cross_val_folds"])
    parser.add_argument("--batch_size", default=32, type=int,
                        help=arg_help["batch_size"])
    parser.add_argument("--shuffle", action="store_true",
                        help=arg_help["shuffle"])
    parser.add_argument("--num_workers", default=4, type=int,
                        help=arg_help["num_workers"])
    parser.add_argument("--pin_memory", action="store_true",
                        help=arg_help["pin_memory"])

    # pl.Trainer arguments
    parser.add_argument("--checkpoint", type=bool, default=True,
                        help=arg_help["checkpoint"])
    parser.add_argument("--precision", default=32, type=int,
                        choices=[16, 32, 64], help=arg_help["precision"])
    parser.add_argument("--grad_clip_norm", default=None, type=float,
                        help=arg_help["grad_clip_norm"])
    parser.add_argument("--stop_epoch", default=50, type=int,
                        help=arg_help["stop_epoch"])
    parser.add_argument("--debug", action="store_true", help=arg_help["debug"])


################################################################################
#                           Training/Inference Flow                            #
################################################################################
def run(hparams, dm, model_cls, results_dir, train=True, test=True, fold=0,
        checkpoint=True, early_stopping=False, version_name="1"):
    """
    Perform (1) model training, and/or (2) load model and perform testing.

    Parameters
    ----------
    hparams : dict
        Contains (data-related, model-related) setup parameters for training and
        testing
    dm : pl.LightningDataModule
        Data module, which already called .setup()
    model_cls : class
        Class reference to model, used to instantiate a pl.LightningModule
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
    early_stopping : bool, optional
        If True, performs early stopping on plateau of val loss, by default
        False.
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
    model = model_cls(**hparams)

    # Loggers
    csv_logger = FriendlyCSVLogger(results_dir, name=version_name,
                                   version=str(fold))
    tensorboard_logger = TensorBoardLogger(results_dir, name=version_name,
                                           version=str(fold))
    # TODO: Consider logging experiments on Weights & Biases
    # wandb_logger = WandbLogger(save_dir=results_dir, name=version_name,
    #                            version=str(fold), project="view_hn")

    # Flag for presence of validation set
    includes_val = (hparams["train_val_split"] < 1.0) or \
        (hparams["cross_val_folds"] > 1)

    # Callbacks (model checkpoint)
    callbacks = []
    if checkpoint:
        callbacks.append(
            ModelCheckpoint(dirpath=experiment_dir, save_last=True,
                            monitor="val_loss" if includes_val else None))
    if early_stopping:
        # TODO: Implement this
        raise NotImplementedError

    # Initialize Trainer
    trainer = Trainer(default_root_dir=experiment_dir,
                      gpus=1,
                      num_sanity_val_steps=0,
                      # log_every_n_steps=100,
                      accumulate_grad_batches=hparams["accum_batches"],
                      precision=hparams['precision'],
                      gradient_clip_val=hparams['grad_clip_norm'],
                      max_epochs=hparams['stop_epoch'],
                      enable_checkpointing=checkpoint,
                      # stochastic_weight_avg=True,
                      callbacks=callbacks,
                      logger=[csv_logger, tensorboard_logger],
                      fast_dev_run=hparams["debug"],
                      )

    # Show number of patients
    num_patients_train = len(np.unique(dm.dset_to_ids['train']))
    LOGGER.info(f"[Training] Num Patients: {num_patients_train}")
    if dm.dset_to_ids['val'] is not None:
        num_patients_val = len(np.unique(dm.dset_to_ids['val']))
        LOGGER.info(f"[Validation] Num Patients: {num_patients_val}")

    # (1) Perform training
    if train:
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader() if includes_val else None
        trainer.fit(model, train_dataloaders=train_loader,
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
    hparams = {"img_size": constants.IMG_SIZE,
               "num_classes": constants.NUM_CLASSES}
    hparams.update(vars(args))

    # 0. Arguments for experiment
    experiment_hparams = {
        "train": hparams["train"],
        "test": hparams["test"],
        "checkpoint": hparams["checkpoint"],
        "version_name": hparams["exp_name"],
    }

    hparams["accum_batches"] = args.batch_size if args.full_seq else None

    # 1. Get image filenames and labels
    df_metadata = load_metadata(extract=True,
                                relative_side=hparams["relative_side"])

    # 2. Instantiate data module
    dataloader_params = {
        'batch_size': args.batch_size if not args.full_seq else 1,
        'shuffle': args.shuffle,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
    }
    dm = UltrasoundDataModule(dataloader_params, df=df_metadata,
                              dir=constants.DIR_IMAGES, **hparams)
    dm.setup()

    # 3. Specify model class
    model_cls = EfficientNetLSTM if args.full_seq else EfficientNetPL

    # 4.1 Run experiment
    if hparams["cross_val_folds"] == 1:
        run(hparams, dm, model_cls, constants.DIR_RESULTS, **experiment_hparams)
    # 4.2 Run experiment  w/ kfold cross-validation)
    else:
        for fold_idx in range(hparams["cross_val_folds"]):
            dm.set_kfold_index(fold_idx)
            run(hparams, dm, model_cls, constants.DIR_RESULTS, fold=fold_idx,
                **experiment_hparams)


if __name__ == '__main__':
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Run main
    main(ARGS)
