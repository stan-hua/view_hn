"""
model_training.py

Description: Used to train PyTorch models.
"""

# Standard libraries
import argparse
import logging
import os
import random
from datetime import datetime

# Non-standard libraries
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Custom libraries
from src.data import constants
from src.data_prep import utils 
from src.data_prep.dataset import UltrasoundDataModule
from src.data_prep.moco_dataset import MoCoDataModule
from src.data_prep.tclr_dataset import TCLRDataModule
from src.drivers import load_model
from src.utilities.custom_logger import FriendlyCSVLogger


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

# Mapping of SSL model name to corresponding data module
SSL_NAME_TO_DATA_MODULE = {
    "moco": MoCoDataModule,
    "tclr": TCLRDataModule,
}

# Default random seed
SEED = None


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
        "seed": "Random seed",

        "self_supervised": "If flagged, trains a MoCo model on US images.",
        "ssl_model": "Name of SSL model",
        "ssl_ckpt_path": "If evaluating SSL method, path to trained SSL model.",
        "ssl_eval_linear": "If flagged, trains linear classifier over "
                           "the SSL-pretrained model.",
        "ssl_eval_linear_lstm": "If flagged, trains linear + lstm classifier "
                                "over the SSL-pretrained model.",
        "freeze_weights": "If flagged, freeze weights when training "
                          "classifier over the SSL-pretrained model.",
        "from_ssl_eval": "If flagged, training SSL eval model, loading weights "
                         "from another SSL eval model.",
        "from_exp_name": "If starting training from a pretrained model, provide"
                         " model's experiment name.",

        "memory_bank_size": "Size of MoCo memory bank. Defaults to 4096.",
        "temperature": "Temperature parameter for NT-Xent loss. Defaults "
                       "to 0.1",
        "exclude_momentum_encoder": "If flagged, do not use Momentum Encoder "
                                    "when possible.",

        "full_seq": "If flagged, trains a CNN-LSTM model on full US sequences.",
        "relative_side": "If flagged, relabels side Left/Right to First/Second "
                         "based on which appeared first per sequence.",
        "same_label": "If flagged, positive samples in SSL pretraining "
                      "are images with the same label. NOTE: This logic "
                      "conflicts with `memory_bank_size` > 0.",
        "custom_collate": "Custom collate function to use for SSL pretraining. "
                          "One of (None, 'same_label'). 'same_label' pairs "
                          "images of the same label",
        "custom_ssl_loss": "Custom SSL loss to use for pretraining. One of "
                           "(None, 'soft', 'same_label')",

        "adam": "If flagged, uses Adam optimizer during training. Otherwise, "
                "uses Stochastic Gradient Descent (SGD).",
        "lr": "Learning rate of optimizer",
        "momentum": "Optimizer momentum",
        "weight_decay": "Weight decay during training",
        "augment_training": "If flagged, add MoCo random augmentations.",

        "multi_output": "If flagged, train multi-output supervised model.",
        "from_imagenet": "If flagged and supervised sequence model, trains "
                         "model from ImageNet weights",

        "n_lstm_layers": "Number of LSTM layers",
        "hidden_dim": "Number of nodes in each LSTM hidden layer",
        "bidirectional": "If flagged, LSTM will be bidirectional",

        "label_part": "Which part of label to use (side/plane). By default, "
                      "uses raw label.",
        "hospital": "Which hospital's data to use",
        "include_unlabeled": "Include unlabeled data for hospital specified.",
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
    parser.add_argument("--seed", help=arg_help["seed"],
                        default=SEED)

    # SSL Model arguments
    parser.add_argument("--self_supervised", action="store_true",
                        help=arg_help["self_supervised"])
    parser.add_argument("--ssl_model",
                        choices=load_model.SSL_NAME_TO_MODEL_CLS.keys(),
                        default="moco",
                        help=arg_help["ssl_model"])
    parser.add_argument("--ssl_ckpt_path", default=None,
                        help=arg_help["ssl_ckpt_path"])
    parser.add_argument("--ssl_eval_linear", action="store_true",
                        help=arg_help["ssl_eval_linear"])
    parser.add_argument("--ssl_eval_linear_lstm", action="store_true",
                        help=arg_help["ssl_eval_linear_lstm"])
    parser.add_argument("--freeze_weights", action="store_true",
                        help=arg_help["freeze_weights"])
    parser.add_argument("--from_ssl_eval", action="store_true",
                        help=arg_help["from_ssl_eval"])
    parser.add_argument("--from_exp_name",
                        help=arg_help["from_exp_name"])
    # SSL Model arguments
    parser.add_argument("--memory_bank_size", default=4096, type=int,
                        help=arg_help["memory_bank_size"])
    parser.add_argument("--temperature", default=0.1, type=float,
                        help=arg_help["temperature"])
    # SSL Model MoCo-specific arguments
    parser.add_argument("--exclude_momentum_encoder", action="store_true",
                        help=arg_help["exclude_momentum_encoder"])
    parser.add_argument("--custom_collate", default=None,
                        help=arg_help["custom_collate"])
    parser.add_argument("--custom_ssl_loss", default=None,
                        help=arg_help["custom_ssl_loss"])
    # SSL Data - related arguments
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
    parser.add_argument("--augment_training", action="store_true",
                        help=arg_help["augment_training"])
    # Supervised model arguments
    parser.add_argument("--multi_output", action="store_true",
                        help=arg_help["multi_output"])
    parser.add_argument("--from_imagenet", action="store_true",
                        help=arg_help["from_imagenet"])

    # LSTM-specific model arguments
    parser.add_argument("--n_lstm_layers", default=1, type=int,
                        help=arg_help["n_lstm_layers"])
    parser.add_argument("--hidden_dim", default=512, type=int, 
                        help=arg_help["hidden_dim"])
    parser.add_argument("--bidirectional", action="store_true",
                        help=arg_help["bidirectional"])

    # Data arguments
    parser.add_argument("--label_part", default=None,
                        choices=constants.LABEL_PARTS,
                        help=arg_help["label_part"])
    parser.add_argument("--hospital", default="sickkids",
                        choices=constants.HOSPITALS,
                        help=arg_help["hospital"])
    parser.add_argument("--include_unlabeled", action="store_true",
                        help=arg_help["include_unlabeled"])
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
    # Self-supervised data arguments
    parser.add_argument("--same_label", action="store_true",
                        help=arg_help["same_label"])

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


def setup_data_module(hparams, img_dir=constants.DIR_IMAGES,
                      **overwrite_hparams):
    """
    Set up data module.

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters
    img_dir : str, optional
        Path to directory containing metadata, by default constants.DIR_IMAGES
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite `hparams`

    Returns
    -------
    pytorch_lightning.LightningDataModule
    """
    # 0. If data splitting parameters are not given, assume defaults
    for split_params in ("train_val_split", "train_test_split"):
        if split_params not in hparams:
            hparams[split_params] = 0.75

    # 0. Create copy and overwrite hparams
    hparams = hparams.copy()
    hparams.update(overwrite_hparams)

    # 1. Load metadata
    df_metadata = utils.load_metadata(
        hospital=hparams["hospital"],
        extract=True,
        img_dir=img_dir,
        label_part=hparams["label_part"],
        relative_side=hparams["relative_side"],
        include_unlabeled=hparams["include_unlabeled"])

    # 2. Instantiate data module
    # 2.1 Choose appropriate class for data module
    if hparams["self_supervised"] and not \
            (hparams["ssl_eval_linear"] or hparams["ssl_eval_linear_lstm"]):
        data_module_cls = SSL_NAME_TO_DATA_MODULE[hparams["ssl_model"]]
    else:
        data_module_cls = UltrasoundDataModule
    # 2.2 Pass in specified dataloader parameters
    dataloader_params = {
        "batch_size": hparams["batch_size"] if not hparams["full_seq"] else 1,
        "shuffle": hparams["shuffle"],
        "num_workers": hparams["num_workers"],
        "pin_memory": hparams["pin_memory"],
    }
    dm = data_module_cls(dataloader_params, df=df_metadata,
                         img_dir=img_dir, **hparams)
    dm.setup()

    return dm


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
    if seed is None:
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
def run(hparams, dm, results_dir, train=True, test=True, fold=0,
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

    # Get model class
    model_cls, model_cls_kwargs = load_model.get_model_cls(hparams)
    # Instantiate model
    model = model_cls(**hparams, **model_cls_kwargs)

    # If specified, attempt to load ImageNet pretrained weights
    if hparams.get("from_imagenet") and hasattr(model, "load_imagenet_weights"):
        model.load_imagenet_weights()
    # If specified, start from a previously trained model
    elif hparams.get("from_exp_name"):
        pretrained_model = load_model.load_pretrained_from_exp_name(
            hparams.get("from_exp_name"),
            **model_cls_kwargs)
        # CASE 1: If pretrained model is the same, replace with existing model
        if type(model) == type(pretrained_model):
            load_model.overwrite_model(model, src_model=pretrained_model)
        # CASE 2: Update model weights with those from pretrained model
        # CASE 2.1: Model weight names don't need to be changed
        elif hparams.get("self_supervised"):
            pretrained_state_dict = pretrained_model.state_dict()
            # NOTE: SSL conv. backbone weights are prefixed by "conv_backbone."
            pattern = r"(conv_backbone\..*)|(temporal_backbone\..*)|(fc\..*)"
            pretrained_state_dict = load_model.prepend_prefix(
                pretrained_state_dict, "conv_backbone.",
                exclude_regex=pattern)
            model = load_model.overwrite_model(
                model,
                src_state_dict=pretrained_state_dict)
        # UNKNOWN CASE: Not supported case
        else:
            raise NotImplementedError

    # TODO: Compile model for speed-up
    # model = torch.compile(model)

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
                      log_every_n_steps=20,
                      accumulate_grad_batches=hparams["accum_batches"],
                      precision=hparams["precision"],
                      gradient_clip_val=hparams["grad_clip_norm"],
                      max_epochs=hparams["stop_epoch"],
                      enable_checkpointing=checkpoint,
                      # stochastic_weight_avg=True,
                      callbacks=callbacks,
                      logger=[csv_logger, tensorboard_logger],
                      fast_dev_run=hparams["debug"],
                      )

    # Show number of patients
    num_patients_train = len(np.unique(dm.dset_to_ids["train"]))
    LOGGER.info(f"[Training] Num Patients: {num_patients_train}")
    if dm.dset_to_ids["val"] is not None:
        num_patients_val = len(np.unique(dm.dset_to_ids["val"]))
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
    # 0. Set random seed
    set_seed(args.seed)

    # 0. Set up hyperparameters
    hparams = {
        "img_size": constants.IMG_SIZE,
        "num_classes": \
            len(constants.LABEL_PART_TO_CLASSES[args.label_part]["classes"])}
    hparams.update(vars(args))

    # 0. Arguments for experiment
    experiment_hparams = {
        "train": hparams["train"],
        "test": hparams["test"],
        "checkpoint": hparams["checkpoint"],
        "version_name": hparams["exp_name"],
    }

    hparams["accum_batches"] = args.batch_size if args.full_seq else None

    # 1. Set up data module
    dm = setup_data_module(hparams)

    # 2.1 Run experiment
    if hparams["cross_val_folds"] == 1:
        run(hparams, dm, constants.DIR_RESULTS, **experiment_hparams)
    # 2.2 Run experiment  w/ kfold cross-validation)
    else:
        for fold_idx in range(hparams["cross_val_folds"]):
            dm.set_kfold_index(fold_idx)
            run(hparams, dm, constants.DIR_RESULTS, fold=fold_idx,
                **experiment_hparams)


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Run main
    main(ARGS)
