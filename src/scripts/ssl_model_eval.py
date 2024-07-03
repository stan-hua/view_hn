"""
ssl_model_eval.py

Description: Used to automatically evaluate a SSL-trained model by
    (1) Training a Linear model on side and plane, separately
    (2) Training a LSTM + Linear model on side and plane, separately
    (3) Evaluate each of the 4+ models
"""

# Standard libraries
import argparse
import glob
import logging
import os
import shutil

# Non-standard libraries
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from jinja2 import Environment

# Custom libraries
from src.data import constants
from src.drivers import load_data, load_model, model_eval, model_training


################################################################################
#                                  Constants                                   #
################################################################################
# Set up logger
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Label parts to evaluate
LABEL_PARTS = ["side", "plane"]  # side, plane, None

# Model types to evaluate
MODEL_TYPES = ["linear_lstm"]   # linear, linear_lstm

# Options to train eval. models with/without fine-tuning backbones
FREEZE_WEIGHTS = [False]    # True, False

# Flag to perform Linear Probing - Fine-tuning
LP_FT = False

# Default args for `model_training.py` when evaluating SSL models
DEFAULT_ARGS = [
    "--self_supervised",
    "--hospital", "sickkids",
    "--train",
    "--test",
    "--train_test_split", "0.75",
    "--train_val_split", "0.75",
    "--batch_size", "16",
    "--num_workers", "4",
    "--pin_memory",
    "--precision", "16",
    "--lr", "0.001",
    "--stop_epoch", "25",
]

# Template for experiment name of a SSL evaluation model
EVAL_EXP_NAME = \
    """{{ exp_name }}__{{ model_type }}__{{ label_part }}
        {%- if lp_ft -%}
            __lp__ft
        {%- elif freeze_weights -%}
            __lp
        {%- else -%}
            __finetuned
        {%- endif -%}
        {%- if augment_training is defined and augment_training -%}
            __aug
        {%- endif -%}
    """
TEMPLATE_EVAL_EXP_NAME = Environment().from_string(EVAL_EXP_NAME)


################################################################################
#                                  Functions                                   #
################################################################################
def init(parser):
    """
    Initialize ArgumentParser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser object
    """
    arg_help = {
        "exp_name": "Base name of experiment (for SSL trained)",
        "dset": "List of dataset split or test dataset name to evaluate",
        "augment_training": "If flagged, use MoCo augmentations during "
                            "fine-tuning."
    }

    parser.add_argument("--exp_name", help=arg_help["exp_name"],
                        nargs="+",
                        required=True)
    parser.add_argument("--dset", default=[constants.DEFAULT_EVAL_DSET],
                        nargs='+',
                        help=arg_help["dset"])
    parser.add_argument("--augment_training",
                        action="store_true", default=False,
                        help=arg_help["augment_training"])


def train_model_with_kwargs(exp_name, **extra_args):
    """
    Executes `model_training.main` with default arguments and additional
    specified arguments

    Parameters
    ----------
    exp_name : str
        Experiment name
    **extra_args: dict, optional
        Keyword arguments for `model_training`
    """
    # Initialize parser
    parser = argparse.ArgumentParser()
    model_training.init(parser)

    # Copy default arguments
    args_list = DEFAULT_ARGS.copy()

    # Add experiment name
    args_list.extend(["--exp_name", exp_name])

    # Add extra keyword arguments
    for name, value in extra_args.items():
        if value is None or (isinstance(value, bool) and not value):
            continue

        args_list.append(f"--{name}")
        # If boolean, no need to add value
        if isinstance(value, bool):
            continue
        # If list
        if isinstance(value, list):
            args_list.extend([str(v) for v in value])
        # If any other type
        else:
            args_list.append(str(value))

    # Parse arguments
    args = parser.parse_args(args_list)

    # Start training
    try:
        model_training.main(args)
    except Exception as error_msg:
        # On exception, delete folder
        exp_dir = load_model.get_exp_dir(exp_name, on_error="ignore")
        if exp_dir:
            shutil.rmtree(exp_dir)

        # Re-raise error
        raise error_msg


def train_eval_model(exp_name, model_type="linear_lstm",
                     label_part="side", freeze_weights=False,
                     augment_training=False,
                     lp_ft=False,
                     **hparams):
    """
    Train single evaluation model.

    Parameters
    ----------
    exp_name : str
        Name of SSL experiment
    model_type : str, optional
        One of ("linear", "linear_lstm"), by default "linear_lstm"
    label_part : str, optional
        One of ("side", "plane"), by default "side"
    freeze_weights : bool, optional
        If True, freezes convolutional weights during training, by default False
    augment_training : bool, optional
        If True, adds augmentation during linear probing / fine-tuning, by
        default False
    lp_ft : bool, optional
        If True, trains eval. model via fine-tuning starting FROM an eval. model
        that was created via linear probing, by default False.
    **hparams : dict, optional
        Keyword arguments to pass into `model_training.main`
    """
    exp_eval_name = TEMPLATE_EVAL_EXP_NAME.render(
        exp_name=exp_name,
        model_type=model_type,
        label_part=label_part,
        freeze_weights=freeze_weights,
        lp_ft=lp_ft,
        augment_training=augment_training,
    )

    # Skip if exists
    if os.path.exists(os.path.join(constants.DIR_RESULTS, exp_eval_name)):
        LOGGER.info(f"`{exp_eval_name}` already exists! Skipping...")
        return

    # Use full sequence if LSTM
    full_seq = "lstm" in model_type

    # Attempt to train model type with specified label part
    try:
        train_model_with_kwargs(
            exp_name=exp_eval_name,
            label_part=label_part,
            freeze_weights=freeze_weights,
            full_seq=full_seq,
            augment_training=augment_training,
            **hparams,
        )
    except MisconfigurationException as error_msg:
        LOGGER.info(error_msg)
        pass
    LOGGER.info(f"`{exp_eval_name}` successfully created!")


def train_eval_models(exp_name, augment_training=False, **kwargs):
    """
    Train all possible evaluation models

    Parameters
    ----------
    exp_name : str
        Base SSL experiment name
    augment_training : bool, optional
        If True, use augmentation during fine-tuning, by default False.
    **kwargs : dict, optional
        Keyword arguments to pass into `model_training.main()`
    """
    # 0. Create copy of kwargs to avoid in-place edits
    kwargs = kwargs.copy()

    # 0. Get experiment directory, where model was trained
    model_dir = os.path.join(constants.DIR_RESULTS, exp_name)
    if not os.path.exists(model_dir):
        raise RuntimeError("`exp_name` (%s) provided does not lead to a valid "
                           "model training directory", exp_name)

    # 1. Get path to base experiment best model checkpoint
    ckpt_path = find_best_ckpt_path(model_dir)

    # 2. Get experiment hyperparameters
    hparams = load_model.get_hyperparameters(model_dir)

    # 3. Determine SSL model type
    ssl_model = hparams.get("ssl_model", "moco")
    # 3.1 Update model type, if loaded model was an eval model
    if hparams.get("ssl_eval_linear"):
        ssl_model = "linear"
    elif hparams.get("ssl_eval_linear_lstm"):
        ssl_model = "linear_lstm"

    # 4. Train models on side/plane prediction with/without fine-tuning
    for model_type in MODEL_TYPES:
        curr_kwargs = kwargs.copy()
        # 4.0 Update keyword arguments to pass in, to specify model to load
        curr_kwargs[f"ssl_eval_{model_type}"] = True

        for label_part in LABEL_PARTS:
            for freeze_weights in FREEZE_WEIGHTS:
                # Attempt to train model type with specified label part
                train_eval_model(
                    exp_name=exp_name,
                    model_type=model_type,
                    label_part=label_part,
                    freeze_weights=freeze_weights,
                    augment_training=augment_training,
                    lp_ft=False,
                    ssl_ckpt_path=ckpt_path,
                    ssl_model=ssl_model,
                    **curr_kwargs)

            # Skip, if not doing LP-FT (fine-tuning after linear probing)
            if not LP_FT:
                continue

            # Get experiment name for linear probing eval. model
            from_exp_name = TEMPLATE_EVAL_EXP_NAME.render(
                exp_name=exp_name,
                model_type=model_type,
                label_part=label_part,
                freeze_weights=True,
                lp_ft=False,
                augment_training=augment_training,
            )

            # Preparpe arguments for loading linear-probing model
            lp_ft_kwargs = curr_kwargs.copy()
            lp_ft_kwargs["from_ssl_eval"] = True
            lp_ft_kwargs["ssl_ckpt_path"] = load_model.find_best_ckpt_path(
                exp_name=from_exp_name,
            )

            # Attempt to train fine-tuned model from linear-probed model
            train_eval_model(
                exp_name=exp_name,
                model_type=model_type,
                label_part=label_part,
                freeze_weights=False,
                augment_training=augment_training,
                lp_ft=True,
                **lp_ft_kwargs,
            )


################################################################################
#                               Analyze Results                                #
################################################################################
def analyze_eval_model_preds(exp_name, dset, model_type="linear_lstm",
                             label_part="side", freeze_weights=False,
                             augment_training=False,
                             lp_ft=False):
    """
    Evaluate single evaluation model.

    Parameters
    ----------
    exp_name : str
        Name of SSL experiment
    dset : str or list, optional
        Name of evaluation split / dataset to perform inference on, by default
        constants.DEFAULT_EVAL_DSET
    model_type : str, optional
        One of ("linear", "linear_lstm"), by default "linear_lstm"
    label_part : str, optional
        One of ("side", "plane"), by default "side"
    freeze_weights : bool, optional
        If True, freezes convolutional weights during training, by default False
    augment_training : bool, optional
        If True, adds augmentation during linear probing / fine-tuning, by
        default False
    lp_ft : bool, optional
        If True, trains eval. model via fine-tuning starting FROM an eval. model
        that was created via linear probing, by default False.
    """
    exp_eval_name = TEMPLATE_EVAL_EXP_NAME.render(
            exp_name=exp_name,
            model_type=model_type,
            label_part=label_part,
            freeze_weights=freeze_weights,
            lp_ft=lp_ft,
            augment_training=augment_training,
    )

    dsets = [dset] if isinstance(dset, str) else dset
    for dset in dsets:
        # Create overwriting parameters, if external dataset desired
        overwrite_hparams = load_data.create_overwrite_hparams(
            dset)
        # Specify to mask bladder, if hospital w/o bladder labels
        mask_bladder = dset in constants.DSETS_MISSING_BLADDER

        # 1. Perform inference on dataset
        model_eval.infer_dset(
            exp_eval_name,
            dset=dset,
            mask_bladder=mask_bladder,
            **overwrite_hparams)

        # 2. Embed dataset
        model_eval.embed_dset(
            exp_eval_name,
            dset=dset,
            **overwrite_hparams,
        )

        # 3. Analyze predictions separately
        model_eval.analyze_dset_preds(
            exp_eval_name,
            dset=dset,
        )

    # 4. Create UMAP together
    model_eval.analyze_dset_preds(
        exp_eval_name,
        dset=dsets,
    )


def analyze_preds(exp_name, augment_training=False,
                  dset=constants.DEFAULT_EVAL_DSET):
    """
    Perform test prediction analysis from `model_eval` on trained evaluations
    models

    Parameters
    ----------
    exp_name : str
        Base SSL experiment name
    augment_training : bool, optional
        If True, check evaluation models fine-tuned WITH augmentation, by
        default False.
    dset : str or list, optional
        Name of evaluation split / dataset to perform inference on, by default
        constants.DEFAULT_EVAL_DSET
    """
    # Evaluate each model separately
    for model_type in MODEL_TYPES:
        for label_part in LABEL_PARTS:
            for freeze_weights in FREEZE_WEIGHTS:
                analyze_eval_model_preds(
                    exp_name=exp_name,
                    dset=dset,
                    model_type=model_type,
                    label_part=label_part,
                    freeze_weights=freeze_weights,
                    augment_training=augment_training,
                    lp_ft=False)
            # Skip, if not doing LP-FT (fine-tuning after linear probing)
            if not LP_FT:
                continue

            # Attempt to train fine-tuned model from linear-probed model
            analyze_eval_model_preds(
                exp_name=exp_name,
                dset=dset,
                model_type=model_type,
                label_part=label_part,
                freeze_weights=False,
                lp_ft=True,
                augment_training=augment_training,
            )


################################################################################
#                               Helper Functions                               #
################################################################################
def find_best_ckpt_path(path_exp_dir):
    """
    Finds the path to the best model checkpoint.

    Parameters
    ----------
    path_exp_dir : str
        Path to a trained model directory

    Returns
    -------
    str
        Path to PyTorch Lightning best model checkpoint

    Raises
    ------
    RuntimeError
        If no valid ckpt files found
    """
    ckpt_paths = glob.glob(f"{path_exp_dir}/0/*.ckpt")

    # Remove last checkpoint. NOTE: The other checkpoint is for the best epoch
    ckpt_paths = [path for path in ckpt_paths if "last.ckpt" not in path]

    if not ckpt_paths:
        raise RuntimeError("No best epoch model checkpoint (.ckpt) found!")

    if len(ckpt_paths) > 1:
        LOGGER.warning("More than 1 checkpoint file (.ckpt) found besides "
                       "last.ckpt!")

    return ckpt_paths[0]


################################################################################
#                                  Main Flow                                   #
################################################################################
def main(args):
    """
    Perform evaluation of SSL model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments
    """
    # For each base SSL experiment name provided,
    for exp_name in args.exp_name:
        # Check that exp_name leads to a valid directory
        if not os.path.exists(os.path.join(
                constants.DIR_RESULTS, exp_name)):
            raise RuntimeError("`exp_name` (%s) provided does not lead to a "
                               "SSL-trained model directory!", exp_name)

        # Get other keyword arguments for SSL eval model training
        train_kwargs = {
            k:v for k,v in vars(args).items() if k not in ["exp_name", "dset"]
        }

        # Train all evaluation models for pretrained SSL model
        train_eval_models(exp_name, **train_kwargs)

        # Analyze results of evaluation models
        analyze_preds(exp_name, dset=args.dset,
                      augment_training=args.augment_training)


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Run main
    main(ARGS)
