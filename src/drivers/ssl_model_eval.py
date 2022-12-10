"""
ssl_model_eval.py

Description: Used to automatically evaluate a SSL-trained model by
    (1) Training a Linear model on side and plane, separately
    (2) Training a LinearLSTM model on side and plane, separately
    (3) Evaluate each of the 4+ models
"""

# Standard libraries
import argparse
import glob
import logging
import os
import shutil

# Non-standard libraries
from jinja2 import Environment

# Custom libraries
from src.data import constants
from src.drivers import load_model, model_eval, model_training


################################################################################
#                                  Constants                                   #
################################################################################
# Set up logger
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Label parts to evaluate
LABEL_PARTS = ["side", "plane"]

# Model types to evaluate
MODEL_TYPES = ["linear", "linear_lstm"]

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
    "--adam",
    "--lr", "0.001",
    "--stop_epoch", "25",
]

# Template for experiment name of a SSL evaluation model
EVAL_EXP_NAME = "{{ exp_name }}__{{ model_type }}__{{ label_part }}" \
                "{% if freeze_weights is defined and not freeze_weights %}" \
                "__finetuned{% endif %}"
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
    }

    parser.add_argument("--exp_name", help=arg_help["exp_name"],
                        required=True)


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

    # Copy default argumeents
    args_list = DEFAULT_ARGS.copy()

    # Add experiment name
    args_list.extend(["--exp_name", exp_name])

    # Add extra keyword arguments
    for name, value in extra_args.items():
        if value is None or (isinstance(value, bool) and not value):
            continue

        args_list.append(f"--{name}")
        if not isinstance(value, bool):
            args_list.append(str(value))

    # Parse arguments
    args = parser.parse_args(args_list)

    # Start training
    try:
        model_training.main(args)
    except Exception as error_msg:
        # On exception, delete folder
        exp_dir = os.path.join(constants.DIR_RESULTS, exp_name)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)

        # Re-raise error
        raise error_msg


def train_eval_models(exp_name):
    """
    Train all possible evaluation models

    Parameters
    ----------
    exp_name : str
        Base SSL experiment name
    """
    # Get path to base experiment best model checkpoint
    ckpt_path = find_best_ckpt_path(os.path.join(constants.DIR_RESULTS,
                                                 exp_name))

    # Determine SSL model type
    ssl_model = None
    for model_name in load_model.SSL_NAME_TO_MODEL_CLS.keys():
        if model_name in exp_name:
            ssl_model = model_name

    # Train models on side/plane prediction with/without fine-tuning
    for model_type in MODEL_TYPES:
        for label_part in LABEL_PARTS:
            for freeze_weights in (True, False):
                exp_eval_name = TEMPLATE_EVAL_EXP_NAME.render(
                    exp_name=exp_name,
                    model_type=model_type,
                    label_part=label_part,
                    freeze_weights=freeze_weights)

                # Skip if exists
                if os.path.exists(os.path.join(constants.DIR_RESULTS,
                                            exp_eval_name)):
                    continue

                # Use full sequence if LSTM
                full_seq = "lstm" in model_type

                # Attempt to train model type with specified label part
                train_model_with_kwargs(
                    exp_name=exp_eval_name,
                    label_part=label_part,
                    freeze_weights=freeze_weights,
                    ssl_ckpt_path=ckpt_path,
                    ssl_model=ssl_model,
                    full_seq=full_seq,
                    **{f"ssl_eval_{model_type}": True})


def analyze_preds(exp_name):
    """
    Perform test prediction analysis from `model_eval` on trained evaluations
    models

    Parameters
    ----------
    exp_name : str
        Base SSL experiment name
    """
    # Evaluate each model separately
    for model_type in MODEL_TYPES:
        for label_part in LABEL_PARTS:
            for freeze_weights in (True, False):
                exp_eval_name = TEMPLATE_EVAL_EXP_NAME.render(
                        exp_name=exp_name,
                        model_type=model_type,
                        label_part=label_part,
                        freeze_weights=freeze_weights)

                model_eval.infer_dset(exp_eval_name)
                model_eval.embed_dset(exp_eval_name)
                model_eval.analyze_dset_preds(exp_eval_name)


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
    # Check that exp_name leads to a valid directory
    if not os.path.exists(os.path.join(constants.DIR_RESULTS, args.exp_name)):
        raise RuntimeError("`exp_name` provided does not lead to a SSL-trained "
                           "model directory!")

    # Train all evaluation models for pretrained SSL model
    train_eval_models(args.exp_name)

    # Analyze results of evaluation models
    analyze_preds(args.exp_name)


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Run main
    main(ARGS)
