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
from jinja2 import Environment

# Custom libraries
from config import constants
from src.scripts import load_data, load_model, model_eval, model_training
from src.utils import config as config_utils


################################################################################
#                                  Constants                                   #
################################################################################
# Set up logger
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Label parts to evaluate
LABEL_PARTS = ["plane"]  # side, plane, None

# Model types to evaluate
MODEL_TYPES = ["linear"]   # linear, linear_lstm

# Options to train eval. models with/without fine-tuning backbones
FREEZE_WEIGHTS = [False]    # True, False

# Flag to perform Linear Probing - Fine-tuning
LP_FT = False

# Template for experiment name of a SSL evaluation model
EVAL_EXP_NAME = \
    """{{ exp_name }}{{ exp_name_suffix }}-{{ model_type }}-{{ label_part }}
        {%- if lp_ft -%}
            -lp_ft
        {%- elif freeze_weights -%}
            -lp
        {%- else -%}
            -ft
        {%- endif -%}
        {%- if augment_training is defined and augment_training -%}
            -aug
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
        "exp_names": "Experiment names of SSL-pretrained models to evaluate",
        "ckpt_option": "Choice of checkpoint to load (last/best)",
        "dsets": "List of dataset names to evaluate",
        "splits": "Name of data splits for each `dset` to evaluate",
        "config": f"Name of configuration file under `{constants.DIR_CONFIG}` "
                  "to overwrite SSL pre-training parameters.",
    }
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help=arg_help["config"]
    )
    parser.add_argument("--exp_names", help=arg_help["exp_names"],
                        nargs="+",
                        required=True)
    parser.add_argument("--ckpt_option", default="best",
                        help=arg_help["ckpt_option"])
    parser.add_argument("--dsets", default=["sickkids"],
                        nargs='+',
                        help=arg_help["dsets"])
    parser.add_argument("--splits", default=["test"],
                        nargs='+',
                        help=arg_help["splits"])


def train_eval_model(hparams):
    """
    Train single evaluation model.

    Parameters
    ----------
    hparams : dict
        Contains updated hyperparameters to train SSL-eval model. Arguments
        will be passed into `model_training.main`
    """
    # Log, if exists
    exp_eval_name = hparams["exp_name"]
    if os.path.exists(os.path.join(constants.DIR_TRAIN_RUNS, exp_eval_name)):
        LOGGER.info(f"`{exp_eval_name}` already exists! Attempting to resume...")

    # Attempt training
    try:
        model_training.main(hparams)
    except RuntimeError as error_msg:
        # On exception, delete folder
        # TODO: Reconsider if this is necessary
        # exp_dir = load_model.get_exp_dir(exp_eval_name, on_error="ignore")
        # if exp_dir:
        #     shutil.rmtree(exp_dir)

        # Re-raise error
        raise error_msg
    LOGGER.info(f"`{exp_eval_name}` successfully created!")


def train_eval_models(exp_name, **overwrite_hparams):
    """
    Train all possible evaluation models

    Parameters
    ----------
    exp_name : str
        Base SSL experiment name
    **overwrite_hparams : dict, optional
        Hyperparameters to overwrite SSL training hyperparams to pass into
        `model_training.main()`
    """
    # 0. Get experiment directory, where model was trained
    model_dir = os.path.join(constants.DIR_TRAIN_RUNS, exp_name)
    if not os.path.exists(model_dir):
        raise RuntimeError("`exp_name` (%s) provided does not lead to a valid "
                           "model training directory", exp_name)

    # 1. Get path to base experiment best model checkpoint
    ckpt_path = find_best_ckpt_path(model_dir)

    # 2. Get experiment hyperparameters
    hparams = load_model.get_hyperparameters(model_dir).copy()

    # 3. Determine SSL model type
    ssl_model = hparams.get("ssl_model", "moco")
    # 3.1 Update model type, if loaded model was an eval model
    if hparams.get("ssl_eval_linear"):
        ssl_model = "linear"
        hparams["tags"].append("from_ssl_eval")
    elif hparams.get("ssl_eval_linear_lstm"):
        ssl_model = "linear_lstm"
        hparams["tags"].append("from_ssl_eval")
    else:
        hparams["tags"].append("from_ssl_pretrain")

    # 4. Train models on side/plane prediction with/without fine-tuning
    for model_type in MODEL_TYPES:
        curr_hparams = hparams.copy()
        curr_hparams.update(overwrite_hparams)

        # 4.0 Update keyword arguments to pass in, to specify model to load
        curr_hparams[f"ssl_eval_{model_type}"] = True

        for label_part in LABEL_PARTS:
            for freeze_weights in FREEZE_WEIGHTS:
                # Create new SSL eval exp. hyperparameters
                curr_hparams = prep_eval_exp_hparams(
                    curr_hparams,
                    exp_name=exp_name,
                    label_part=label_part,
                    ssl_model=ssl_model,
                    ssl_ckpt_path=ckpt_path,
                    model_type=model_type,
                    freeze_weights=freeze_weights,
                    lp_ft=False,
                )

                # Attempt to train model type with specified label part
                train_eval_model(curr_hparams)

                # Skip, if not doing LP-FT (fine-tuning after linear probing)
                if not freeze_weights or not LP_FT:
                    continue

                # Prepare arguments for loading linear-probing (LP) model
                # NOTE: Just-trained model should've been an LP model
                lp_ft_hparams = prep_eval_exp_hparams(curr_hparams, from_ssl_eval=True)

                # Attempt to train fine-tuned model from linear-probed (LP) model
                train_eval_model(lp_ft_hparams)


################################################################################
#                               Analyze Results                                #
################################################################################
def analyze_eval_model_preds(hparams, dsets, splits, **kwargs):
    """
    Evaluate single evaluation model.

    Parameters
    ----------
    hparams : dict
        Eval. experiment  and evaluation hyperparameters
    dsets : str or list, optional
        Name of evaluation dataset/s
    splits : str or list, optional
        Name of splits corresponding to each dataset to evaluate
    **kwargs : Any
        Keyword arguments to pass into all `model_eval` functions
    """
    dsets = [dsets] if isinstance(dsets, str) else dsets
    splits = [splits] if isinstance(splits, str) else splits

    for idx, curr_dset in enumerate(dsets):
        curr_split = splits[idx]

        # Create overwriting parameters, if external dataset desired
        eval_hparams = load_data.create_eval_hparams(curr_dset, curr_split)

        # 1. Perform inference on dataset
        model_eval.infer_dset(
            hparams["exp_name"],
            dset=curr_dset,
            split=curr_split,
            **kwargs,
            **eval_hparams)

        # 2. Embed dataset
        model_eval.embed_dset(
            hparams["exp_name"],
            dset=curr_dset,
            split=curr_split,
            **kwargs,
            **eval_hparams,
        )

        # 3. Analyze predictions separately
        model_eval.analyze_dset_preds(
            hparams["exp_name"],
            dsets=curr_dset,
            splits=curr_split,
            log_to_comet=True,
            **kwargs,
        )

    # 4. Create UMAP together
    model_eval.analyze_dset_preds(
        hparams["exp_name"], dsets=dsets, splits=splits, **kwargs)


def analyze_preds(exp_name, hparams, dsets="sickkids", splits="val", **kwargs):
    """
    Perform test prediction analysis from `model_eval` on trained evaluations
    models

    Parameters
    ----------
    exp_name : str
        Base SSL experiment name
    hparams : dict
        Eval experiment hyperparameters
    dsets : str or list, optional
        Name of evaluation dataset/s
    splits : str or list, optional
        Name of splits corresponding to each dataset to evaluate
    **kwargs : Any
        Keyword arguments to pass into `analyze_eval_model_preds`
    """
    # Evaluate each model separately
    for model_type in MODEL_TYPES:
        for label_part in LABEL_PARTS:
            for freeze_weights in FREEZE_WEIGHTS:
                # Create hyperparameters
                curr_hparams = prep_eval_exp_hparams(
                    hparams,
                    exp_name=exp_name,
                    label_part=label_part,
                    model_type=model_type,
                    freeze_weights=freeze_weights,
                    lp_ft=False,
                )
                # Analyze predictions of SSL eval model
                analyze_eval_model_preds(curr_hparams, dsets=dsets, splits=splits, **kwargs)

                # Skip, if not doing LP-FT (fine-tuning after linear probing)
                if not freeze_weights or not LP_FT:
                    continue

                # Prepare arguments for loading linear-probing (LP) model
                # NOTE: Just-trained model should've been an LP model
                lp_ft_hparams = prep_eval_exp_hparams(curr_hparams, from_ssl_eval=True)

                # Analyze predictions of (LP-FT) SSL eval model
                analyze_eval_model_preds(lp_ft_hparams, dsets=dsets, splits=splits, **kwargs)


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


def prep_eval_exp_hparams(hparams, **overwrite_hparams):
    """
    Prepare hyperparameters for training SSL linear probe / fine-tuning.

    Parameters
    ----------
    hparams : dict
        Contains all hyperparameters of the experiment

    Returns
    -------
    dict
        Updated dictionary of hyperparameters for SSL eval model
    """
    # Create copy of hyperparameters
    hparams = hparams.copy()

    # Update hparams
    hparams.update(overwrite_hparams)

    # Remove comet ML key
    hparams.pop("comet_exp_key", None)

    # Get experiment name suffix
    exp_name_suffix = hparams.get("exp_name_suffix")
    exp_name_suffix = f"-{exp_name_suffix}" if exp_name_suffix else ""

    # Create new eval exp. name
    exp_eval_name = TEMPLATE_EVAL_EXP_NAME.render(
        exp_name=hparams["exp_name"],
        exp_name_suffix=exp_name_suffix,
        model_type=hparams["model_type"],
        label_part=hparams["label_part"],
        freeze_weights=hparams["freeze_weights"],
        lp_ft=hparams["lp_ft"],
        augment_training=hparams["augment_training"],
    )

    # If new eval exp. name already exists, store the Comet ML experiment key
    # NOTE: Useful for resuming failed SSL eval experiments
    try:
        old_hparams = load_model.get_hyperparameters(exp_name=exp_eval_name,
                                                 on_error="raise")
        hparams["comet_exp_key"] = old_hparams.get("comet_exp_key", None)
    except:
        pass

    # Ensure number of classes is as expected
    if "num_classes" not in hparams:
        label_part = hparams["label_part"]
        hparams["num_classes"] = len(constants.LABEL_PART_TO_CLASSES[label_part]["classes"])

    # Specify to load pretrained weights
    hparams["from_exp_name"] = hparams["exp_name"]

    # Update experiment name
    hparams["exp_name"] = exp_eval_name

    return hparams


################################################################################
#                                  Main Flow                                   #
################################################################################
def main(args, hparams):
    """
    Perform evaluation of SSL model.

    Parameters
    ----------
    exp_names : list
        List of experiments to evaluate
    dsets : list
        List of datasets to evaluate
    hparams : dict
        Arguments for training SSL evaluation model
    """
    # Get dataset and splits
    exp_names = args.exp_names
    ckpt_option = args.ckpt_option
    dsets = args.dsets
    splits = args.splits
    # If only one of dset/split is > 1, assume it's meant to be broadcast
    if len(dsets) == 1 and len(splits) > 1:
        LOGGER.info("Only 1 `dset` provided! Assuming same `dset` for all `splits`...")
        dsets = dsets * len(splits)
    if len(splits) == 1 and len(dsets) > 1:
        LOGGER.info("Only 1 `split` provided! Assuming same `split` for all `dsets`...")
        splits = splits * len(dsets)

    # For each base SSL experiment name provided,
    for exp_name in exp_names:
        # Check that exp_name leads to a valid directory
        if not os.path.exists(os.path.join(
                constants.DIR_TRAIN_RUNS, exp_name)):
            raise RuntimeError("`exp_name` (%s) provided does not lead to a "
                               "SSL-trained model directory!", exp_name)

        # Train all evaluation models for pretrained SSL model
        train_eval_models(exp_name, **hparams)

        # Analyze results of evaluation models
        analyze_preds(exp_name, dsets=dsets, splits=splits,
                      hparams=hparams, ckpt_option=ckpt_option)


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Load shared training configurations
    CONF = config_utils.load_config(__file__, ARGS.config, copy=True)
    LOGGER.debug("""
################################################################################
#                       Starting `ssl_model_eval` Script                       #
################################################################################""")

    # 2.1 Process configuration parameters
    # Flatten nesting in configuration file
    HPARAMS = config_utils.flatten_nested_dict(CONF)

    # 3. Run main
    main(ARGS, HPARAMS)
