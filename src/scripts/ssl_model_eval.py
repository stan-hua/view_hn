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
from src.data import constants
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
FREEZE_WEIGHTS = [True, False]    # True, False

# Flag to perform Linear Probing - Fine-tuning
LP_FT = False

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
        "exp_names": "Experiment names of SSL-pretrained models to evaluate",
        "dsets": "List of dataset split or test dataset name to evaluate",
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
    parser.add_argument("--dsets", default=[constants.DEFAULT_EVAL_DSET],
                        nargs='+',
                        help=arg_help["dsets"])


def train_eval_model(hparams):
    """
    Train single evaluation model.

    Parameters
    ----------
    hparams : dict
        Contains updated hyperparameters to train SSL-eval model. Arguments
        will be passed into `model_training.main`
    """
    # Skip if exists
    exp_eval_name = hparams["exp_name"]
    if os.path.exists(os.path.join(constants.DIR_RESULTS, exp_eval_name)):
        LOGGER.info(f"`{exp_eval_name}` already exists! Skipping...")
        return

    # Attempt training
    try:
        model_training.main(hparams)
    except Exception as error_msg:
        # On exception, delete folder
        exp_dir = load_model.get_exp_dir(exp_eval_name, on_error="ignore")
        if exp_dir:
            shutil.rmtree(exp_dir)

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
    model_dir = os.path.join(constants.DIR_RESULTS, exp_name)
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
            if not LP_FT:
                continue

            # Get hparams for the linear-probed (LP) model
            # NOTE: Fine-tuning over linear probed weights
            lp_hparams = prep_eval_exp_hparams(
                curr_hparams,
                label_part=label_part,
                ssl_model=ssl_model,
                ssl_ckpt_path=ckpt_path,
                model_type=model_type,
                freeze_weights=True,
                lp_ft=False,
            )

            # Prepare arguments for loading linear-probing (LP) model
            lp_ft_hparams = lp_hparams.copy()
            lp_ft_hparams["from_ssl_eval"] = True
            lp_ft_hparams["ssl_ckpt_path"] = load_model.find_best_ckpt_path(
                exp_name=lp_hparams["exp_name"],
            )

            # Get hparams for the LP-FT model
            # NOTE: Using this function to create the eval experiment name
            lp_ft_hparams = prep_eval_exp_hparams(lp_ft_hparams)

            # Attempt to train fine-tuned model from linear-probed (LP) model
            train_eval_model(lp_ft_hparams)


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

    # Ensure number of classes is as expected
    label_part = hparams["label_part"]
    hparams["num_classes"] = len(constants.LABEL_PART_TO_CLASSES[label_part]["classes"])

    # Create new eval exp. name
    exp_eval_name = TEMPLATE_EVAL_EXP_NAME.render(
        exp_name=hparams["exp_name"],
        model_type=hparams["model_type"],
        label_part=hparams["label_part"],
        freeze_weights=hparams["freeze_weights"],
        lp_ft=hparams["lp_ft"],
        augment_training=hparams["augment_training"],
    )

    # Update experiment name
    hparams["exp_name"] = exp_eval_name

    return hparams
    

################################################################################
#                                  Main Flow                                   #
################################################################################
def main(exp_names, dsets, hparams):
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
    # For each base SSL experiment name provided,
    for exp_name in exp_names:
        # Check that exp_name leads to a valid directory
        if not os.path.exists(os.path.join(
                constants.DIR_RESULTS, exp_name)):
            raise RuntimeError("`exp_name` (%s) provided does not lead to a "
                               "SSL-trained model directory!", exp_name)

        # Train all evaluation models for pretrained SSL model
        train_eval_models(exp_name, **hparams)

        # Analyze results of evaluation models
        analyze_preds(exp_name, dset=dsets,
                      augment_training=hparams["augment_training"])


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
    main(ARGS.exp_names, ARGS.dsets, HPARAMS)
