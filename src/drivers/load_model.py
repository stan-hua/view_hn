"""
load_model.py

Description: Contains utility functions for instantiating model classes/objects.

Note: `hparams` is a direct dependence on arguments in `model_training.py`.
"""

# Standard libraries
import logging
import os
from pathlib import Path

# Non-standard libraries
import torch
import yaml
from efficientnet_pytorch import EfficientNet
from tensorflow.keras.applications.efficientnet import EfficientNetB0

# Custom libraries
from src.data import constants
from src.models.cpc import CPC
from src.models.efficientnet_lstm_pl import EfficientNetLSTM
from src.models.efficientnet_pl import EfficientNetPL
from src.models.linear_classifier import LinearClassifier
from src.models.linear_lstm import LinearLSTM
from src.models.moco import MoCo
from src.models.tclr import TCLR


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Mapping of SSL model name to model class
SSL_NAME_TO_MODEL_CLS = {
    "moco": MoCo,
    "tclr": TCLR,

    # Evaluation models
    "linear": LinearClassifier,
    "linear_lstm": LinearLSTM,
}


################################################################################
#                               Helper Functions                               #
################################################################################
def get_model_cls(hparams):
    """
    Given experiment hyperparameters, get appropriate model class.

    Note
    ----
    Adds `backbone` to hparams, if needed.

    Parameters
    ----------
    hparams : dict
        Experiment parameters

    Returns
    -------
    class
        Model class
    """
    # For self-supervised (SSL) image-based model
    if hparams.get("self_supervised"):
        # If training SSL
        ssl_model = hparams.get("ssl_model", "moco")
        model_cls = SSL_NAME_TO_MODEL_CLS[ssl_model]
        # If not evaluating an SSL method
        if not (hparams["ssl_eval_linear"] or hparams["ssl_eval_linear_lstm"]):
            return model_cls

        # If loading another SSL eval model
        extra_model_kwargs = {}
        if hparams.get("from_ssl_eval"):
            # Instantiate conv. model, if required
            extra_model_kwargs["conv_backbone"] = EfficientNet.from_name(
                hparams.get("model_name", "efficientnet-b0"),
                image_size=hparams.get("img_size", (256, 256)),
                include_top=False)

        # If no SSL checkpoint path provided, assume MoCo
        if not hparams["ssl_ckpt_path"]:
            hparams["ssl_ckpt_path"] = constants.MOCO_CKPT_PATH

        # Load pretrained model
        try:
            pretrained_model = model_cls.load_from_checkpoint(
                hparams["ssl_ckpt_path"], **extra_model_kwargs)
        except Exception as error_msg:
            LOGGER.warning(error_msg)
            rename_torch_module(hparams["ssl_ckpt_path"])
            LOGGER.info("Renamed model module names!")
            pretrained_model = model_cls.load_from_checkpoint(
                checkpoint_path=hparams["ssl_ckpt_path"], **extra_model_kwargs)

        # Get convolutional backbone
        # NOTE: Pretrained backbone/s, needs to be inserted as an argument
        for conv_backbone_name in ["conv_backbone", "backbone"]:
            if hasattr(pretrained_model, conv_backbone_name):
                hparams["conv_backbone"] = \
                    getattr(pretrained_model, conv_backbone_name)
                break
        if "conv_backbone" not in hparams:
            raise RuntimeError("Could not find `conv_backbone` for model: "
                                f"{ssl_model}!")

        # Get temporal backbone (if TCLR)
        if hparams["ssl_eval_linear_lstm"] and \
                hasattr(pretrained_model, "temporal_backbone"):
            temporal_backbone = pretrained_model.temporal_backbone
            hparams["temporal_backbone"] = temporal_backbone
        if ssl_model == "tclr" and "temporal_backbone" not in hparams:
            raise RuntimeError("Could not find `temporal_backbone` for "
                                f"model: {ssl_model}!")

        # Specify eval. model to load
        model_cls = LinearClassifier if hparams["ssl_eval_linear"] \
            else LinearLSTM
    # For supervised full-sequence model
    elif not hparams.get("self_supervised") and hparams.get("full_seq"):
        model_cls = EfficientNetLSTM
    # For supervised image-based model
    else:
        model_cls = EfficientNetPL

    return model_cls


def get_hyperparameters(hparam_dir=None, filename="hparams.yaml"):
    """
    Load hyperparameters from model training directory. If not provided, return
    default hyperparameters.

    Parameters
    ----------
    hparam_dir : str
        Path to model training directory containing hyperparameters.
    filename : str
        Filename of YAML file with hyperparameters, by default "hparams.yaml"

    Returns
    -------
    dict
        Hyperparameters
    """
    if hparam_dir:
        file_path = None
        # Recursively find hyperparameter file
        for path in Path(hparam_dir).rglob(filename):
            file_path = str(path)

        # Load hyperparameter file
        with open(file_path, "r") as stream:
            try:
                hparams = yaml.full_load(stream)
                return hparams
            except yaml.YAMLError as exc:
                LOGGER.critical(exc)
                LOGGER.critical("Using default hyperparameters...")

    # If above does not succeed, use default hyperparameters
    hparams = {
        "img_size": constants.IMG_SIZE,
        "train": True,
        "test": True,
        "train_test_split": 0.75,
        "train_val_split": 0.75
    }

    return hparams


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
    # Look for checkpoint files
    ckpt_paths = [str(path) for path in Path(path_exp_dir).rglob("*.ckpt")]

    # Remove last checkpoint. NOTE: The other checkpoint is for the best epoch
    ckpt_paths = [path for path in ckpt_paths if "last.ckpt" not in path]

    if not ckpt_paths:
        raise RuntimeError("No best epoch model checkpoint (.ckpt) found!")

    if len(ckpt_paths) > 1:
        LOGGER.warning("More than 1 checkpoint file (.ckpt) found besides "
                       "last.ckpt!")

    return ckpt_paths[0]


def load_pretrained_from_exp_name(exp_name, **overwrite_hparams):
    """
    Load pretrained model from experiment name.

    Parameters
    ----------
    exp_name : str
        Name of experiment

    Returns
    -------
    torch.nn.Module
        Pretrained model
    """
    # 0. Get experiment directory, where model was trained
    model_dir = os.path.join(constants.DIR_RESULTS, exp_name)
    if not os.path.exists(model_dir):
        raise RuntimeError("`exp_name` provided does not lead to a valid model "
                           "training directory")

    # 1 Get experiment hyperparameters
    hparams = get_hyperparameters(model_dir)
    hparams.update(overwrite_hparams)

    # 2. Load existing model and send to device
    # 2.1 Get checkpoint path
    ckpt_path = find_best_ckpt_path(model_dir)
    # 2.2 Get model class and extra parameters for loading from checkpoint
    hparams_copy = hparams.copy()
    model_cls = get_model_cls(hparams)
    extra_ckpt_params = {k:v for k,v in hparams.items() \
        if k not in hparams_copy}
    # 2.3 Load model
    try:
        model = model_cls.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            **extra_ckpt_params)
    except:
        rename_torch_module(ckpt_path)
        LOGGER.info("Renamed model module names!")
        model = model_cls.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            **extra_ckpt_params)

    return model


################################################################################
#                                  Deprecated                                  #
################################################################################
def load_pretrained_from_model_name(model_name):
    """
    Loads pretrained model by name.

    Parameters
    ----------
    model_name : str
        Model name
    
    Returns
    -------
    tf.Model or torch.nn.Module
        Pretrained model
    """
    # Attempt to get model weights
    weights = constants.MODEL_NAME_TO_WEIGHTS.get(model_name)

    if model_name == "cytoimagenet":
        feature_extractor = EfficientNetB0(weights=weights,
                                           include_top=False,
                                           input_shape=(None, None, 3),
                                           pooling="avg")
    elif model_name == "imagenet":
        feature_extractor = EfficientNetPL.from_pretrained(
            model_name="efficientnet-b0")
    elif model_name == "cpc":
        feature_extractor = CPC.load_from_checkpoint(weights)
    elif model_name == "moco":
        feature_extractor = MoCo.load_from_checkpoint(weights)
    elif model_name == "random":
        # Randomly initialized EfficientNet model
        feature_extractor = EfficientNetPL()
    else:
        raise RuntimeError("Invalid model_name specified!")

    return feature_extractor


def rename_torch_module(ckpt_path):
    """
    Rename module in a saved model checkpoint

    Parameters
    ----------
    ckpt_path : str
        Path to PyTorch Lightning checkpoint file
    """
    ckpt_dict = torch.load(ckpt_path)
    state_dict = ckpt_dict["state_dict"]
    name_mapping = {
        "conv_conv_backbone.": "conv_backbone.",
        "backbone.": "conv_backbone.",
        "backbone_momentum.": "conv_backbone_momentum.",
        "lstm_backbone.": "temporal_backbone.",
        "_lstm.": "temporal_backbone.",
        "_fc.": "fc.",
    }
    for module_name in list(state_dict.keys()):
        for pre_name in sorted(name_mapping.keys(),
                               key=lambda x: len(x),
                               reverse=True):
            post_name = name_mapping[pre_name]
            if module_name.startswith(pre_name):
                new_module_name = module_name.replace(pre_name, post_name)
                state_dict[new_module_name] = state_dict.pop(module_name)
    torch.save(ckpt_dict, ckpt_path)
