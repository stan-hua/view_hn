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
import yaml
from tensorflow.keras.applications.efficientnet import EfficientNetB0

# Custom libraries
from src.data import constants
from src.models.cpc import CPC
from src.models.efficientnet_lstm_pl import EfficientNetLSTM
from src.models.efficientnet_pl import EfficientNetPL
from src.models.linear_classifier import LinearClassifier
from src.models.linear_lstm import LinearLSTM
from src.models.moco import MoCo
from src.models.siamnet import load_siamnet
from src.models.tclr import TCLR


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

# Mapping of SSL model name to model class
SSL_NAME_TO_MODEL_CLS = {
    "moco": MoCo,
    "tclr": TCLR,
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

        # If evaluating SSL method
        if hparams["ssl_eval_linear"] or hparams["ssl_eval_linear_lstm"]:
            # NOTE: Pretrained backbone/s, needs to be inserted as an arg.
            pretrained_model = model_cls.load_from_checkpoint(
                hparams["ssl_ckpt_path"])
            
            if ssl_model == "moco":
                hparams["conv_backbone"] = pretrained_model.backbone
            elif ssl_model == "tclr":
                hparams["conv_backbone"] = pretrained_model.conv_backbone

                # Get temporal backbone
                temporal_backbone = pretrained_model.temporal_backbone
                hparams["temporal_backbone"] = temporal_backbone

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
    elif model_name == "hn":
        feature_extractor = load_siamnet()
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