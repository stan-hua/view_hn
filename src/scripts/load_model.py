"""
load_model.py

Description: Contains utility functions for instantiating model classes/objects.

Note: `hparams` is a direct dependence on arguments in `model_training.py`.
"""

# Standard libraries
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

# Non-standard libraries
import torch
import yaml
from efficientnet_pytorch import EfficientNet
# from tensorflow.keras.applications.efficientnet import EfficientNetB0

# Custom libraries
from src.data import constants
from src.models import (
    EfficientNetLSTM, EfficientNetLSTMMulti, EfficientNetPL,
    LinearEval, LSTMLinearEval, EnsembleLinear, EnsembleLSTMLinear,
    BYOL, CPC, MoCo, TCLR,
)
from src.utils import efficientnet_pytorch_utils as effnet_utils


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
    "byol": BYOL,

    # Evaluation models
    "linear": LinearEval,
    "linear_lstm": LSTMLinearEval,
    "ensemble_linear": EnsembleLinear,
    "ensemble_linear_lstm": EnsembleLSTMLinear,
}

# Argument renaming
HPARAM_RENAMED = {
    "hospital": "dsets"
}


################################################################################
#                               Helper Functions                               #
################################################################################
def load_model(hparams):
    """
    Given experiment hyperparameters, instantiate/load model specified.

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters

    Returns
    ------
    torch.nn.Module
        Desired model
    """
    # Get model class
    model_cls, model_cls_kwargs = get_model_cls(hparams)
    # Instantiate model
    model = model_cls(**hparams, **model_cls_kwargs)

    # If specified, attempt to load ImageNet pretrained weights
    if hparams.get("from_imagenet") and hasattr(model, "load_imagenet_weights"):
        model.load_imagenet_weights()
    # If specified, start from a previously trained model
    elif hparams.get("from_exp_name") \
            and (isinstance(hparams.get("from_exp_name"), str) or
                 len(hparams.get("from_exp_name")) == 1):
        # Parse argument
        from_exp_name = hparams.get("from_exp_name")
        from_exp_name = from_exp_name[0] if isinstance(from_exp_name, list) \
            else from_exp_name

        # Load pretrained model
        pretrained_model = load_pretrained_from_exp_name(
            hparams.get("from_exp_name"),
            **model_cls_kwargs)
        # CASE 1: If pretrained model is the same, replace with existing model
        if type(model) == type(pretrained_model):
            overwrite_model(model, src_model=pretrained_model)
        # CASE 2: Update model weights with those from pretrained model
        # CASE 2.1: Model weight names don't need to be changed
        elif hparams.get("self_supervised"):
            pretrained_state_dict = pretrained_model.state_dict()
            # NOTE: SSL conv. backbone weights are prefixed by "conv_backbone."
            pattern = r"(conv_backbone\..*)|(temporal_backbone\..*)|(fc\..*)"
            pretrained_state_dict = prepend_prefix(
                pretrained_state_dict, "conv_backbone.",
                exclude_regex=pattern)
            model = overwrite_model(
                model,
                src_state_dict=pretrained_state_dict)
        # UNKNOWN CASE: Not supported case
        else:
            raise NotImplementedError

    # If specified, compile model
    if hparams.get("torch_compile"):
        LOGGER.debug("Compiling model...")
        model = torch.compile(model)
        LOGGER.debug("Compiling model...DONE")

    return model


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
    # 0. Redirect if `exp_name` is "imagenet"
    if exp_name == "imagenet":
        # Instantiate EfficientNet model
        model = EfficientNetPL(
            effnet_name=overwrite_hparams.get("effnet_name", "efficientnet-b0"),
            img_size=overwrite_hparams.get("img_size", constants.IMG_SIZE),
        )

        # Load ImageNet weights
        model.load_imagenet_weights()
        return model

    # 0. Get experiment directory, where model was trained
    model_dir = get_exp_dir(exp_name)

    # 1 Get experiment hyperparameters
    hparams = get_hyperparameters(model_dir)
    hparams.update(overwrite_hparams)

    # 2. Load existing model and send to device
    # 2.1 Get checkpoint path
    ckpt_path = find_best_ckpt_path(model_dir)
    # 2.2 Get model class and extra parameters for loading from checkpoint
    model_cls, model_cls_kwargs = get_model_cls(hparams)
    # 2.3 Load model
    try:
        model = model_cls.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            **model_cls_kwargs)
    except:
        rename_torch_module(ckpt_path)
        LOGGER.info("Renamed model module names!")
        model = model_cls.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            **model_cls_kwargs)

    # If specified, compile model
    if hparams.get("torch_compile"):
        LOGGER.debug("Compiling model...")
        model = torch.compile(model)
        LOGGER.debug("Compiling model...DONE")

    return model


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
    tuple of (class, dict) 
        Model class, and dict of keyword arguments needed to instantiate class
    """
    # Accumulate arguments, needed to instantiate class
    model_cls_kwargs = {}

    # For self-supervised (SSL) image-based model
    if hparams.get("self_supervised"):
        ssl_model = hparams.get("ssl_model", "moco")
        ssl_model_cls = SSL_NAME_TO_MODEL_CLS[ssl_model]

        # If training SSL (not evaluating an SSL-pretrained model)
        if not (hparams["ssl_eval_linear"] or hparams["ssl_eval_linear_lstm"]):
            return ssl_model_cls, model_cls_kwargs

        # Check if loading backbones from multiple SSL-finetuned models
        multi_backbone = isinstance(hparams["ssl_ckpt_path"], list) and \
            len(hparams["ssl_ckpt_path"]) > 1

        # Specify eval. model to load
        if hparams["ssl_eval_linear"]:
            model_cls = LinearEval
        elif multi_backbone:
            if hparams["full_seq"]:
                model_cls = EnsembleLSTMLinear
            else:
                model_cls = EnsembleLinear
        else:
            model_cls = LSTMLinearEval

        # Load backbones from ssl checkpoint path/s, provided in hyperparameters
        # NOTE: If loading from previous SSL eval. model, use that model class
        #       instead of SSL model
        backbone_dict = extract_backbones_from_ssl(
            hparams,
            model_cls if hparams.get("from_ssl_eval") else ssl_model_cls,
        )

        # Check temporal backbone (if TCLR)
        if ssl_model == "tclr" and \
                hparams["ssl_eval_linear_lstm"] and \
                "temporal_backbone" not in backbone_dict:
            raise RuntimeError("Could not find `temporal_backbone` for model!")

        # NOTE: Backbones need to be added to load the model
        model_cls_kwargs.update(backbone_dict)

    # For ensembling multiple models. NOTE: Needs to be sequence model
    elif hparams.get("from_exp_name") \
            and not isinstance(hparams.get("from_exp_name"), str) \
            and len(hparams.get("from_exp_name")) > 1:
        if hparams.get("full_seq"):
            model_cls = EnsembleLSTMLinear
        else:
            model_cls = EnsembleLinear

        # Load pretrained models
        exp_names = hparams.get("from_exp_name")
        pretrained_models = [load_pretrained_from_exp_name(exp_name)
                             for exp_name in exp_names]

        # Extract conv. backbones
        conv_backbones = []
        for pretrained_model in pretrained_models:
            conv_backbone = pretrained_model
            # CASE 1: Pretrained model is an EfficientNet model
            if isinstance(pretrained_model, EfficientNet):
                backbone_dict = extract_backbone_dict_from_efficientnet_model(
                    conv_backbone)
                conv_backbone = backbone_dict["conv_backbone"]
            # CASE 2: Pretrained model is NOT an EfficientNet model
            else:
                backbone_dict = extract_backbone_dict_from_ssl_model(
                    pretrained_model)
                conv_backbone = backbone_dict["conv_backbone"]
            conv_backbones.append(conv_backbone)

        # Send models to device
        device = hparams.get("device", constants.DEVICE)
        if device != "cpu" and torch.cuda.is_available():
            conv_backbones = [conv_backbone.to(device)
                              for conv_backbone in conv_backbones]

        # Update accumulator
        model_cls_kwargs["conv_backbones"] = conv_backbones

    # For supervised full-sequence model
    elif not hparams.get("self_supervised") and hparams.get("full_seq"):
        # If multi-output
        if hparams.get("multi_output"):
            model_cls = EfficientNetLSTMMulti
        else:
            model_cls = EfficientNetLSTM
    # For supervised image-based model
    else:
        # NOTE: Multi-output single-image model is not implemented
        if hparams.get("multi_output"):
            raise NotImplementedError("Supervised Multi-output model is not "
                                      "implemented for single images!")
        model_cls = EfficientNetPL
    return model_cls, model_cls_kwargs


def get_hyperparameters(hparam_dir=None, exp_name=None,
                        filename="hparams.yaml",
                        on_error="use_default"):
    """
    Load hyperparameters from model training directory. If not provided, return
    default hyperparameters.

    Parameters
    ----------
    hparam_dir : str
        Path to model training directory containing hyperparameters.
    exp_name : str, optional
        If `hparam_dir` not provided but `exp_name` is, use to find model
        directory, by default None.
    filename : str, optional
        Filename of YAML file with hyperparameters, by default "hparams.yaml"
    on_error : str, optional
        If "use_default", return default hyperparameters. If "raise", raises
        error, by default "use_default".

    Returns
    -------
    dict
        Hyperparameters
    """
    # 1. If hyperparameter directory not specified but experiment name is, check
    #    if model directory exists
    if not hparam_dir and exp_name:
        model_dir = get_exp_dir(exp_name, on_error="ignore")
        hparam_dir = model_dir or hparam_dir

    # 2. Load hyperparameters from directory
    if hparam_dir:
        file_path = None
        # Recursively find hyperparameter file
        for path in Path(hparam_dir).rglob(filename):
            file_path = str(path)

        # Raise error, if unable to find file
        if file_path is None:
            raise RuntimeError("No hyperparameters found in experiment "
                               f"directory!\n\tDirectory: {hparam_dir}")

        # Load hyperparameter file
        with open(file_path, "r") as stream:
            try:
                hparams = yaml.full_load(stream)
                # Rename required arguments if necessary
                for old_key, new_key in HPARAM_RENAMED.items():
                    if old_key in hparams:
                        hparams[new_key] = hparams.pop(old_key)
                return hparams
            except yaml.YAMLError as exc:
                LOGGER.critical(exc)
                LOGGER.critical("Using default hyperparameters...")

    # If above does not succeed,
    # CASE 0: Use default hyperparameters
    if on_error == "use_default":
        LOGGER.warning("Unable to find hyperparameters for specified "
                       "experiment! Resorting to default hyperparameters...")
        hparams = {
            "img_size": constants.IMG_SIZE,
            "dsets": ["sickkids"],
            "train": True,
            "test": True,
            "train_test_split": 0.75,
            "train_val_split": 0.75,
            "batch_size": 16,
            "shuffle": False,
        }
        return hparams
    # CASE 1: Raise error
    elif on_error == "raise":
        raise RuntimeError("Unable to find hyperparameters for specified "
                           f"experiment! ({exp_name})")


def find_best_ckpt_path(path_exp_dir=None, exp_name=None):
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
    # INPUT: Ensure at least one of `path_exp_dir` or `exp_name` is provided
    assert path_exp_dir or exp_name

    # If only `exp_name` provided, attempt to find experiment training directory
    if not path_exp_dir and exp_name:
        path_exp_dir = get_exp_dir(exp_name, on_error="raise")

    # Look for checkpoint files
    ckpt_paths = [str(path) for path in Path(path_exp_dir).rglob("*.ckpt")]

    # Remove last checkpoint. NOTE: The other checkpoint is for the best epoch
    ckpt_paths = [path for path in ckpt_paths if "last.ckpt" not in path]

    if not ckpt_paths:
        raise RuntimeError("No best epoch model checkpoint (.ckpt) found! "
                           f"\nDirectory: {path_exp_dir}")

    if len(ckpt_paths) > 1:
        raise RuntimeError("More than 1 checkpoint file (.ckpt) found besides "
                           f"last.ckpt! \nDirectory: {path_exp_dir}")

    return ckpt_paths[0]


def extract_backbones_from_ssl(hparams, model_cls):
    """
    Given experiment hyperparameters with 1+ specified SSL checkpoints, extract
    their conv. backbone and temporal backbone, if available.

    Parameters
    ----------
    hparams : dict
        Experiment parameters, containing 1+ SSL-pretrained model ckpt paths
    model_cls : class
        Reference to model class, of type torch.nn.Module

    Returns
    -------
    dict
        Contains mapping of name to backbones (conv_backbone or
        temporal_backbone)
    """
    hparams = hparams.copy()
    ssl_ckpt_paths = hparams["ssl_ckpt_path"]

    # If only one SSL ckpt path provided
    if isinstance(ssl_ckpt_paths, list) and len(ssl_ckpt_paths) == 1:
        hparams["ssl_ckpt_path"] = ssl_ckpt_paths[0]
    if isinstance(hparams["ssl_ckpt_path"], str):
        return extract_backbones_from_ssl_single(hparams, model_cls)

    # If multiple SSL ckpt path provided
    backbone_dict = defaultdict(list)
    for ssl_ckpt_path in ssl_ckpt_paths:
        # Create copy of hyperparameters
        hparams_copy = hparams.copy()
        hparams_copy["ssl_ckpt_path"] = ssl_ckpt_path

        # Extract backbones
        backbone_dict_i = extract_backbones_from_ssl_single(
            hparams_copy, model_cls)

        # Accumulate backbones
        for backbone_name in ["conv_backbone", "temporal_backbone"]:
            if backbone_name in backbone_dict_i:
                backbone_dict[f"{backbone_name}s"].append(
                    backbone_dict_i[backbone_name])

    return dict(backbone_dict)


def extract_backbones_from_ssl_single(hparams, model_cls):
    """
    Given experiment hyperparameters for 1 SSL-pretrained model, extract its
    conv. backbone and temporal backbone, if available.

    Parameters
    ----------
    hparams : dict
        Experiment parameters
    model_cls : class
        Reference to model class of type torch.nn.Module

    Returns
    -------
    dict
        Contains mapping of name to backbones (conv_backbone or
        temporal_backbone)
    """
    # If no SSL checkpoint path provided, assume MoCo
    ssl_ckpt_path = hparams.get("ssl_ckpt_path")
    if not ssl_ckpt_path:
        raise RuntimeError("No SSL checkpoint path provided!")

    # If loading another SSL eval model, instantiate required conv. backbones
    extra_model_kwargs = {}
    if hparams.get("from_ssl_eval"):
        extra_model_kwargs["conv_backbone"] = create_conv_backbone(hparams)

    # Load pretrained model
    try:
        pretrained_model = model_cls.load_from_checkpoint(
            ssl_ckpt_path, **extra_model_kwargs)
    except Exception as error_msg:
        LOGGER.warning(error_msg)
        rename_torch_module(ssl_ckpt_path)
        LOGGER.info("Renamed model module names!")
        pretrained_model = model_cls.load_from_checkpoint(
            checkpoint_path=ssl_ckpt_path, **extra_model_kwargs)

    return extract_backbone_dict_from_ssl_model(pretrained_model)


def extract_backbone_dict_from_ssl_model(model):
    """
    Given SSL-pretrained model instance, extract conv. (and temporal) backbones
    into a dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        Must contain "conv_backbone" (and "temporal_backbone") attributes.

    Returns
    -------
    dict
        Contains "conv_backbone" and optionally "temporal_backbone"
    """
    backbone_dict = {}

    # Get convolutional backbone
    # NOTE: Pretrained backbone/s, needs to be inserted as an argument
    for conv_backbone_name in ["conv_backbone", "backbone"]:
        if hasattr(model, conv_backbone_name):
            backbone_dict["conv_backbone"] = \
                getattr(model, conv_backbone_name)
            break
    if "conv_backbone" not in backbone_dict:
        raise RuntimeError("Could not find `conv_backbone` for model!")

    # Get temporal backbone
    if hasattr(model, "temporal_backbone"):
        temporal_backbone = model.temporal_backbone
        backbone_dict["temporal_backbone"] = temporal_backbone

    return backbone_dict


def extract_backbone_dict_from_efficientnet_model(model):
    """
    Given an EfficientNet model instance, extract conv. (and temporal) backbones
    into a dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        Must be an EfficientNet model

    Returns
    -------
    dict
        Contains "conv_backbone" and optionally "temporal_backbone"
    """
    assert isinstance(model, EfficientNet)

    backbone_dict = {}

    # CASE 1: Remove Linear layer (from forward pass), if exists
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()

    # CASE 2: Remove LSTM layer (from forward pass), if exists
    if hasattr(model, "temporal_backbone"):
        model.temporal_backbone = torch.nn.Identity()
    if hasattr(model, "temporal_backbone_forward"):
        identity_module = torch.nn.Identity()
        model.temporal_backbone_forward = lambda x: identity_module(x)

    # Verify that conv. layers exist
    if not len(find_layers_in_model(model, torch.nn.Conv2d)):
        raise RuntimeError("No conv. layers found!")

    # Get convolutional backbone
    backbone_dict["conv_backbone"] = model

    return backbone_dict


def create_conv_backbone(hparams=None):
    """
    Return base EfficientNet convolutional backbone, based on parameters.

    Parameters
    ----------
    hparams : dict
        Experiment parameters

    Returns
    -------
    torch.nn.Module
        EfficientNet convolutional backbone
    """
    # Default value for `hparams`
    hparams = hparams or {}

    # Create conv. backbone
    conv_backbone = EfficientNet.from_name(
        hparams.get("model_name", "efficientnet-b0"),
        image_size=hparams.get("img_size", (256, 256)),
        include_top=False)

    return conv_backbone


def overwrite_model(dst_model, src_model=None, src_state_dict=None):
    """
    Given a (new) model, overwrite its existing parameters based on a source
    model or its provided state dict.

    Note
    ----
    One of `src_model` or `src_state_dict` must be provided.

    Parameters
    ----------
    dst_model : torch.nn.Module
        Model whose weights to overwrite
    src_model : torch.nn.Module, optional
        Pretrained model whose weights to use in overwriting, by default None
    src_state_dict : dict, optional
        Pretrained model's state dict, by default None

    Returns
    -------
    torch.nn.Module
        Model whose weights were overwritten (in-place)
    """
    # INPUT: Ensure at least one of `src_model` or `src_state_dict` is provided
    assert src_model is not None or src_state_dict is not None, \
        "At least one of `src_model` or `src_state_dict` must be provided!"

    # Get model state dicts
    pretrained_weights = src_state_dict if src_state_dict is not None \
        else src_model.state_dict()
    new_weights = dst_model.state_dict()

    # Get names of overlapping weights
    pretrained_weight_names = set(pretrained_weights.keys())
    new_weight_names = set(new_weights.keys())
    overlapping_weights = pretrained_weight_names.intersection(new_weight_names)

    # Log skipped weights, due to incompatibility
    missing_weights = list(pretrained_weight_names.difference(new_weight_names))
    if missing_weights:
        LOGGER.warning("Loading pretrained model, where the following weights "
                       "were incompatible: \n\t%s",
                       "\n\t".join(missing_weights))

    # Overwrite overlapping weights with pretrained
    for weight_name in list(overlapping_weights):
        new_weights[weight_name] = pretrained_weights[weight_name]

    # Update the model's weights
    dst_model.load_state_dict(new_weights)

    LOGGER.info("Loaded weights from pretrained model successfully!")
    return dst_model


def prepend_prefix(state_dict, prefix, exclude_regex=None):
    """
    Given a state dict, prepend prefix to every weight name.

    Parameters
    ----------
    state_dict : dict
        Model state dict for a torch.nn.Module object
    prefix : str
        Prefix to prepend to each weight name
    exclude_regex : str, optional
        Regex for weights to exclude, by default None

    Returns
    -------
    dict
        Modified state dict
    """
    # Create copy to avoid in-place modification
    state_dict = state_dict.copy()

    # Compile regex if provided
    exclude_regex_pattern = re.compile(exclude_regex)

    # Prepend prefix, for valid weight names
    for weight_name in list(state_dict.keys()):
        if exclude_regex_pattern.match(weight_name) is not None:
            continue
        new_weight_name = prefix + weight_name
        state_dict[new_weight_name] = state_dict.pop(weight_name)

    return state_dict


def get_last_conv_layer(model):
    """
    Get last convolutional layer in model.

    Parameters
    ----------
    model : torch.nn.Module
        Convolutional model

    Returns
    -------
    torch.nn.Conv2d
        Last convolutional layer
    """
    # CASE 1: Model is a wrapper, storing a conv. backbone
    if isinstance(model, (LinearEval, LSTMLinearEval,
                          EnsembleLinear, EnsembleLSTMLinear)):
        return get_last_conv_layer(model.conv_backbone)
    # CASE 2: Model is an EfficientNetB0
    elif isinstance(model, EfficientNet):
        return model._conv_head

    # Raise error, if not found
    raise NotImplementedError


def get_exp_dir(exp_name, on_error="raise"):
    """
    Get experiment directory, given experiment name.

    Parameters
    ----------
    exp_name : str
        Experiment name
    on_error : str, optional
        If "raise", raises an error, if expected directory does not exist. If
        "ignore", simply returns None, by default "raise".

    Returns
    -------
    str
        Path to experiment directory, where model was trained
    """
    # INPUT: Verify provided `on_error` is valid
    assert on_error in ("raise", "ignore"), \
        "`on_error` must be one of ('raise', 'ignore')"

    # Create full path
    model_dir = os.path.join(constants.DIR_RESULTS, exp_name)

    # Raise error, if model directory does not exist
    if not os.path.exists(model_dir):
        if on_error == "raise":
            raise RuntimeError(f"`exp_name` ({exp_name}) provided does not lead"
                               " to a valid model training directory")
        model_dir = None
    return model_dir


def find_layers_in_model(model, layer_type):
    """
    Find specified layers in model.

    Parameters
    ----------
    model : torch.nn.Module
        Model
    layer_type : class
        Class of layer desired

    Returns
    -------
    lists
        List of model indices containing layer
    """
    fc_idx = []
    for idx, layer in enumerate(model.children()):
        if isinstance(layer, layer_type):
            fc_idx.append(idx)

    return fc_idx


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
        raise NotImplementedError()
        # model = EfficientNetB0(weights=weights,
        #                        include_top=False,
        #                        input_shape=(None, None, 3),
        #                        pooling="avg")
    elif model_name == "imagenet":
        model = EfficientNetPL()
        model.load_imagenet_weights()
    elif model_name == "cpc":
        model = CPC.load_from_checkpoint(weights)
    elif model_name == "random":
        model = EfficientNetLSTM()
    else:
        raise RuntimeError("Invalid model_name specified!")

    return model


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
