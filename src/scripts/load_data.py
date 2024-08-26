"""
load_data.py

Description: Contains utility functions for instantiating DataModule objects.

Note: `hparams` is a direct dependence on arguments in `model_training.py`.
"""

# Standard libraries
import logging
import os

# Non-standard libraries
from torch.utils.data import DataLoader

# Custom libraries
from src.data import constants
from src.data_prep import utils
from src.data_prep.dataset import (
    DEFAULT_DATALOADER_PARAMS,
    UltrasoundDataModule, UltrasoundDatasetDataFrame,
)
from src.data_prep import BYOLDataModule, MoCoDataModule, TCLDataModule


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)

# Mapping of SSL model name to corresponding data module
SSL_NAME_TO_DATA_MODULE = {
    "byol": BYOLDataModule,
    "moco": MoCoDataModule,
    "tcl": TCLDataModule,
}

# Default hyperparameters
DEFAULT_HPARAMS = {
    "dsets": "sickkids",
    "train_val_split": 0.75,
    "train_test_split": 0.75,
    "train": True,
    "test": True,

    "img_size": constants.IMG_SIZE,
    "label_part": None,

    "self_supervised": False,

    "batch_size": 16,
    "full_seq": False,
    "shuffle": False,
    "num_workers": 8,
}


################################################################################
#                                  Functions                                   #
################################################################################
def setup_data_module(hparams=None, use_defaults=False,
                      full_path=False,
                      **overwrite_hparams):
    """
    Set up data module.

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters
    img_dir : str, optional
        Path to directory containing images, by default image directory
        corresponding to hospital (if any).
    use_defaults : bool, optional
        If True, start from default hyperparameters. Defaults to False.
    full_path : bool, optional
        If True, `filename` in metadata dicts is a full path. Defaults to False.
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite `hparams`

    Returns
    -------
    lightning.pytorch.LightningDataModule
    """
    all_hparams = {
        "full_path": full_path,
    }
    # 0. If specified, start from default hyperparameters
    if use_defaults:
        all_hparams.update(DEFAULT_HPARAMS)

    # INPUT: Ensure `hparams` is a dict
    hparams = hparams or {}

    # If excluding "Others" class, ensure `num_classes` is correct
    if not hparams.get("include_labeled_other"):
        label_part = hparams.get("label_part")
        classes = constants.LABEL_PART_TO_CLASSES[label_part]["classes"]
        hparams["num_classes"] = len(classes) - 1
        LOGGER.info("[DataModule Setup] Ensuring `num_classes` is correct")

    # 0. Overwrite defaults
    all_hparams.update(hparams)
    all_hparams.update(overwrite_hparams)

    # If sequence model, batch size must be 1
    if all_hparams.get("full_seq"):
        all_hparams["batch_size"] = 1

    # 1. Load metadata
    # 1.1 Prepare keyword arguments
    load_meta_config = {
        "label_part": all_hparams.get("label_part"),
    }
    # 1.2 Load metadata
    df_metadata = utils.load_metadata(
        dsets=all_hparams["dsets"],
        prepend_img_dir=True,
        **load_meta_config
    )

    # 2. Instantiate data module
    # 2.1 Choose appropriate class for data module
    if all_hparams.get("self_supervised") and not \
            (all_hparams["ssl_eval_linear"]
             or all_hparams["ssl_eval_linear_lstm"]):
        data_module_cls = SSL_NAME_TO_DATA_MODULE[all_hparams["ssl_model"]]
    else:
        data_module_cls = UltrasoundDataModule

    # 2.2 Pass in specified dataloader parameters
    dm = data_module_cls(df=df_metadata, **all_hparams)
    dm.setup()

    # Modify hyperparameters in-place to store training/val/test set IDs
    for split in ("train", "val", "test"):
        hparams[f"{split}_ids"] = dm.get_patient_ids(split)

    return dm


def setup_default_data_module_for_dset(dset=None, split="test", **kwargs):
    """
    Get image dataloader for dataset split/name specified.

    Parameters
    ----------
    dset : str
        Name of dataset
    split : str, optional
        Name of data split
    **kwargs : dict, optional
        Keyword arguments for `setup_data_module`
        
    Returns
    -------
    lightning.pytorch.DataModule
        Each batch returns images and a dict containing metadata
    """
    # Prepare arguments for data module
    dm_kwargs = create_eval_hparams(dset, split=split)
    # Update with kwargs
    dm_kwargs.update(kwargs)

    # Set up data module
    dm = setup_data_module(use_defaults=True, **dm_kwargs)

    return dm


def setup_default_dataloader_for_dset(dset, split=None, filters=None, **overwrite_hparams):
    """
    Create DataLoader for specific dataset and train/val/test split.

    Parameters
    ----------
    dset : str
        Name of dataset
    split : str
        Name of data split
    filters : dict, optional
        Mapping of column name to allowed value/s
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite hyperparameters
    """
    # Ensure filters is a dict
    filters = filters or {}

    # Create DataModule
    dm = setup_default_data_module_for_dset(
        dset=dset,
        **overwrite_hparams
    )

    # Get filtered dataloader
    dataloader = dm.get_filtered_dataloader(split=split, **filters)

    return dataloader


def create_dataloader_from_metadata_table(df_metadata,
                                          hparams=None,
                                          dataloader_params=None,
                                          **overwrite_hparams):
    """
    Given a metadata table, create a DataLoader.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Metadata table containing necessary data for image loading
    hparams : dict, optional
        Experiment hyperparameters. If not provided, resort to defaults.
    dataloader_params : dict, optional
        DataLoader parameters. If not provided, resort to defaults.
    **overwrite_hparams: dict, optional
        Keyword arguments to overwrite experiment hyperparameters
    """
    # If not provided, use default hyperparameters
    hparams = hparams.copy() if hparams else DEFAULT_HPARAMS
    dataloader_params = dataloader_params if dataloader_params \
        else DEFAULT_DATALOADER_PARAMS

    # Overwrite with keyword arguments
    hparams.update(overwrite_hparams)

    # Create Dataset object
    us_dataset = UltrasoundDatasetDataFrame(df_metadata, **hparams)

    # Create DataLoader with parameters specified
    return DataLoader(us_dataset, **dataloader_params)


def create_eval_hparams(dset=None, split="test"):
    """
    Create hyperparameters to evaluate on a data split (typically test)

    Parameters
    ----------
    dset : str
        If provided, filter by dataset name
    split : str, optional
        Data split, by default "test"

    Returns
    -------
    dict
        Contains hyperparameters to overwrite, if necessary
    """
    # Accumulate hyperparameters to overwrite
    overwrite_hparams = {
        "shuffle": False,
        "augment_training": False,
        "self_supervised": False,
    }

    # Check that provided dataset or split is valid
    if dset:
        assert dset in constants.DSET_TO_IMG_SUBDIR_FULL
    # Set dataset
    overwrite_hparams["dsets"] = [dset]
    assert split in ("train", "val", "test")

    return overwrite_hparams
