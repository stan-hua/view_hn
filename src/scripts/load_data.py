"""
load_data.py

Description: Contains utility functions for instantiating DataModule objects.

Note: `hparams` is a direct dependence on arguments in `model_training.py`.
"""

# Standard libraries
import logging

# Non-standard libraries
import pandas as pd
from torch.utils.data import DataLoader

# Custom libraries
from src.data import constants
from src.data_prep import utils
from src.data_prep.dataset import (
    DEFAULT_DATALOADER_PARAMS,
    UltrasoundDataModule, UltrasoundDatasetDataFrame,
)
from src.data_prep.moco_dataset import MoCoDataModule
from src.data_prep.tclr_dataset import TCLRDataModule


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)

# Mapping of SSL model name to corresponding data module
SSL_NAME_TO_DATA_MODULE = {
    "moco": MoCoDataModule,
    "tclr": TCLRDataModule,
}

# Default hyperparameters
DEFAULT_HPARAMS = {
    "hospital": "sickkids",
    "train_val_split": 0.75,
    "train_test_split": 0.75,
    "train": True,
    "test": True,

    "img_size": constants.IMG_SIZE,
    "label_part": None,
    "relative_side": False,
    "include_unlabeled": False,

    "self_supervised": False,

    "batch_size": 16,
    "full_seq": True,
    "shuffle": False,
    "num_workers": 8,
    "pin_memory": True,
}


################################################################################
#                                  Functions                                   #
################################################################################
def setup_data_module(hparams=None, img_dir=None, use_defaults=False,
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

    # 0. Overwrite defaults
    all_hparams.update(hparams)
    all_hparams.update(overwrite_hparams)

    # 0. If no image directory provided, resort to defaults based on hospital
    #    chosen.
    img_dir = img_dir if img_dir \
        else constants.HOSPITAL_TO_IMG_DIR.get(all_hparams["hospital"])

    # 1. Load metadata
    df_metadata = utils.load_metadata(
        hospital=all_hparams["hospital"],
        extract=True,
        img_dir=img_dir,
        label_part=all_hparams.get("label_part"),
        relative_side=all_hparams.get("relative_side", False),
        include_unlabeled=all_hparams.get("include_unlabeled", False),
        keep_orig_label=all_hparams.get("keep_orig_label", False),
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
    dataloader_params = {
        "batch_size": all_hparams["batch_size"]
                      if not all_hparams["full_seq"] else 1,
        "shuffle": all_hparams["shuffle"],
        "num_workers": all_hparams["num_workers"],
        "pin_memory": all_hparams["pin_memory"],
    }
    dm = data_module_cls(dataloader_params, df=df_metadata,
                         img_dir=img_dir, **all_hparams)
    dm.setup()

    return dm


def get_dset_data_module(dset, **kwargs):
    """
    Get image dataloader for dataset split/name specified.

    Parameters
    ----------
    dset : str
        Name of dataset split or evaluation set
    **kwargs : dict, optional
        Keyword arguments for `setup_data_module`
        
    Returns
    -------
    lightning.pytorch.DataModule
        Each batch returns images and a dict containing metadata
    """
    # Prepare arguments for data module
    overwrite_hparams = create_overwrite_hparams(dset)
    # Update with kwargs
    overwrite_hparams.update(kwargs)

    # Set up data module
    dm = setup_data_module(use_defaults=True,
                           **overwrite_hparams)

    return dm


def get_dset_dataloader(dset, **kwargs):
    """
    Get image dataloader for dataset split/name specified.

    Parameters
    ----------
    dset : str
        Name of dataset split or evaluation set
    **kwargs : dict, optional
        Keyword arguments for `setup_data_module`
        
    Returns
    -------
    torch.DataLoader
        Each batch returns images and a dict containing metadata
    """
    # Set up data module
    dm = get_dset_data_module(dset=dset, **kwargs)

    # Get dataloader
    if dset == "val":
        img_dataloader = dm.val_dataloader()
    elif dset == "test":
        img_dataloader = dm.test_dataloader()
    else:
        img_dataloader = dm.train_dataloader()

    return img_dataloader


def get_dset_dataloader_filtered(dset, filters=None, **overwrite_hparams):
    """
    Get DataLoader for dataset split or evaluation dataset specified with
    filters, specified.

    Parameters
    ----------
    dset : str
        Name of data split or evaluation dataset
    filters : dict, optional
        Mapping of column name to allowed value/s
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite hyperparameters
    """
    # Create DataModule
    dm = get_dset_data_module(
        dset=dset,
        **overwrite_hparams
    )

    # Extract metadata table
    df_metadata = get_dset_metadata(
        dm=dm,
        dset=dset,
        **overwrite_hparams
    )

    # If provided, perform filters
    if filters:
        for col, val in filters.items():
            # Raise errors, if column not present
            if col not in df_metadata:
                raise RuntimeError(f"Column {col} not in table provided!")
            
            # CASE 1: Value is a list/tuple
            if isinstance(val, (list, tuple, set)):
                mask = df_metadata[col].isin(val)
                df_metadata = df_metadata[mask]
            # CASE 2: Value is a single item
            else:
                mask = (df_metadata[col] == val)
                df_metadata = df_metadata[mask]

    # Create DataLoader
    dataloader = create_dataloader_from_metadata_table(
        df_metadata=df_metadata,
        **overwrite_hparams
    )

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


def get_dset_metadata(dm, hparams=None,
                      dset=constants.DEFAULT_EVAL_DSET,
                      **overwrite_hparams):
    """
    Get metadata table containing (filename, label) for each image in the
    specified set (train/val/test).

    Parameters
    ----------
    dm : L.LightningDataModule
        DataModule used in model training run, used to load exact dset split
    hparams : dict, optional
        Experiment hyperparameters. If not provided, resort to defaults.
    dset : str, optional
        Specific split of dataset, or name of test set. If not one of (train,
        val, test), assume "train"., by default constants.DEFAULT_EVAL_DSET.
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite hyperparameters

    Returns
    -------
    pandas.DataFrame
        Metadata of each image in the dset split
    """
    # If not provided, use default hyperparameters
    hparams = hparams.copy() if hparams else DEFAULT_HPARAMS

    # Overwrite with keyword arguments
    hparams.update(overwrite_hparams)

    # Coerce to train, if not valid
    # NOTE: This case is for non-standard dset names (i.e., external test sets)
    hospital = "sickkids"
    if dset not in ("train", "val", "test"):
        LOGGER.warning(f"`{dset}` is not a valid data split. Assuming train "
                       "set is desired...")
        hospital = dset
        dset = "train"

    # Get filename and label of dset split data from data module
    df_dset = pd.DataFrame({
        "filename": dm.dset_to_paths[dset],
        "label": dm.dset_to_labels[dset],
    })

    # Extract data via join
    df_dset = utils.extract_data_from_filename_and_join(
        df_dset,
        hospital=hospital,
        label_part=hparams.get("label_part"),
        keep_orig_label=hparams.get("keep_orig_label", False),
    )

    return df_dset


def create_overwrite_hparams(dset):
    """
    If `dset` provided is for an external test set, return hyperparameters to
    overwrite experiment hyperparameters to load test data for evaluation.

    Parameters
    ----------
    dset : str
        Dataset split (train/val/test), or test dataset name (stanford)

    Returns
    -------
    dict
        Contains hyperparameters to overwrite, if necessary
    """
    overwrite_hparams = {
        "shuffle": False,
    }

    if dset not in ("sickkids", "train", "val", "test") \
            and dset in constants.HOSPITAL_TO_IMG_DIR:
        overwrite_hparams = {
            "hospital": dset,
            "train_val_split": 1.0,
            "train_test_split": 1.0,
            "test": False,
        }

    return overwrite_hparams
