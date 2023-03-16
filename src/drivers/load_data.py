"""
load_data.py

Description: Contains utility functions for instantiating DataModule objects.

Note: `hparams` is a direct dependence on arguments in `model_training.py`.
"""

# Standard libraries
import logging

# Non-standard libraries
import pandas as pd

# Custom libraries
from src.data import constants
from src.data_prep import utils
from src.data_prep.dataset import UltrasoundDataModule
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
def setup_data_module(hparams, img_dir=None, use_defaults=False,
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
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite `hparams`

    Returns
    -------
    pytorch_lightning.LightningDataModule
    """
    all_hparams = {}
    # 0. If specified, start from default hyperparameters
    if use_defaults:
        all_hparams.update(DEFAULT_HPARAMS)

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
        include_unlabeled=all_hparams.get("include_unlabeled", False))

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
    overwrite_hparams = {}

    if dset not in ("sickkids", "train", "val", "test") \
            and dset in constants.HOSPITAL_TO_IMG_DIR:
        overwrite_hparams = {
            "hospital": dset,
            "train_val_split": 1.0,
            "train_test_split": 1.0,
            "test": False,
        }

    return overwrite_hparams


def get_dset_metadata(dm, hparams, dset=constants.DEFAULT_EVAL_DSET):
    """
    Get metadata table containing (filename, label) for each image in the
    specified set (train/val/test).

    Parameters
    ----------
    dm : pl.LightningDataModule
        Hyperparameters used in model training run, used to load exact dset
        split
    hparams : dict
        Experiment hyperparameters
    dset : str, optional
        Specific split of dataset, or name of test set. If not one of (train,
        val, test), assume "train"., by default constants.DEFAULT_EVAL_DSET.

    Returns
    -------
    pandas.DataFrame
        Metadata of each image in the dset split
    """
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
    )

    return df_dset
