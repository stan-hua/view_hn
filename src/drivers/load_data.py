"""
load_data.py

Description: Contains utility functions for instantiating DataModule objects.

Note: `hparams` is a direct dependence on arguments in `model_training.py`.
"""

# Custom libraries
from src.data import constants
from src.data_prep import utils
from src.data_prep.dataset import UltrasoundDataModule
from src.data_prep.moco_dataset import MoCoDataModule
from src.data_prep.tclr_dataset import TCLRDataModule


################################################################################
#                                  Constants                                   #
################################################################################
# Mapping of SSL model name to corresponding data module
SSL_NAME_TO_DATA_MODULE = {
    "moco": MoCoDataModule,
    "tclr": TCLRDataModule,
}


################################################################################
#                                  Functions                                   #
################################################################################
def setup_data_module(hparams, img_dir=None, **overwrite_hparams):
    """
    Set up data module.

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters
    img_dir : str, optional
        Path to directory containing images, by default image directory
        corresponding to hospital (if any).
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite `hparams`

    Returns
    -------
    pytorch_lightning.LightningDataModule
    """
    # 0. If data splitting parameters are not given, assume defaults
    for split_params in ("train_val_split", "train_test_split"):
        if split_params not in hparams:
            hparams[split_params] = 0.75

    # 0. Create copy and overwrite hparams
    hparams = hparams.copy()
    hparams.update(overwrite_hparams)

    # 0. If no image directory provided, resort to defaults based on hospital
    #    chosen.
    img_dir = img_dir if img_dir \
        else constants.HOSPITAL_TO_IMG_DIR.get(hparams["hospital"])

    # 1. Load metadata
    df_metadata = utils.load_metadata(
        hospital=hparams["hospital"],
        extract=True,
        img_dir=img_dir,
        label_part=hparams["label_part"],
        relative_side=hparams["relative_side"],
        include_unlabeled=hparams["include_unlabeled"])

    # 2. Instantiate data module
    # 2.1 Choose appropriate class for data module
    if hparams["self_supervised"] and not \
            (hparams["ssl_eval_linear"] or hparams["ssl_eval_linear_lstm"]):
        data_module_cls = SSL_NAME_TO_DATA_MODULE[hparams["ssl_model"]]
    else:
        data_module_cls = UltrasoundDataModule
    # 2.2 Pass in specified dataloader parameters
    dataloader_params = {
        "batch_size": hparams["batch_size"] if not hparams["full_seq"] else 1,
        "shuffle": hparams["shuffle"],
        "num_workers": hparams["num_workers"],
        "pin_memory": hparams["pin_memory"],
    }
    dm = data_module_cls(dataloader_params, df=df_metadata,
                         img_dir=img_dir, **hparams)
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
