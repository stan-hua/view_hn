"""
dataloaders.py

Description: Contains functions/classes to load dataset in PyTorch.
"""
# Standard libraries
import glob
import logging
import os
from abc import abstractmethod

# Non-standard libraries
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode

# Custom libraries
from src.data import constants


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Torchvision Grayscale/RGB constants
IMAGE_MODES = {1: ImageReadMode.GRAY, 3: ImageReadMode.RGB}


################################################################################
#                                Main Functions                                #
################################################################################
def load_dataset_from_dir(dir):
    """
    Loads image dataset from directory of images.

    Parameters
    ----------
    dir : str
        Path to directory containing ultrasound images.
    
    Returns
    -------
    torch.utils.data.Dataset
        Contains images and metadata from filename
    """
    return UltrasoundDatasetDir(dir, img_size=None)


def load_dataset_from_dataframe(df, dir=None):
    """
    Loads image dataset from dataframe of image paths and labels.

    Parameters
    ----------
    df : pd.DataFrame
        Contains column with absolute/relative path to images, and labels
    dir : str, optional
        Path to directory containing ultrasound images, by default None.
    
    Returns
    -------
    torch.utils.data.Dataset
        Contains images, metadata from filename, and labels from dataframe
    """
    return UltrasoundDatasetDataFrame(df, dir=dir)


################################################################################
#                               Helper Function                                #
################################################################################
def get_patient_ids(paths):
    """
    Get patient IDs from a list of paths.

    Returns
    -------
    np.array
        List of patient ID strings
    """
    patient_ids = []
    for path in paths:
        patient_ids.append(os.path.basename(path).split("_")[0])
    return np.array(patient_ids)


def split_by_ids(patient_ids, train_split=0.8, seed=constants.SEED):
    """
    Splits list of patient IDs into training and val/test set.

    Note
    ----
    Split may not be exactly equal to desired train_split due to imbalance of
    patients.

    Parameters
    ----------
    patient_ids : np.array or array-like
        List of patient IDs (IDs can repeat).
    train_split : float, optional
        Proportion of total data to leave for training, by default 0.8
    seed : int, optional
        If provided, sets random seed to value, by default constants.SEED

    Returns
    -------
    tuple of (np.array, np.array)
        Contains (train_indices, val_indices), which are arrays of indices into
        patient_ids to specify which are used for training or validation/test.
    """
    # Get expected # of items in training set
    n = len(patient_ids)
    n_train = int(n * train_split)

    # Add soft lower/upper bounds (5%) to expected number. 
    # NOTE: Assume it won't become negative
    n_train_min = int(n_train - (n * 0.05))
    n_train_max = int(n_train + (n * 0.05))

    # Create mapping of patient ID to number of occurrences
    id_to_len = {}
    for _id in patient_ids:
        if _id not in id_to_len:
            id_to_len[_id] = 0
        id_to_len[_id] += 1

    # Shuffle unique patient IDs
    unique_ids = list(id_to_len.keys())
    shuffled_unique_ids = shuffle(unique_ids, random_state=seed)

    # Randomly choose patients to add to training set until full
    train_ids = set()
    n_train_curr = 0
    for _id in shuffled_unique_ids:
        # Add patient if number of training samples doesn't exceed upper bound
        if n_train_curr + id_to_len[_id] <= n_train_max:
            train_ids.add(_id)
            n_train_curr += id_to_len[_id]

        # Stop when there is roughly enough in the training set 
        if n_train_curr >= n_train_min:
            break

    # Create indices
    train_idx = []
    val_idx = []
    for idx, _id in enumerate(patient_ids):
        if _id in train_ids:
            train_idx.append(idx)
        else:
            val_idx.append(idx)

    # Convert to arrays
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    return train_idx, val_idx


def cross_validation_by_patient(patient_ids, num_folds=5):
    """
    Create train/val indices for Cross-Validation with exclusive patient ids
    betwen training and validation sets.

    Parameters
    ----------
    patient_ids : np.array or array-like
        List of patient IDs (IDs can repeat).
    num_folds : int
        Number of folds for cross-validation

    Returns
    -------
    list of <num_folds> tuples of (np.array, np.array)
        Each tuple in the list corresponds to a fold's (train_ids, val_ids)
    """
    folds = []

    training_folds = []
    remaining_ids = patient_ids

    # Get patient IDs for training set in each folds
    while num_folds > 1:
        proportion = 1 / num_folds
        train_idx, rest_idx = split_by_ids(remaining_ids, proportion)

        training_folds.append(np.unique(remaining_ids[train_idx]))
        remaining_ids = remaining_ids[rest_idx]
        
        num_folds -= 1

    # The last remaining IDs are the patient IDs of the last fold
    training_folds.append(np.unique(remaining_ids))

    # Create folds
    fold_idx = list(range(len(training_folds)))
    for i in fold_idx:
        # Get training set indices
        uniq_train_ids = set(training_folds[i])
        train_idx = np.where([_id in uniq_train_ids for _id in patient_ids])[0]

        # Get validation set indices
        val_indices = fold_idx.copy()
        val_indices.remove(i)
        val_patient_ids = np.concatenate(
            np.array(training_folds, dtype=object)[val_indices])
        uniq_val_ids = set(val_patient_ids)
        val_idx = np.where([_id in uniq_val_ids for _id in patient_ids])[0]

        folds.append((train_idx, val_idx))

    return folds


################################################################################
#                             Data Module Classes                              #
################################################################################
class UltrasoundDataModule(pl.LightningDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities.
    """
    def __init__(self, dataloader_params=None, df=None, dir=None, **kwargs):
        """
        Initialize UltrasoundDataModule object.

        Note
        ----
        Either df or dir must be exclusively specified to load in data.

        By default, does not split data.

        Parameters
        ----------
        dataloader_params : dict, optional
            Used to overrite default parameters for DataLoaders, by default None
        df : pd.DataFrame, optional
            Contains paths to image files and labels for each image, by default
            None
        dir : str, optional
            Path to directory containing ultrasound images, by default None
        **kwargs : dict
            Optional keyword arguments:
                img_size : int or tuple of ints, optional
                    If int provided, resizes found images to
                    (img_size x img_size), by default None.
                train_test_split : float
                    Percentage of data to leave for training. The rest will be
                    used for testing
                train_val_split : float
                    Percentage of training set (test set removed) to leave for
                    validation
                cross_val_folds : int, 
                    Number of folds to use for cross-validation
        """
        super().__init__()
        assert dataloader_params is None or isinstance(dataloader_params, dict)

        # Used to instantiate UltrasoundDataset
        self.df = df
        self.dir = dir
        self.dataset = None

        # Get image paths, patient IDs, and labels
        if self.df is not None:
            if dir:
                df["filename"] = df["filename"].map(
                    lambda x: os.path.join(dir, x))
            self.img_paths = df["filename"].to_numpy()
            self.labels = df["label"].to_numpy()
            self.patient_ids = get_patient_ids(self.img_paths)
        else:
            self.img_paths = np.array(glob.glob(os.path.join(dir, "*")))
            self.labels = np.array([None] * len(self.img_paths))
            self.patient_ids = get_patient_ids(self.img_paths)

        ########################################################################
        #                        DataLoader Parameters                         #
        ########################################################################
        # Default parameters for data loader
        default_data_params = {'batch_size': 32,
                               'shuffle': False,
                               'num_workers': 4,
                               'pin_memory': True}

        # Parameters for training/validation DataLoaders
        self.train_dataloader_params = default_data_params
        if dataloader_params:
            self.train_dataloader_params.update(dataloader_params)

        self.val_dataloader_params = self.train_dataloader_params.copy()
        self.val_dataloader_params['batch_size'] = 1
        self.val_dataloader_params['shuffle'] = False

        ########################################################################
        #                          Dataset Splitting                           #
        ########################################################################
        # Mapping of dataset split to patient IDs and paths
        self.dset_to_ids = {"train": self.patient_ids, "val": None,
                            "test": None}
        self.dset_to_paths = {"train": self.img_paths, "val": None, 
                              "test": None}
        self.dset_to_labels = {"train": self.labels, "val": None, "test": None}

        # (1) To split dataset into training and test sets
        if "train_test_split" in kwargs:
            self.train_test_split = kwargs.get("train_test_split")

        # (2) To further split training set into train-val or cross-val sets
        if "train_val_split" in kwargs and kwargs["train_val_split"] != 1.0:
            self.train_val_split = kwargs.get("train_val_split")

        if "cross_val_folds" in kwargs and kwargs["cross_val_folds"] > 1:
            self.cross_val_folds = kwargs.get("cross_val_folds")
            self.fold = 0

            # Store total training metadata split during cross validation
            self._cross_val_train_dict = None

            # Stores list of (train_idx, val_idx) for each fold
            self.cross_fold_indices = None


    def setup(self, stage='fit'):
        """
        Prepares data for model training/validation/testing
        """
        # (1) Split into training and test sets
        if hasattr(self, "train_test_split") and self.train_test_split < 1:
            train_idx, test_idx = split_by_ids(self.patient_ids, 
                                               self.train_test_split)
            self._assign_dset_idx("train", "test", train_idx, test_idx)
        
        # (2) Further split training set into train-val or cross-val sets
        # (2.1) Train-Val Split
        if hasattr(self, "train_val_split"):
            train_idx, val_idx = split_by_ids(self.dset_to_ids["train"], 
                                              self.train_val_split)
            self._assign_dset_idx("train", "val", train_idx, val_idx)
        # (2.2) K-Fold Cross Validation
        elif hasattr(self, "cross_val_folds"):
            self._cross_val_train_dict = {
                "ids": self.dset_to_ids["train"],
                "paths": self.dset_to_paths["train"],
                "labels": self.dset_to_labels["train"]
            }

            self.cross_fold_indices = cross_validation_by_patient(
                self.dset_to_ids["train"], self.cross_val_folds)
            # By default, set to first kfold
            self.set_kfold_index(0)


    def train_dataloader(self):
        """
        Returns DataLoader for training set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for training data
        """
        # Instantiate UltrasoundDatasetDataFrame
        df_train = pd.DataFrame({
            "filename": self.dset_to_paths["train"],
            "label": self.dset_to_labels["train"]
        })

        train_dataset = UltrasoundDatasetDataFrame(df_train, self.dir)

        # Create DataLoader with parameters specified
        return DataLoader(train_dataset, **self.train_dataloader_params)


    def val_dataloader(self):
        """
        Returns DataLoader for validation set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for validation data
        """
        # Instantiate UltrasoundDatasetDataFrame
        df_val = pd.DataFrame({
            "filename": self.dset_to_paths["val"],
            "label": self.dset_to_labels["val"]
        })
        val_dataset = UltrasoundDatasetDataFrame(df_val, self.dir)

        # Create DataLoader with parameters specified
        return DataLoader(val_dataset, **self.val_dataloader_params)


    def test_dataloader(self):
        """
        Returns DataLoader for test set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for test data
        """
        # Instantiate UltrasoundDatasetDataFrame
        df_test = pd.DataFrame({
            "filename": self.dset_to_paths["test"],
            "label": self.dset_to_labels["test"]
        })
        test_dataset = UltrasoundDatasetDataFrame(df_test, self.dir)

        # Create DataLoader with parameters specified
        return DataLoader(test_dataset, **self.val_dataloader_params)


    ############################################################################
    #                           Helper Functions                               #
    ############################################################################
    def set_kfold_index(self, fold):
        """
        If performing cross-validation, sets fold index.

        Note
        ----
        Fold index can range between [0, num_folds - 1]

        Parameters
        ----------
        fold : int
            Fold index
        """
        assert hasattr(self, "cross_val_folds")
        assert fold in list(range(self.cross_val_folds))

        self.fold = fold

        # Set training and validation data for fold
        train_idx, val_idx = self.cross_fold_indices[self.fold]
        self.dset_to_ids["train"] = self._cross_val_train_dict["ids"]
        self.dset_to_paths["train"] = self._cross_val_train_dict["paths"]
        self.dset_to_labels["train"] = self._cross_val_train_dict["labels"]
        self._assign_dset_idx("train", "val", train_idx, val_idx)

        LOGGER.info(f"==Fold {fold + 1}/{self.cross_val_folds}==:")


    def _assign_dset_idx(self, from_dset, to_dset, keep_idx, transfer_idx):
        """
        Reassigns patient IDs, image paths and labels from one dset to another,
        given indices to assign from source dset.

        Note
        ----
        <to_dset>'s items are reassigned, so previous samples assigned to
        <to_dset> (if any) are lost.

        Parameters
        ----------
        from_dset : str
            One of ("train", "val", "test")
        to_dset : str
            One of ("train", "val", "test")
        keep_idx : np.array
            Array of indices of samples to keep in source dataset
        transfer_idx : np.array
            Array of indices of samples to
        """
        # Local references
        dset_to_ids = self.dset_to_ids
        dset_to_paths = self.dset_to_paths
        dset_to_labels = self.dset_to_labels

        # Reassign target dset
        dset_to_ids[to_dset] = dset_to_ids[from_dset][transfer_idx]
        dset_to_paths[to_dset] = dset_to_paths[from_dset][transfer_idx]
        dset_to_labels[to_dset] = dset_to_labels[from_dset][transfer_idx]

        # Reassign source dset
        dset_to_ids[from_dset] = dset_to_ids[from_dset][keep_idx]
        dset_to_paths[from_dset] = dset_to_paths[from_dset][keep_idx]
        dset_to_labels[from_dset] = dset_to_labels[from_dset][keep_idx]


################################################################################
#                               Dataset Classes                                #
################################################################################
class UltrasoundDataset(torch.utils.data.Dataset):
    """
    Abstract Dataset class to load images.
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Do not instantiate this class directly!")


    def __getitem__(self, index):
        """
        Loads an image with metadata.

        Parameters
        ----------
        index : int
            Integer index to paths.

        Returns
        -------
        tuple
            Contains torch.Tensor and dict (containing metadata). Metadata may
            include path to image, patient ID, and hospital.
        """
        img_path = self.paths[index]
        X = read_image(img_path, self.mode)
        X = self.transforms(X)

        # Normalize between [0, 1]
        X = X / 255.

        # Get metadata from filename
        filename = os.path.basename(img_path)
        filename_parts = filename.split("_")
        patient_id = filename_parts[0]
        us_num = int(filename_parts[-1].replace(".jpg", ""))
        # NOTE: ID naming is used to identify hospital
        hospital = "Stanford" if filename.startswith("SU2") else "SickKids"

        metadata = {"filename": filename, "id": patient_id, "us_num": us_num,
                    "hospital": hospital}

        return X, metadata


    def __len__(self):
        return len(self.paths)


class UltrasoundDatasetDir(UltrasoundDataset):
    """
    Dataset to load images from a directory.
    """
    def __init__(self, dir, img_size=None, mode=3):
        """
        Initialize KidneyDatasetDir object.

        Parameters
        ----------
        dir : str
            Path to flat directory containing ultrasound images.
        img_size : int or tuple of ints, optional
            If int provided, resizes found images to (img_size x img_size), by
            default None.
        mode : int
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 1.
        """
        assert mode in (1, 3)
        self.mode = IMAGE_MODES[mode]

        # Get all images in flat directory
        self.paths = np.array(glob.glob(os.path.join(dir, "*")))

        # Get all patient IDs
        self.ids = np.array(get_patient_ids(self.paths))

        # Define image loading and transforms
        transforms = []
        if img_size:
            transforms.append(T.Resize(img_size))

        self.transforms = T.Compose(transforms)


class UltrasoundDatasetDataFrame(UltrasoundDataset):
    """
    Dataset to load images and labels from a DataFrame.
    """
    def __init__(self, df, dir=None, img_size=None, mode=3):
        """
        Initialize KidneyDatasetDataFrame object.

        Note
        ----
        Expects path column to be "filename", and label column to be "label".

        Parameters
        ----------
        df : pd.DataFrame
            Contains path to images and labels.
        dir : str, optional
            If provided, uses paths in dataframe as relative paths find
            ultrasound images, by default None.
        img_size : int or tuple of ints, optional
            If int provided, resizes found images to (img_size x img_size), by
            default None.
        mode : int
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 1.
        """
        assert mode in (1, 3)
        self.mode = IMAGE_MODES[mode]

        # Get paths to images. Add directory to path, if not already in.
        if dir:
            has_path = df.filename.map(lambda x: dir in x)
            df[~has_path]["filename"] = df[~has_path]["filename"].map(
                lambda x: os.path.join(dir, x))
        self.paths = df["filename"].to_numpy()

        # Get labels
        self.labels = df["label"].to_numpy()

        # Get all patient IDs
        self.ids = np.array(get_patient_ids(self.paths))

        # Define image loading and transforms
        transforms = []
        if img_size:
            transforms.append(T.Resize(img_size))

        # TODO: Try standardizing image
        # transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5],
        #                               std=[0.5, 0.5, 0.5]))

        self.transforms = T.Compose(transforms)


    def __getitem__(self, index):
        """
        Loads an image with metadata.

        Parameters
        ----------
        index : int
            Integer index to paths.

        Returns
        -------
        tuple
            Contains torch.Tensor and dict (containing metadata). Metadata may
            include path to image and label.
        """
        X, metadata = super().__getitem__(index)
        metadata["label"] = constants.CLASS_TO_IDX[self.labels[index]]

        return X, metadata
