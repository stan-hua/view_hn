"""
dataset.py

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
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode

# Custom libraries
from src.data import constants
from src.data_prep import utils


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

# Torchvision Grayscale/RGB constants
IMAGE_MODES = {1: ImageReadMode.GRAY, 3: ImageReadMode.RGB}


################################################################################
#                                Main Functions                                #
################################################################################
def load_dataset_from_dir(img_dir, full_seq=True):
    """
    Loads image dataset from directory of images.

    Parameters
    ----------
    img_dir : str
        Path to directory containing ultrasound images.
    full_seq : str
        If True, groups ultrasound images by unique patient-visits, by default
        False.

    Returns
    -------
    torch.utils.data.Dataset
        Contains images and metadata from filename
    """
    return UltrasoundDatasetDir(img_dir, img_size=None, full_seq=full_seq)


def load_dataset_from_dataframe(df, img_dir=None, full_seq=True):
    """
    Loads image dataset from dataframe of image paths and labels.

    Parameters
    ----------
    df : pd.DataFrame
        Contains column with absolute/relative path to images, and labels
    img_dir : str, optional
        Path to directory containing ultrasound images, by default None.
    full_seq : str
        If True, groups ultrasound images by unique patient-visits, by default
        False.

    Returns
    -------
    torch.utils.data.Dataset
        Contains images, metadata from filename, and labels from dataframe
    """
    return UltrasoundDatasetDataFrame(df, img_dir=img_dir, full_seq=full_seq)


################################################################################
#                             Data Module Classes                              #
################################################################################
class UltrasoundDataModule(pl.LightningDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities.
    """
    def __init__(self, dataloader_params=None, df=None, img_dir=None,
                 full_seq=False, mode=3,
                 augment_training=False,
                 **kwargs):
        """
        Initialize UltrasoundDataModule object.

        Note
        ----
        Either df or img_dir must be exclusively specified to load in data.

        By default, does not split data.

        Parameters
        ----------
        dataloader_params : dict, optional
            Used to override default parameters for DataLoaders, by default None
        df : pd.DataFrame, optional
            Contains paths to image files and labels for each image, by default
            None
        img_dir : str, optional
            Path to directory containing ultrasound images, by default None
        full_seq : bool, optional
            If True, each item has all ordered images for one full
            ultrasound sequence (determined by patient ID and visit). If False,
            treats each image under a patient as independent, by default False.
        mode : int, optional
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 3.
        augment_training : bool, optional
            If True, add random augmentations during training, by default False.
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
        self.img_dir = img_dir
        self.dataset = None
        self.full_seq = full_seq
        self.mode = mode
        self.augment_training = augment_training
        self.img_size = kwargs.get("img_size", 258)
        self.no_label = (self.df is None)   # flag to use UltrasoundDatasetDir
        self.label_part = kwargs.get("label_part")
        self.split_label = kwargs.get("multi_output", False)

        # Get image paths, patient IDs, and labels (and visit)
        if self.df is not None:
            if img_dir:
                df["filename"] = df["filename"].map(
                    lambda x: os.path.join(img_dir, x))

            # Filter for existing images
            exists_mask = df.filename.map(os.path.exists)
            if not all(exists_mask):
                num_missing = len(~exists_mask[~exists_mask])
                LOGGER.warning(f"{num_missing} image files in table don't exist "
                               "at path! Skipping...")
                df = df[exists_mask]

            # Get specific metadata for splitting
            self.img_paths = df["filename"].to_numpy()
            self.labels = df["label"].to_numpy()
            self.patient_ids = df["id"].to_numpy()
        else:
            self.img_paths = np.array(glob.glob(os.path.join(img_dir, "*")))
            self.labels = np.array([None] * len(self.img_paths))
            # NOTE: Following may not work for newer hospital data
            self.patient_ids = utils.get_from_paths(self.img_paths)

        ########################################################################
        #                        DataLoader Parameters                         #
        ########################################################################
        # Default parameters for data loader
        default_data_params = {"batch_size": 32,
                               "shuffle": False,
                               "num_workers": 4,
                               "pin_memory": True}

        # Parameters for training/validation DataLoaders
        self.train_dataloader_params = default_data_params
        if dataloader_params:
            self.train_dataloader_params.update(dataloader_params)

        # NOTE: Shuffle is turned off during validation/test
        # NOTE: Batch size is set to 1 during validation/test
        self.val_dataloader_params = self.train_dataloader_params.copy()
        self.val_dataloader_params["shuffle"] = False

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

        ########################################################################
        #                            Augmentations                             #
        ########################################################################
        augmentations = []
        if self.augment_training:
            augmentations.extend([
                T.RandomAdjustSharpness(1.25, p=0.25),
                T.RandomApply([T.GaussianBlur(1, 0.1)], p=0.5),
                T.RandomRotation(15),
                T.RandomResizedCrop(self.img_size, scale=(0.5, 1)),
            ])
        self.transforms = T.Compose(augmentations)


    def setup(self, stage="fit"):
        """
        Prepares data for model training/validation/testing
        """
        # (1) Split into training and test sets
        if hasattr(self, "train_test_split") and self.train_test_split < 1:
            train_idx, test_idx = utils.split_by_ids(self.patient_ids, 
                                                     self.train_test_split)
            self._assign_dset_idx("train", "test", train_idx, test_idx)
        
        # (2) Further split training set into train-val or cross-val sets
        # (2.1) Train-Val Split
        if hasattr(self, "train_val_split"):
            train_idx, val_idx = utils.split_by_ids(self.dset_to_ids["train"], 
                                                    self.train_val_split)
            self._assign_dset_idx("train", "val", train_idx, val_idx)
        # (2.2) K-Fold Cross Validation
        elif hasattr(self, "cross_val_folds"):
            self._cross_val_train_dict = {
                "ids": self.dset_to_ids["train"],
                "paths": self.dset_to_paths["train"],
                "labels": self.dset_to_labels["train"]
            }

            self.cross_fold_indices = utils.cross_validation_by_patient(
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
        df_train = pd.DataFrame({
            "filename": self.dset_to_paths["train"],
            "label": self.dset_to_labels["train"]
        })

        # Get patient ID, visit number and sequence number, from orig. table
        df_train = utils.left_join_filtered_to_source(
            df_train, self.df,
            index_cols="filename")

        # Instantiate UltrasoundDatasetDataFrame
        train_dataset = UltrasoundDatasetDataFrame(
            df_train, self.img_dir,
            self.full_seq,
            img_size=self.img_size,
            mode=self.mode,
            label_part=self.label_part,
            split_label=self.split_label,
            transforms=self.transforms,
        )

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

        # Get patient ID, visit number and sequence number, from orig. table
        df_val = utils.left_join_filtered_to_source(
            df_val, self.df,
            index_cols="filename")

        val_dataset = UltrasoundDatasetDataFrame(
            df_val, self.img_dir,
            self.full_seq,
            img_size=self.img_size,
            mode=self.mode,
            label_part=self.label_part,
            split_label=self.split_label,
        )

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

        # Get patient ID, visit number and sequence number, from orig. table
        df_test = utils.left_join_filtered_to_source(
            df_test, self.df,
            index_cols="filename")

        test_dataset = UltrasoundDatasetDataFrame(
            df_test, self.img_dir,
            self.full_seq,
            img_size=self.img_size,
            mode=self.mode,
            label_part=self.label_part,
            split_label=self.split_label,
        )

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

        # Load image
        X = self.load_image(img_path)

        # Get metadata from filename
        filename = os.path.basename(img_path)
        patient_id = self.ids[index]
        visit = self.visits[index]
        seq_number = self.seq_numbers[index]
        hospital = self.hospitals[index]

        metadata = {
            "filename": filename,
            "id": patient_id,
            "visit": visit,
            "seq_number": seq_number,
            "hospital": hospital,
        }

        return X, metadata


    def load_image(self, img_path):
        """
        Loads an image given the image path

        Parameters
        ----------
        img_path : str
            Path to image

        Returns
        -------
        torch.Tensor
            Image
        """
        assert os.path.exists(img_path), "No image at path specified!"

        X = read_image(img_path, self.mode)
        X = self.transforms(X)

        # Normalize between [0, 1]
        X = X / 255.

        return X


    def __len__(self):
        return len(self.paths)


class UltrasoundDatasetDir(UltrasoundDataset):
    """
    Dataset to load images from a directory.
    """
    def __init__(self, img_dir, full_seq=False, img_size=None, mode=3,
                 transforms=None):
        """
        Initialize UltrasoundDatasetDir object.

        Parameters
        ----------
        img_dir : str
            Path to flat directory containing ultrasound images.
        full_seq : bool, optional
            If True, each item returned is a full ultrasound sequence with shape
            (sequence_length, num_channels, img_height, img_width). Otherwise,
            each item is an ultrasound image of shape of
            (num_channels, img_height, img_width).
        img_size : int or tuple of ints, optional
            If int provided, resizes found images to (img_size x img_size), by
            default None.
        mode : int
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 3.
        transforms : torchvision.transforms.Compose or Transforms, optional
            If provided, perform transform on images loaded, by default None.
        """
        assert mode in (1, 3)
        self.mode = IMAGE_MODES[mode]

        # Get all images in flat directory
        self.paths = np.array(glob.glob(os.path.join(img_dir, "*")))

        # Get all patient IDs
        self.ids = np.array(utils.get_from_paths(self.paths))

        # Get hospital visit number
        self.visits = utils.get_from_paths(self.paths, "visit")

        # Get number in US sequence
        self.seq_numbers = utils.get_from_paths(self.paths, "seq_number")

        # Add placeholder for hospital
        # NOTE: This is done for compatibility
        self.hospitals = np.empty(len(self.paths))

        ########################################################################
        #                  For Full US Sequence Data Loading                   #
        ########################################################################
        self.full_seq = full_seq
        self.id_visit = None

        # Get unique patient ID and visits, corresponding to unique US seqs
        if full_seq:
            self.id_visit = np.unique(tuple(zip(self.ids, self.visits)), axis=0)

        ########################################################################
        #                           Image Transforms                           #
        ########################################################################
        transforms = [transforms] if transforms is not None else []
        if img_size:
            transforms.insert(0, T.Resize(img_size))

        self.transforms = T.Compose(transforms)


    def __getitem__(self, index):
        """
        Loads an image with metadata, or a group of images from the same US
        sequence.

        Parameters
        ----------
        index : int
            Integer index to paths.

        Returns
        -------
        tuple
            Contains torch.Tensor and dict (containing metadata). Metadata may
            include path to image.
        """
        # If returning all images from full US sequences, override logic
        if self.full_seq:
            return self.get_sequence(index)

        # If returning an image
        X, metadata = super().__getitem__(index)

        return X, metadata


    def get_sequence(self, index):
        """
        Used to override __getitem__ when loading ultrasound sequences as each
        item.

        Parameters
        ----------
        index : int
            Integer index to a list of unique (patient id, visit number)

        Returns
        -------
        tuple
            Contains torch.Tensor and dict (containing metadata). Metadata may
            include paths to images.
        """
        # 1. Create boolean mask for the right US sequence
        patient_id, visit = self.id_visit[index]
        id_mask = (self.ids == patient_id)
        visit_mask = (self.visits == visit)
        mask = (id_mask & visit_mask)

        # 2. Filter for the image paths and metadata
        paths = self.paths[mask]
        seq_numbers = self.seq_numbers[mask]

        # 3. Order by sequence number
        sort_idx = np.argsort(seq_numbers)
        paths = paths[sort_idx]
        seq_numbers = seq_numbers[sort_idx]

        # 4. Load images
        imgs = []
        for path in paths:
            imgs.append(self.load_image(path)) 
        X = torch.stack(imgs)

        # 4.1 If only 1 image for a sequence, pad first dimension
        if len(imgs) == 1:
            X = torch.unsqueeze(X, 0)

        # 5. Metadata
        filenames = [os.path.basename(path) for path in paths]
        hospital = "Stanford" if filenames[0].startswith("SU2") else "SickKids"

        metadata = {"filename": filenames, "id": patient_id,
                    "visit": visit, "seq_number": seq_numbers,
                    "hospital": hospital}

        return X, metadata


    def __len__(self):
        """
        Return number of items in the dataset. If returning full sequences,
        groups images under the same specific patient ID and hospital visit.

        Returns
        -------
        int
            Number of items in dataset
        """
        if self.full_seq:
            return len(self.id_visit)

        return super().__len__()


class UltrasoundDatasetDataFrame(UltrasoundDataset):
    """
    Dataset to load images and labels from a DataFrame.
    """
    def __init__(self, df, img_dir=None, full_seq=False, img_size=None, mode=3,
                 label_part=None, split_label=False,
                 transforms=None):
        """
        Initialize UltrasoundDatasetDataFrame object.

        Note
        ----
        Expects path column to be "filename", and label column to be "label".

        Parameters
        ----------
        df : pd.DataFrame
            Contains path to images and labels.
        img_dir : str, optional
            If provided, uses paths in dataframe as relative paths find
            ultrasound images, by default None.
        full_seq : bool, optional
            If True, each item returned is a full ultrasound sequence with shape
            (sequence_length, num_channels, img_height, img_width). Otherwise,
            each item is an ultrasound image of shape of
            (num_channels, img_height, img_width).
        img_size : int or tuple of ints, optional
            If int provided, resizes found images to (img_size x img_size). If
            tuple provided, resizes images to (img_height, img_width), by
            default None.
        mode : int, optional
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 3.
        label_part : str, optional
            Label type. One of ("side", "plane", None). Used to get the correct
            classes and indices, by default None.
        split_label : bool, optional
            If True, additionally provide both side/plane separately in metadata
            dict, by default False.
        transforms : torchvision.transforms.Compose or Transforms, optional
            If provided, perform transform on images loaded, by default None.
        """
        assert mode in (1, 3)
        self.mode = IMAGE_MODES[mode]

        # Get paths to images. Add directory to path, if not already in.
        if img_dir:
            has_path = df.filename.map(lambda x: img_dir in x)
            df.loc[~has_path, "filename"] = df.loc[~has_path, "filename"].map(
                lambda x: os.path.join(img_dir, x))
        self.paths = df["filename"].to_numpy()

        # Get labels
        self.labels = df["label"].to_numpy()
        self.label_part = label_part
        self.split_label = split_label

        # Get all patient IDs
        self.ids = df["id"].to_numpy()

        # Get hospital visit number
        self.visits = df["visit"].to_numpy()

        # Get number in US sequence
        self.seq_numbers = df["seq_number"].to_numpy()

        # Get hospital
        self.hospitals = df["hospital"].to_numpy()

        ########################################################################
        #                  For Full US Sequence Data Loading                   #
        ########################################################################
        self.full_seq = full_seq
        self.id_visit = None

        # Get unique patient ID and visits, corresponding to unique US seqs
        if full_seq:
            self.id_visit = np.unique(tuple(zip(self.ids, self.visits)), axis=0)

        ########################################################################
        #                           Image Transforms                           #
        ########################################################################
        transforms = [transforms] if transforms is not None else []
        if img_size:
            transforms.insert(0, T.Resize(img_size))

        # TODO: Try standardizing image
        # transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5],
        #                               std=[0.5, 0.5, 0.5]))

        self.transforms = T.Compose(transforms)


    def __getitem__(self, index):
        """
        Loads an image with metadata, or a group of images from the same US
        sequence.

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
        # If returning all images from full US sequences, override logic
        if self.full_seq:
            return self.get_sequence(index)

        # If returning an image
        X, metadata = super().__getitem__(index)

        # Encode label to integer (-1, if not found)
        class_to_idx = \
            constants.LABEL_PART_TO_CLASSES[self.label_part]["class_to_idx"]
        metadata["label"] = class_to_idx.get(self.labels[index], -1)
        # If specified, split label into side/plane, and store separately
        if self.split_label and not self.label_part:
            for label_part in constants.LABEL_PARTS:
                metadata[label_part] = utils.extract_from_label(
                    self.labels[index], label_part)

        return X, metadata


    def get_sequence(self, index):
        """
        Used to override __getitem__ when loading ultrasound sequences as each
        item.

        Parameters
        ----------
        index : int
            Integer index to a list of unique (patient id, visit number)

        Returns
        -------
        tuple
            Contains torch.Tensor and dict (containing metadata). Metadata may
            include paths to images and labels.
        """
        if isinstance(index, list) and len(index) == 1:
            index = index[0]

        # 1. Create boolean mask for the right US sequence
        patient_id, visit = self.id_visit[index]
        id_mask = (self.ids == patient_id)
        visit_mask = (self.visits == visit)
        mask = (id_mask & visit_mask)

        # 2. Filter for the image paths and metadata
        paths = self.paths[mask]
        labels = self.labels[mask]
        seq_numbers = self.seq_numbers[mask]
        hospital = self.hospitals[mask]

        # 3. Order by sequence number
        sort_idx = np.argsort(seq_numbers)
        paths = paths[sort_idx]
        labels = labels[sort_idx]
        seq_numbers = seq_numbers[sort_idx]

        # 4. Load images
        imgs = []
        filenames = []
        for path in paths:
            imgs.append(self.load_image(path))

            filename = os.path.basename(path)
            filenames.append(filename)

        X = torch.stack(imgs)

        # 5. Encode labels to integers (-1, if not found)
        class_to_idx = \
            constants.LABEL_PART_TO_CLASSES[self.label_part]["class_to_idx"]
        encoded_labels = torch.LongTensor(
            [class_to_idx.get(label, -1) for label in labels])

        metadata = {
            "filename": filenames,
            "label": encoded_labels,
            "id": patient_id,
            "visit": visit,
            "seq_number": seq_numbers,
            "hospital": hospital,
        }

        # 6.1 If specified, split label into side/plane, and store separately
        if self.split_label and not self.label_part:
            for label_part in constants.LABEL_PARTS:
                # Extract side/plane from label
                extracted_labels = [utils.extract_from_label(label, label_part)
                    for label in labels]
                # Encode as an integer
                class_to_idx = \
                    constants.LABEL_PART_TO_CLASSES[label_part]["class_to_idx"]
                encoded_labels = torch.LongTensor(
                    [class_to_idx.get(label, -1) for label in extracted_labels])
                metadata[label_part] = encoded_labels

        return X, metadata


    def __len__(self):
        """
        Return number of items in the dataset. If returning full sequences,
        groups images under the same specific patient ID and hospital visit.

        Returns
        -------
        int
            Number of items in dataset
        """
        if self.full_seq:
            return len(self.id_visit)

        return super().__len__()


################################################################################
#                        Dataset Classes (Paired Data)                         #
################################################################################
class UltrasoundDatasetDataFramePair(UltrasoundDatasetDataFrame):
    """
    Dataset to load images and labels from a DataFrame in item pairs.
    """

    def __init__(self, df, random_pair=False, **kwargs):
        """
        Initialize UltrasoundDatasetDataFramePair object.

        Note
        ----
        Expects path column to be "filename", and label column to be "label".

        Parameters
        ----------
        df : pd.DataFrame
            Contains path to images and labels.
        random_pair : bool, optional
            If True, second item can be any item in the dataset randomly.
            Otherwise, second item is the next item in the index, by default
            False.
        kwargs : keyword arguments for UltrasoundDatasetDataFrame
            img_dir : str, optional
            full_seq : bool, optional
            img_size : int or tuple of ints, optional
            mode : int, optional
            label_part : str, optional
        """
        self.random_pair = random_pair
        super().__init__(df=df, **kwargs)


    def __getitem__(self, index):
        """
        Loads a pair of ultrasound images, or ultrasound image sequences.

        Note
        ----
        Second item in pair is either deterministic (next item) or random.

        Parameters
        ----------
        index : int
            Integer index to paths.

        Returns
        -------
        tuple of tuple of (torch.Tensor, dict)
            Pair of items to return
        """
        # Choose index of next pair
        if self.random_pair:
            second_index = index
            while second_index == index:
                second_index = np.random.randint(low=0, high=self.__len__())
        else:
            second_index = (index + 1) % self.__len__()
        
        # If returning all images from full US sequences, override logic
        first_item = super().__getitem__(index)
        second_item = super().__getitem__(second_index)

        return first_item, second_item
