"""
dataset.py

Description: Contains functions/classes to load dataset in PyTorch.
"""
# Standard libraries
import glob
import logging
import os
from abc import abstractmethod
from collections import defaultdict

# Non-standard libraries
import numpy as np
import pandas as pd
import lightning as L
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode

# Custom libraries
from src.data import constants
from src.data_prep import utils
from src.data_prep.sampler import ImbalancedDatasetSampler


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

# Torchvision Grayscale/RGB constants
IMAGE_MODES = {1: ImageReadMode.GRAY, 3: ImageReadMode.RGB}

# Default parameters for data loader
DEFAULT_DATALOADER_PARAMS = {
    "batch_size": 16,
    "shuffle": False,
    "num_workers": 4,
}

# Pre-Computed Mean & Std for SickKids Training Set
SK_TRAIN_MEAN = 123. / 255.
SK_TRAIN_STD = 74. / 255.


################################################################################
#                                Main Functions                                #
################################################################################
def load_dataset_from_dir(img_dir, full_seq=False):
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


def load_dataset_from_dataframe(df, img_dir=None, full_seq=False):
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
# TODO: Allow filtering training set for proportion of images with masks
class UltrasoundDataModule(L.LightningDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities.
    """
    def __init__(self, df=None, img_dir=None,
                 full_seq=False, mode=3,
                 full_path=False,
                 augment_training=False,
                 imbalanced_sampler=False,
                 default_dl_params=DEFAULT_DATALOADER_PARAMS,
                 **kwargs):
        """
        Initialize UltrasoundDataModule object.

        Note
        ----
        Either df or img_dir must be exclusively specified to load in data.

        By default, does not split data.

        Parameters
        ----------
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
        full_path : bool, optional
            If True, "filename" metadata contains full path to the image/s.
            Otherwise, contains path basename, by default False.
        augment_training : bool, optional
            If True, add random augmentations during training, by default False.
        imbalanced_sampler : bool, optional
            If True, perform imbalanced sampling during training to
            account for label imbalances, by default False
        default_dl_params : dict, optional
            Default dataloader parameters
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
                cross_val_folds : int
                    Number of folds to use for cross-validation
                force_train_ids : list
                    List of patient IDs to place into training set
                crop_scale : float
                    If augmenting training samples, lower bound on proportion of
                    area cropped relative to the full image.
                ensure_seg_mask : bool
                    If True, filter training set (post-split) for those with
                    segmentation masks (likely for use in GradCAM loss)
                exclude_filename_json : str
                    Path to JSON file containing image files to intentionally
                    exclude from training/val/test set (post-split), by default
                    None.
                include_unlabeled : bool
                    If True, include unlabeled as part of training set, by
                    default False.
                batch_size : int
                    Batch size
                shuffle : bool
                    If True, shuffle data during training
                num_workers : int
                    Number of CPU data gathering workers
        """
        super().__init__()

        # General arguments, used to instantiate UltrasoundDataset
        self.df = df
        self.img_size = kwargs.get("img_size", constants.IMG_SIZE)
        self.us_dataset_kwargs = {
            "img_dir": img_dir,
            "full_seq": full_seq,
            "mode": mode,
            "img_size": self.img_size,
            "label_part": kwargs.get("label_part"),
            "full_path": full_path,
            **kwargs,
        }

        # Process (or create) metadata table
        # CASE 1: Metadata table is provided with file paths in each row
        if self.df is not None:
            exists_mask = df["filename"].map(os.path.exists)

            # Raise error, if none exist
            if not exists_mask.any():
                raise RuntimeError(
                    f"None of `{len(df)}` paths exist! Please inspect `df`...")

            # Filter for existing images
            if not all(exists_mask):
                num_missing = len(~exists_mask[~exists_mask])
                missing_paths = "\n\t".join(df[~exists_mask]["filename"].tolist())
                LOGGER.warning(f"{num_missing} image files in table don't exist "
                               f"at path! Skipping...\n\t{missing_paths}")
                self.df = df[exists_mask]
        # CASE 2: Only image directory is provided
        else:
            # Get images in directory
            img_paths = np.array(glob.glob(os.path.join(img_dir, "*")))

            # Create dataframe
            # NOTE: Assume all training for now, can be split later
            self.df = pd.DataFrame({
                "dset": ["unknown"] * len(img_paths),
                "split": ["train"] * len(img_paths),
                "filename": img_paths,
                "label": [None] * len(img_paths),
                # NOTE: Following may not work for newer hospital data
                "id": utils.get_from_paths(img_paths, "id"),
            })
        
        # NOTE: Store static (original) version of dataframe
        self.df_orig = self.df.copy()

        ########################################################################
        #                        DataLoader Parameters                         #
        ########################################################################
        # Store parameter for imbalanced sampler
        self.imbalanced_sampler = imbalanced_sampler

        # Extract parameters for training/validation DataLoaders
        self.train_dataloader_params = {}
        for key, default_val in default_dl_params.items():
            self.train_dataloader_params[key] = kwargs.get(key, default_val)

        # NOTE: Shuffle is turned off during validation/test
        self.val_dataloader_params = self.train_dataloader_params.copy()
        self.val_dataloader_params["shuffle"] = False

        ########################################################################
        #                          Dataset Splitting                           #
        ########################################################################
        # Get list of patient IDs specifically to put in training set
        self.force_train_ids = kwargs.get("force_train_ids")

        # (1) To split dataset into training and test sets
        if "train_test_split" in kwargs:
            self.train_test_split = kwargs.get("train_test_split")

        # (2) To further split training set into train-val set
        if "train_val_split" in kwargs and kwargs["train_val_split"] != 1.0:
            self.train_val_split = kwargs.get("train_val_split")

        # (3) To further split training set into cross-validation sets
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
        # Standard augmentations used for all training
        self.augmentations = utils.prep_strong_augmentations(
            img_size=self.img_size,
            crop_scale=kwargs.get("crop_scale", 0.5))

        # HACK: If SSL dataset, it should use `self.augmentations` directly
        #       and set `augment_training` to False in super().__init()
        # NOTE: This is to avoid using augmentations twice:
        #       i) During data loading
        #       ii) In SSL collate function
        self.transforms = None
        if augment_training:
            self.transforms = self.augmentations


    def setup(self, stage="fit"):
        """
        Prepares data for model training/validation/testing
        """
        # (1) Split into training and test sets
        if hasattr(self, "train_test_split") and self.train_test_split < 1:
            # Split into train/test by each dataset
            # NOTE: Do not overwrite train/test split if they already exist
            self.df = utils.assign_split_table(
                self.df, other_split="test",
                train_split=self.train_test_split,
                force_train_ids=self.force_train_ids,
                overwrite=False,
            )

        # (2) Further split training set into train-val or cross-val sets
        # (2.1) Train-Val Split
        if hasattr(self, "train_val_split"):
            # Split data into training split and rest
            train_val_mask = self.df["split"].isin(["train", "val"])
            df_train_val = self.df[train_val_mask]
            df_rest = self.df[~train_val_mask]

            # Split training set into train/val by each dataset
            # NOTE: Do not overwrite train/val split if they already exist
            df_train_val = utils.assign_split_table(
                df_train_val, other_split="val",
                train_split=self.train_val_split,
                force_train_ids=self.force_train_ids,
                overwrite=False,
            )

            # Recombine
            self.df = pd.concat([df_train_val, df_rest], ignore_index=True)

        # (2.2) K-Fold Cross Validation
        elif hasattr(self, "cross_val_folds"):
            # TODO: Re-implement when needed
            raise NotImplementedError("Not currently implemented!")
            train_ids = self.df[self.df["split"] == "train"]["id"].tolist()
            self.cross_fold_indices = utils.cross_validation_by_patient(
                train_ids, self.cross_val_folds)
            # By default, set to first kfold
            self.set_kfold_index(0)

        # Assign data split for unlabeled data
        unlabeled_split = "train" if self.us_dataset_kwargs.get("include_unlabeled") else None
        self.df = utils.assign_unlabeled_split(self.df, unlabeled_split)

        # If specified, filter training data for those with segmentation masks
        if self.us_dataset_kwargs.get("ensure_seg_mask"):
            # Split data into training split and rest
            df_train = self.df[self.df["split"] == "train"]
            df_rest = self.df[self.df["split"] != "train"]

            # Remove images in the training set that DONT have a seg. mask
            has_seg_mask = df_train["filename"].map(utils.has_seg_mask)
            df_train.loc[~has_seg_mask, "split"] = None

            # Recombine
            self.df = pd.concat([df_train, df_rest], ignore_index=True)

        # If specified, remove explicitly listed images from the training set
        exclude_filename_json = self.us_dataset_kwargs.get("exclude_filename_json")
        if exclude_filename_json:
            self.df = utils.exclude_from_any_split(self.df, exclude_filename_json)


    def train_dataloader(self):
        """
        Returns DataLoader for training set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for training data
        """
        # Instantiate UltrasoundDatasetDataFrame
        train_dataset = UltrasoundDatasetDataFrame(
            self.df[self.df["split"] == "train"],
            transforms=self.transforms,
            **self.us_dataset_kwargs,
        )

        # If specified, instantiate imbalanced sampler
        if self.imbalanced_sampler:
            LOGGER.info("Using imbalanced sampler for training!")
            sampler = ImbalancedDatasetSampler(train_dataset)
            self.train_dataloader_params["sampler"] = sampler
            self.train_dataloader_params["shuffle"] = False

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
        val_dataset = UltrasoundDatasetDataFrame(
            self.df[self.df["split"] == "val"],
            **self.us_dataset_kwargs,
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
        test_dataset = UltrasoundDatasetDataFrame(
            self.df[self.df["split"] == "test"],
            **self.us_dataset_kwargs,
        )

        # Create DataLoader with parameters specified
        return DataLoader(test_dataset, **self.val_dataloader_params)


    def get_dataloader(self, split):
        """
        Get specific data loader

        Parameters
        ----------
        split : str
            Split must be one of train/val/test

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader
        """
        assert split in ("train", "val", "test")
        split_to_func = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader
        }
        return split_to_func[split]()


    def get_filtered_dataloader(self, split, **filters):
        """
        Get data loader for a specific split with option to filter for specific
        dataset.

        Parameters
        ----------
        split : str
            Name of data split to load. One of (train/val/test)
        filters : Any
            Keyword arguments, containing row filters.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader with filtered data
        """
        # Create copy of data to restore later
        df_orig = self.df.copy()

        # Split data into specific split and rest
        split_mask = self.df["split"] == split
        df_split = self.df[split_mask].copy()
        df_rest = self.df[~split_mask].copy()

        # If provided, filter out rows in the specific split
        df_split = self.filter_metadata(split=split, **filters)

        # Modify stored metadata table
        # NOTE: So that created dataloader would use applied filters
        self.df = pd.concat([df_split, df_rest], ignore_index=True)
        # Construct dataloader
        dataloader = self.get_dataloader(split)

        # Restore original table
        self.df = df_orig

        return dataloader


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
        raise NotImplementedError("Not currently implemented!")

        assert hasattr(self, "cross_val_folds")
        assert fold in list(range(self.cross_val_folds))

        self.fold = fold

        # Set training and validation data for fold
        train_idx, val_idx = self.cross_fold_indices[self.fold]
        # self.split_to_ids["train"] = self._cross_val_train_dict["ids"]
        # self.split_to_paths["train"] = self._cross_val_train_dict["paths"]
        # self.split_to_labels["train"] = self._cross_val_train_dict["labels"]
        # self._assign_dset_idx("train", "val", train_idx, val_idx)

        LOGGER.info(f"==Fold {fold + 1}/{self.cross_val_folds}==:")


    def filter_metadata(self, dset=None, split=None, **filters):
        """
        Get metadata filtered for dataset or split.

        Parameters
        ----------
        dset : str, optional
            Dataset to filter for
        split : str, optional
            Data split to filter for, by default None
        **filters : Any
            Column to value keyword arguments to filter
        """
        df = self.df.copy()

        # Filter on dataset and data split
        if dset is not None:
            df = df[df["dset"] == dset]
        if split is not None:
            df = df[df["split"] == split]

        # If provided, perform other filters
        if filters:
            for col, val in filters.items():
                # Raise errors, if column not present
                if col not in df:
                    raise RuntimeError(f"Column {col} not in table provided!")

                # CASE 1: Value is a list/tuple
                if isinstance(val, (list, tuple, set)):
                    mask = df[col].isin(val)
                    df = df[mask]
                # CASE 2: Value is a single item
                else:
                    mask = (df[col] == val)
                    df = df[mask]

        return df


    def get_patient_ids(self, split=None):
        """
        Get unique patient IDs for dataset or a specific data split

        Parameters
        ----------
        split : str, optional
            Data split (train/val/test)

        Returns
        -------
        list
            List of patient IDs
        """
        return sorted(self.filter_metadata(split=split)["id"].unique().tolist())


    def size(self, split=None):
        """
        Get size of dataset or specific data split

        Parameters
        ----------
        split : str, optional
            Data split (train/val/test)

        Returns
        -------
        list
            List of patient IDs
        """
        return len(self.filter_metadata(split=split))


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
        assert os.path.exists(img_path), "No image at path specified!"

        # Load image
        X = read_image(img_path, self.mode)

        # Transforms 1. Apply texture transforms
        if self.transforms.get("texture") is not None:
            X = self.transforms["texture"](X)

        # Transforms 2. Apply geometric transforms
        if self.transforms.get("geometric") is not None:
            # CASE 1: Segmentation mask was stored, and will be transformed with image
            if hasattr(self, "_seg_masks") and index in self._seg_masks:
                seg_mask = self._seg_masks[index]
                # Add channel dimension
                if len(seg_mask.shape) == 2:
                    seg_mask = seg_mask.unsqueeze(0)

                # Append to X as last channel
                X_and_mask = torch.cat([X, seg_mask.float()])

                # Apply geometric transform
                X_and_mask = self.transforms["geometric"](X_and_mask)

                # Separate and store
                self._seg_masks[index] = X_and_mask[-1].round().bool()
                X = X_and_mask[:-1]
            # CASE 2: Just the image
            else:
                X = self.transforms["geometric"](X)

        # Transforms 3. Apply post-processing transforms, if specified
        if self.transforms.get("post-processing") is not None:
            X = self.transforms["post-processing"](X)

        # If image values not between 0 and 1, divide by 255
        if (X > 1).any():
            X = X / 255.

        # Get metadata
        filename = img_path if self.full_path else os.path.basename(img_path)
        patient_id = self.ids[index]
        visit = self.visits[index]
        seq_number = self.seq_numbers[index]
        dset = self.dsets[index]

        # Prepare metadata
        metadata = {
            "filename": filename,
            "id": patient_id,
            "visit": visit,
            "seq_number": seq_number,
            "dset": dset,
            # Store dataset index as well
            "dataset_idx": index,
        }

        return X, metadata


    def __len__(self):
        return len(self.paths)


class UltrasoundDatasetDir(UltrasoundDataset):
    """
    Dataset to load images from a directory.
    """
    def __init__(self, img_dir, full_seq=False, img_size=None, mode=3,
                 transforms=None,
                 full_path=False,
                ):
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
        transforms : dict, optional
            Maps from type of transform to composed transform
        full_path : bool, optional
            If True, "filename" metadata contains full path to the image/s.
            Otherwise, contains path basename, by default False.
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

        # Add placeholder for dataset
        # NOTE: This is done for compatibility
        self.dsets = np.empty(len(self.paths))

        # Flag if `filename` should be the full path (or the basename)
        self.full_path = full_path

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
        transforms = transforms if transforms is not None else {}
        if img_size:
            transform_type = "geometric"
            transforms[transform_type] = [transforms[transform_type]] if transform_type in transforms else []
            transforms[transform_type].insert(0, T.Resize(img_size))
            transforms[transform_type] = T.Compose(transforms[transform_type])

        self.transforms = transforms


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
        filenames = [path if self.full_path else os.path.basename(path)
                     for path in paths]
        dset = None

        metadata = {"filename": filenames, "id": patient_id,
                    "visit": visit, "seq_number": seq_numbers,
                    "dset": dset}

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
                 label_part=None,
                 load_seg_mask=False, include_liver_seg=False,
                 standardize_images=False, transforms=None, full_path=False,
                 **ignore_kwargs,
                ):
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
        load_seg_mask : bool, optional
            If True, load (available) segmentation masks for each image into
            one mask and store in the metadata, by default False
        include_liver_seg : bool, optional
            If True, include liver segmentation, if available. Otherwise, only
            use kidney/bladder segmentations, by default False.
        standardize_images : bool, optional
            If True, standardize images by the training set pre-computed stats.
        transforms : dict, optional
            Maps from type of transform to composed transform
        full_path : bool, optional
            If True, "filename" metadata contains full path to the image/s.
            Otherwise, contains path basename, by default False.
        **ignore_kwargs : dict, optional
            Misc. keyword arguments are ignored.
        """
        assert mode in (1, 3)
        self.mode = IMAGE_MODES[mode]
        self.img_size = img_size

        # Hidden variable to retrieve masks during data loading
        self._seg_masks = {}
        # Set to load available masks
        self.load_seg_mask = load_seg_mask
        self.include_liver_seg = include_liver_seg

        # Get paths to images. Add directory to path, if not already in.
        if img_dir:
            has_path = df["filename"].map(lambda x: img_dir in x)
            df.loc[~has_path, "filename"] = df.loc[~has_path, "filename"].map(
                lambda x: os.path.join(img_dir, x))
        self.paths = df["filename"].to_numpy()

        # Get labels
        self.labels = df["label"].to_numpy()
        self.label_part = label_part

        # Get all patient IDs
        self.ids = df["id"].to_numpy()

        # Get hospital visit number
        self.visits = df["visit"].to_numpy()

        # Get number in US sequence
        self.seq_numbers = df["seq_number"].to_numpy()

        # Get dset
        self.dsets = df["dset"].to_numpy()

        # Flag if `filename` should be the full path (or the basename)
        self.full_path = full_path

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
        transforms = transforms if transforms is not None else {}
        # If image size specified, at Resize transform
        if img_size:
            transform_type = "geometric"
            transforms[transform_type] = [transforms[transform_type]] if transform_type in transforms else []
            transforms[transform_type].insert(0, T.Resize(img_size))
            transforms[transform_type] = T.Compose(transforms[transform_type])
        # If specified, standardize images by pre-computed channel means/stds
        if standardize_images:
            transform_type = "post-processing"
            transforms[transform_type] = [transforms[transform_type]] if transform_type in transforms else []
            transforms.append(T.ToDtype(torch.float32, scale=True))
            transforms.append(T.Normalize(mean=[SK_TRAIN_MEAN] * 3,
                                          std=[SK_TRAIN_STD] * 3))
            transforms[transform_type] = T.Compose(transforms[transform_type])
        self.transforms = transforms


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

        # HACK: Load segmentation mask before image to group in transform
        # NOTE: See __getitem__ in UltrasoundDataset
        seg_metadata = {}
        if self.load_seg_mask:
            seg_metadata.update(self.get_segmentation_mask(index))
            if seg_metadata.get("has_seg_mask"):
                self._seg_masks[index] = seg_metadata["seg_mask"]

        # If returning an image
        X, metadata = super().__getitem__(index)

        # If segmentation mask was transformed, update
        if self.load_seg_mask and seg_metadata.get("has_seg_mask"):
            seg_metadata["seg_mask"] = self._seg_masks[index]
            self._seg_masks.clear()

        # Encode label to integer (-1, if not found)
        class_to_idx = \
            constants.LABEL_PART_TO_CLASSES[self.label_part]["class_to_idx"]
        # NOTE: This assumes that label part was extracted prior to this
        metadata["label"] = int(class_to_idx.get(self.labels[index], -1))

        # Record if has segmentation mask or not
        metadata["has_seg_mask"] = False

        # Early return, if not loading segmentation masks
        if not self.load_seg_mask:
            return X, metadata

        # Load and store segmentation masks
        metadata.update(seg_metadata)
        return X, metadata


    def get_segmentation_mask(self, index):
        """
        Get segmentation masks for image at index.

        Parameters
        ----------
        index : int
            Integer index to paths.

        Returns
        -------
        dict
            Contains flag if segmentation mask exists and segmentation mask
        """
        metadata_overwrite = {}
        metadata_overwrite["seg_mask"] = None

        # CASE 1: Bladder image
        seg_fname_suffixes = []
        if str(self.labels[index]).lower() in ("bladder", "none"):
            seg_fname_suffixes.append("_bseg")
        # CASE 2: Kidney image
        else:
            seg_fname_suffixes.append("_kseg")

        # Add flag to include liver
        if self.include_liver_seg:
            seg_fname_suffixes.append("_lseg")

        # Load kidney/bladder/liver segmentations
        # NOTE: If 2+ exist, they're combined into 1 mask
        has_mask = False
        img_path = self.paths[index]
        for suffix in seg_fname_suffixes:
            # Create potential name of mask image (located in the same directory)
            fname = os.path.basename(img_path)
            split_fname = fname.split(".")
            mask_fname = ".".join(split_fname[:-1]) + suffix + "." + split_fname[-1]
            mask_path = os.path.join(os.path.dirname(img_path), mask_fname)

            # Skip, if mask doesn't exist
            if not os.path.exists(mask_path):
                continue

            # Load segmentation mask
            has_mask = True
            curr_mask = read_image(mask_path, IMAGE_MODES[3])
            # Extract mask by getting red pixels (236, 28, 36)
            curr_mask = (
                torch.where(torch.isin(curr_mask[0], torch.arange(200, 250)), 1, 0) *
                torch.where(torch.isin(curr_mask[1], torch.arange(15, 35)), 1, 0) *
                torch.where(torch.isin(curr_mask[2], torch.arange(26, 46)), 1, 0)
            )
            curr_mask = curr_mask.bool()

            # CASE 1: No previous segmentation mask
            if metadata_overwrite["seg_mask"] is None:
                metadata_overwrite["seg_mask"] = curr_mask
            # CASE 2: 2+ masks found for image
            # NOTE: Used if liver segmentation is added
            else:
                metadata_overwrite["seg_mask"] = metadata_overwrite["seg_mask"] | curr_mask

        # If no mask loaded, make placeholder mask
        if not has_mask:
            img_size = self.img_size or constants.IMG_SIZE
            metadata_overwrite["seg_mask"] = torch.full(img_size, True,
                                                        dtype=torch.bool)

        # Record if has mask or not
        metadata_overwrite["has_seg_mask"] = has_mask
        return metadata_overwrite


    # TODO: Implement getting masks
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
        dset = self.dsets[mask]

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
            filename = path if self.full_path else os.path.basename(path)
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
            "id": list(patient_id.astype(str)),
            "visit": list(visit.astype(str)),
            "seq_number": list(seq_numbers),
            "dset": list(dset),
        }

        # Early return, if not loading segmentation masks
        if not self.load_seg_mask:
            return X, metadata

        # Get segmentation mask for each image and concatenate them
        img_indices = np.nonzero(mask)
        accum_seg_metadata = defaultdict(list)
        for img_index in img_indices:
            curr_seg_metadata = self.get_segmentation_mask(img_index)
            for key, val in curr_seg_metadata.items():
                accum_seg_metadata[key].append(val)
        accum_seg_metadata = dict(accum_seg_metadata)

        # Get image size
        img_size = None
        for mask in accum_seg_metadata["seg_mask"]:
            if mask is not None:
                img_size = mask.shape
                break

        # TODO: Consider adding 1-only mask for missing masks
        raise NotImplementedError("Getting segmentation masks is not yet implemented for videos!")

        metadata.update(accum_seg_metadata)
        return X, metadata


    def get_labels(self, encoded=False):
        """
        Get all labels by index

        Parameters
        ----------
        encoded : bool, optional
            If True, then encode as integer label

        Returns
        -------
        list
            If encoded, return list of integer labels. Otherwise, return
            list of string labels
        """
        # CASE 1: If not encoded
        if not encoded:
            return self.labels

        # CASE 2: Encoded
        # Encode label to integer (-1, if not found)
        class_to_idx = \
            constants.LABEL_PART_TO_CLASSES[self.label_part]["class_to_idx"]
        # NOTE: This assumes that label part was extracted prior to this
        encoded_labels = [int(class_to_idx.get(label, -1)) for label in self.labels]

        return encoded_labels


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
