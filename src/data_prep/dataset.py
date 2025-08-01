"""
dataset.py

Description: Contains functions/classes to load dataset in PyTorch.
"""
# Standard libraries
import glob
import logging
import os
from collections import defaultdict

# Non-standard libraries
import numpy as np
import lightning as L
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from lightning.pytorch.utilities.combined_loader import CombinedLoader
# from hocuspocus.data.dataset import HocusPocusRealDataset, HocusPocusNoiseDataset, load_real_metadata, load_noise_metadata
# from hocuspocus.data.sampler import InfiniteBatchSampler
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode

# Custom libraries
from config import constants
from src.data_prep import utils
from src.data_prep.sampler import (
    ImbalancedDatasetSampler, InfiniteUnlabeledDatasetSampler,
)


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

# Default parameters for dataset
DEFAULT_DATASET_PARAMS = {
    "img_dir": None,
    "img_size": constants.IMG_SIZE,
    "full_seq": False,
    "mode": 3,
    "full_path": False,
    "label_part": None,
}

# Default parameters for data module
DEFAULT_DM_HPARAMS = {
    "augment_training": False,
    "imbalanced_sampler": False,
    "use_ood_unlabeled": False,
}
DEFAULT_DM_HPARAMS.update(DEFAULT_DATASET_PARAMS)

# Pre-Computed Mean & Std for SickKids Training Set
SK_TRAIN_MEAN = 123. / 255.
SK_TRAIN_STD = 74. / 255.


################################################################################
#                                Main Functions                                #
################################################################################
def load_dataset_from_dataframe(df, hparams=None, **kwargs):
    """
    Loads image dataset from dataframe of image paths and labels.

    Parameters
    ----------
    df : pd.DataFrame
        Contains column with absolute/relative path to images, and labels
    hparams : dict, optional
        Data hyperparameters
    **kwargs : Keyword Arguments
        If specified, simply attach them to hparams

    Returns
    -------
    torch.utils.data.Dataset
        Contains images, metadata from filename, and labels from dataframe
    """
    hparams = hparams or {}
    hparams.update(kwargs)
    return UltrasoundDatasetDataFrame(df, hparams=hparams)


################################################################################
#                                 Data Classes                                 #
################################################################################
# TODO: Allow filtering training set for proportion of images with masks
class UltrasoundDataModule(L.LightningDataModule):
    """
    UltrasoundDataModule class.

    Note
    ----
    Used to create training/validation/test dataloaders
    """

    def __init__(self, hparams, df=None, **overwrite_params):
        """
        Initialize UltrasoundDataModule object.

        Note
        ----
        Either df or img_dir must be exclusively specified to load in data.

        By default, does not split data.

        Parameters
        ----------
        hparams : dict
            Keyword arguments include:
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
                use_ood_unlabeled : bool, optional
                    If True, sample batches of labeled and "other" labeled data with the
                    same batch size.
                img_size : int or tuple of ints, optional
                    If int provided, resizes found images to
                    (img_size x img_size), by default None.
                train_test_split : float
                    Percentage of data to leave for training. The rest will be
                    used for testing
                train_val_split : float
                    Percentage of training set (test set removed) to leave for
                    validation
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
                batch_size : int
                    Batch size
                shuffle : bool
                    If True, shuffle data during training
                num_workers : int
                    Number of CPU data gathering workers
        df : pd.DataFrame, optional
            Contains paths to image files and labels for each image, by default
            None
        **overwrite_params : Keyword Argument
            If provided, simply overwrites values in hparams
        """
        super().__init__()

        # Add default hyperparameters
        hparams = hparams.copy() or {}
        hparams.update({k:v for k,v in DEFAULT_DM_HPARAMS.items() if k not in hparams})
        hparams.update(overwrite_params)
        self.my_hparams = hparams

        # General arguments, used to instantiate UltrasoundDataset
        self.df = df
        # Create table of filenames, if only image directory is provided
        if self.df is None:
            assert self.my_hparams.get("img_dir"), "Please provide either `df` or `img_dir`"
            # Get images in directory
            img_paths = np.array(glob.glob(os.path.join(self.my_hparams["img_dir"], "*")))

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
        #                            Augmentations                             #
        ########################################################################
        # Standard augmentations used for all training
        self.augmentations = utils.prep_strong_augmentations(
            img_size=self.my_hparams["img_size"],
            crop_scale=self.my_hparams.get("crop_scale", 0.5))

        # HACK: If SSL dataset, it should use `self.augmentations` directly
        #       and set `augment_training` to False in super().__init()
        # NOTE: This is to avoid using augmentations twice:
        #       i) During data loading
        #       ii) In SSL collate function
        self.transforms = None
        if self.my_hparams.get("augment_training"):
            self.transforms = self.augmentations


    def setup(self, stage="fit"):
        """
        Prepares data for model training/validation/testing
        """
        # If specified, include/drop images labeled "Others"
        # NOTE: Need to update hparams["num_classes"] elsewhere
        if not self.my_hparams.get("use_ood_unlabeled"):
            label_col = self.my_hparams.get("label_col", "label")
            na_mask = (self.df[label_col] == "Other") | (self.df[label_col].isna())
            LOGGER.info(f"[Pre-Split] Removing {na_mask.sum()} images labeled 'Other'")
            self.df.loc[na_mask, "split"] = None

        # (1) Split into training and test sets
        if self.my_hparams.get("train_test_split", 1) < 1:
            # Split into train/test by each dataset
            # NOTE: Do not overwrite train/test split if they already exist
            self.df = utils.assign_split_table(
                self.df, other_split="test",
                train_split=self.my_hparams.get("train_test_split"),
                force_train_ids=self.my_hparams.get("force_train_ids"),
                overwrite=False,
            )

        # (2) Further split training set into train-val or cross-val sets
        # (2.1) Train-Val Split
        if self.my_hparams.get("train_val_split", 1) < 1:
            # Split data into training split and rest
            train_val_mask = self.df["split"].isin(["train", "val"])
            df_train_val = self.df[train_val_mask]
            df_rest = self.df[~train_val_mask]

            # Split training set into train/val by each dataset
            # NOTE: Do not overwrite train/val split if they already exist
            df_train_val = utils.assign_split_table(
                df_train_val, other_split="val",
                train_split=self.my_hparams.get("train_val_split"),
                force_train_ids=self.my_hparams.get("force_train_ids"),
                overwrite=False,
            )

            # Recombine
            self.df = pd.concat([df_train_val, df_rest], ignore_index=True)


        # If specified, filter training data for those with segmentation masks
        if self.my_hparams.get("ensure_seg_mask"):
            # Split data into training split and rest
            df_train = self.df[self.df["split"] == "train"]
            df_rest = self.df[self.df["split"] != "train"]

            # Remove images in the training set that DONT have a seg. mask
            has_seg_mask = df_train["filename"].map(utils.has_seg_mask)
            df_train.loc[~has_seg_mask, "split"] = None
            LOGGER.info(
                f"[Post-Split] Keeping only {(~has_seg_mask).sum()} images "
                "with segmentation masks")

            # Recombine
            self.df = pd.concat([df_train, df_rest], ignore_index=True)

        # If specified, remove explicitly listed images from the training set
        exclude_filename_json = self.my_hparams.get("exclude_filename_json")
        if exclude_filename_json:
            self.df = utils.exclude_from_any_split(self.df, exclude_filename_json)

        # If specified, downsample training set proportion
        downsample_train_prop = self.my_hparams.get("downsample_train_prop", 1.)
        if downsample_train_prop and downsample_train_prop < 1:
            df_train = self.df[self.df["split"] == "train"]
            df_rest = self.df[self.df["split"] != "train"]

            # 0. Set seed
            rng_state = np.random.get_state()
            np.random.seed(42)

            # 1. Randomly assign samples to remove based on downsample train prop.
            # NOTE: Stratify based on label
            assert df_train.index.is_unique, "Metadata table index must be unique!"
            num_train_before = len(df_train)
            remove_train_ids = np.concatenate(df_train.groupby(by=["label"], group_keys=False).apply(
                lambda df_label: df_label.index[np.random.choice(
                    a=[False, True],
                    size=len(df_label),
                    p=(downsample_train_prop, 1-downsample_train_prop),
                )].tolist(),
                include_groups=False,
            ).values)
            # 2. Remove sampled rows from training set
            df_train.loc[remove_train_ids, "split"] = None

            # 3. Recombine
            self.df = pd.concat([df_train, df_rest], ignore_index=True)

            # Restore random state
            np.random.set_state(rng_state)

            # Log
            num_train_after = len(self.df[self.df["split"] == "train"])
            LOGGER.info("[Post-Split] Downsampled (labeld) training samples to "
                        f"{round(100*downsample_train_prop, 2)}% "
                        f"({num_train_after}/{num_train_before})")


    def train_dataloader(self):
        """
        Returns DataLoader for training set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for training data
        """
        return self.create_dataloaders("train")


    def val_dataloader(self):
        """
        Returns DataLoader for validation set.

        Returns
        -------
        L.CombinedLoader
            CombinedLoader of DataLoaders for evaluation
        """
        return self.create_dataloaders("val")


    def test_dataloader(self):
        """
        Returns DataLoader for test set.

        Returns
        -------
        L.CombinedLoader
            CombinedLoader of DataLoaders for evaluation
        """
        return self.create_dataloaders("test")


    def create_dataloaders(self, split):
        """
        Creates a CombinedLoader of DataLoaders for training/evaluation.

        Parameters
        ----------
        split : str
            Data split. Must be one of ["train", "val", "test"]

        Returns
        -------
        L.CombinedLoader
            CombinedLoader of DataLoaders for evaluation
        """
        # Store multiple dataloaders
        #   id := in-distribution ultrasound data
        #   ood_unlabeled := OOD unlabeled ultrasound data
        #   ood_hp_real := HocusPocus OOD real ultrasound data
        #   ood_hp_noise := HocusPocus OOD simulated noisy ultrasound data
        name_to_loader = {}
        name_to_loader.update(self.create_id_dataloaders(split))
        name_to_loader.update(self.create_ood_dataloaders(split))

        # Package all dataloaders into a CombinedLoader
        combined_loader = CombinedLoader(name_to_loader, "min_size")

        # Sample only as much as the first dataloader
        first_num_samples = len(name_to_loader["id"])
        combined_loader.limits = [first_num_samples] * len(name_to_loader)

        return combined_loader


    def create_id_dataloaders(self, split):
        """
        Create data loaders for in-distribution data

        Parameters
        ----------
        split : str
            One of {"train", "val", "test"}

        Returns
        -------
        dict
            Contains dataloader for in-distribution data under "id" key.
        """
        assert split in ["train", "val", "test"]

        # Prepare dataloader parameters
        base_dl_params = self.create_dl_params(split)

        # Get labeled data
        df_metadata = self.df[self.df["split"] == split]
        label_col = self.my_hparams.get("label_col", "label")
        na_mask = (df_metadata[label_col] == "Other") | (df_metadata[label_col].isna())
        df_labeled = df_metadata[~na_mask].reset_index(drop=True)

        # Create dataset for labeled data
        labeled_dataset = UltrasoundDatasetDataFrame(
            df_labeled,
            hparams=self.my_hparams,
            transforms=self.transforms,
        )

        # Store train dataloader
        name_to_loader = {}

        # CASE 1: Imbalanced sampler (NOTE: Only available during training)
        if self.my_hparams.get("imbalanced_sampler"):
            if split == "train":
                LOGGER.info(f"Using imbalanced sampler for {split}!")
                sampler = ImbalancedDatasetSampler(labeled_dataset)
                base_dl_params["sampler"] = sampler
                base_dl_params["shuffle"] = False
            else:
                LOGGER.info("Not using imbalanced sampler for validation/testing!")
        name_to_loader["id"] = DataLoader(labeled_dataset, **base_dl_params)

        return name_to_loader


    def create_ood_dataloaders(self, split):
        """
        Create data loaders for OOD detection.

        Parameters
        ----------
        split : str
            One of {"train", "val", "test"}

        Returns
        -------
        dict
            Contains dataloaders for OOD detection
        """
        assert split in ["train", "val", "test"]
        ood_split = f"ood_{split}"

        # Prepare dataloader parameters for OOD data
        base_dl_params = self.create_dl_params(split)
        ood_dl_params = base_dl_params.copy()
        ood_dl_params.pop("batch_size")
        ood_dl_params.pop("shuffle")

        name_to_loader = {}
        # CASE 1: OOD unlabeled sampler
        if self.my_hparams.get("use_ood_unlabeled"):
            LOGGER.info(f"Using unlabeled samples as OOD for {split}!")

            # Get unlabeled data
            df_metadata = self.df[self.df["split"] == split]
            label_col = self.my_hparams.get("label_col", "label")
            na_mask = (df_metadata[label_col] == "Other") | (df_metadata[label_col].isna())
            df_unlabeled = df_metadata[na_mask].reset_index(drop=True)
            df_unlabeled[label_col] = None

            # Create dataset
            assert not df_unlabeled.empty, f"No unlabeled samples for split `{split}`!"
            unlabeled_dataset = UltrasoundDatasetDataFrame(
                df_unlabeled,
                hparams=self.my_hparams,
                transforms=self.transforms if self.my_hparams.get("ood_augment") else None,
            )

            # Create sampler
            batch_sampler = InfiniteUnlabeledDatasetSampler(
                unlabeled_dataset,
                batch_size=base_dl_params["batch_size"],
                shuffle=base_dl_params["shuffle"],
            )
            curr_dl_params = ood_dl_params.copy()
            curr_dl_params["batch_sampler"] = batch_sampler
            name_to_loader["ood_unlabeled"] = DataLoader(unlabeled_dataset, **curr_dl_params)

        # Prepare transforms for OOD HocusPocus data
        ood_transform = None
        if self.my_hparams.get("ood_augment") and self.transforms is not None:
            ood_transform = utils.flatten_augmentations(self.transforms)

        # Shared kwargs for HocusPocus datasets
        hocus_pocus_shared_kwargs = {
            "split": ood_split,
            "img_mode": self.my_hparams.get("mode", 3),
            "img_size": self.my_hparams.get("img_size", constants.IMG_SIZE),
            "transform": ood_transform,
            "scale": True,
        }

        # CASE 2: HocusPocus OOD Real Dataset
        if self.my_hparams.get("use_ood_hocuspocus_real"):
            LOGGER.info(f"Using HocusPocus OOD Real dataset for {split}!")
            # Remove adult renal data
            real_ood_dataset = HocusPocusRealDataset(
                separate_background=True,
                exclude_views=["renal"],
                **hocus_pocus_shared_kwargs,
            )
            batch_sampler = InfiniteBatchSampler(
                real_ood_dataset,
                batch_size=base_dl_params["batch_size"],
            )
            curr_dl_params = ood_dl_params.copy()
            curr_dl_params["batch_sampler"] = batch_sampler
            name_to_loader["ood_hp_real"] = DataLoader(real_ood_dataset, **curr_dl_params)

        # CASE 3: HocusPocus OOD Noise datasets
        if self.my_hparams.get("use_ood_hocuspocus_noise"):
            LOGGER.info(f"Using HocusPocus OOD Noise dataset for {split}!")
            noise_ood_dataset = HocusPocusNoiseDataset(
                **hocus_pocus_shared_kwargs,
            )
            batch_sampler = InfiniteBatchSampler(
                noise_ood_dataset,
                batch_size=base_dl_params["batch_size"],
            )
            curr_dl_params = ood_dl_params.copy()
            curr_dl_params["batch_sampler"] = batch_sampler
            name_to_loader["ood_hp_noise"] = DataLoader(noise_ood_dataset, **curr_dl_params)

        return name_to_loader


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
        return self.create_id_dataloaders(split)["id"]


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
    def create_dl_params(self, split):
        """
        Create data loader parameters based on the specified data split.

        Parameters
        ----------
        split : str
            The data split for which to create the data loader parameters.
            Must be one of "train", "val", or "test".

        Returns
        -------
        dict
            A dictionary containing data loader parameters. For training,
            returns parameters with potential overrides from `my_hparams`.
            For validation/testing, returns parameters with shuffle set to False.
        """
        # Extract parameters for training/validation DataLoaders
        train_dl_params = {}
        for key, default_val in DEFAULT_DATALOADER_PARAMS.items():
            train_dl_params[key] = self.my_hparams.get(key, default_val)

        # CASE 1: Training
        if split == "train":
            return train_dl_params

        # CASE 2: Validation/Test
        # NOTE: Shuffle is turned off during validation/test
        assert split in ["val", "test"]
        val_dl_params = train_dl_params.copy()
        val_dl_params["shuffle"] = False
        return val_dl_params


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

        # SPECIAL CASE 1: HocusPocus dataset
        if dset and dset.startswith("ood_hp_"):
            ood_kwargs = {}
            if dset == "ood_hp_real":
                ood_kwargs["exclude_view"] = "renal"
            df = load_metadata_hocus_pocus(dset)
        # CASE 2: Specific dset in `df`
        elif dset is not None:
            df = df[df["dset"] == dset]

        # Filter on split
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
                    df = df[df[col] == val]

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


class UltrasoundDatasetDataFrame(torch.utils.data.Dataset):
    """
    UltrasoundDatasetDataFrame class.

    Note
    ----
    Used to load images from a metadata table
    """

    def __init__(self, df, hparams=None, transforms=None, **overwrite_params):
        """
        Initialize UltrasoundDatasetDataFrame object.

        Note
        ----
        Expects path column to be "filename", and label column to be "label".

        Parameters
        ----------
        df : pd.DataFrame
            Contains path to images and labels.
        hparams : dict, optional
            Contains the following keyword arguments:
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
        transforms : torchvision.transforms
            Image transformation
        **overwrite_params : Any
            Keyword arguments to overwrite hparams
        """
        self.df = df

        # Add default hyperparameters
        hparams = hparams or {}
        hparams.update({k:v for k,v in DEFAULT_DATASET_PARAMS.items() if k not in hparams})
        hparams.update(overwrite_params)
        self.my_hparams = hparams

        # Hidden variable to retrieve masks during data loading
        self._seg_masks = {}

        # Metadata columns
        self.id_col = self.my_hparams.get("id_col", "id")
        self.visit_col = self.my_hparams.get("visit_col", "visit")
        self.seq_number_col = self.my_hparams.get("seq_number_col", "seq_number")
        self.label_col = self.my_hparams.get("label_col", "label")
        self.path_cols = self.my_hparams.get("path_cols", ["filename"])
        self.dset_col = self.my_hparams.get("dset_col", "dset")

        # Get mapping of label to index
        class_metadata = constants.LABEL_PART_TO_CLASSES[self.my_hparams.get("label_part")]
        self.class_to_idx = class_metadata["class_to_idx"]

        ########################################################################
        #                  For Full US Sequence Data Loading                   #
        ########################################################################
        # Get unique patient ID and visits, corresponding to unique US seqs
        self.id_visit = None
        if self.my_hparams.get("full_seq"):
            self.id_visit = self.df[[self.id_col, self.visit_col]].drop_duplicates().apply(tuple, axis=1).to_numpy()

        ########################################################################
        #                           Image Transforms                           #
        ########################################################################
        transforms = transforms if transforms is not None else {}
        # If image size specified, at Resize transform
        if self.my_hparams.get("img_size"):
            transform_type = "geometric"
            transforms[transform_type] = [transforms[transform_type]] if transform_type in transforms else []
            transforms[transform_type].insert(0, T.Resize(self.my_hparams["img_size"]))
            transforms[transform_type] = T.Compose(transforms[transform_type])
        # If specified, standardize images by pre-computed channel means/stds
        if self.my_hparams.get("standardize_images"):
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
        if self.my_hparams.get("full_seq"):
            return self.get_sequence(index)

        # Get metadata row for image
        row = self.df.iloc[index]

        # HACK: Load segmentation mask before image to group in transform
        # NOTE: See __getitem__ in UltrasoundDataset
        seg_metadata = {}
        if self.my_hparams.get("load_seg_mask"):
            seg_metadata.update(self.get_segmentation_mask(index))
            if seg_metadata.get("has_seg_mask"):
                self._seg_masks[index] = seg_metadata["seg_mask"]

        # If returning an image
        X, metadata = self.get_single_image(index)

        # If segmentation mask was transformed, update
        if self.my_hparams.get("load_seg_mask") and seg_metadata.get("has_seg_mask"):
            seg_metadata["seg_mask"] = self._seg_masks[index]
            self._seg_masks.clear()

        # Encode label to integer (-1, if not found)
        metadata["label"] = int(self.class_to_idx.get(row[self.label_col], -1))

        # Record if has segmentation mask or not
        metadata["has_seg_mask"] = False

        # Early return, if not loading segmentation masks
        if not self.my_hparams.get("load_seg_mask"):
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

        # Get metadata row
        row = self.df.iloc[index]

        # CASE 1: Bladder image
        seg_fname_suffixes = []
        if str(row[self.label_col]).lower() in ("bladder", "none"):
            seg_fname_suffixes.append("_bseg")
        # CASE 2: Kidney image
        else:
            seg_fname_suffixes.append("_kseg")

        # Add flag to include liver
        if self.my_hparams.get("include_liver_seg"):
            seg_fname_suffixes.append("_lseg")

        # Load kidney/bladder/liver segmentations
        # NOTE: If 2+ exist, they're combined into 1 mask
        has_mask = False
        img_path = os.path.join(*row[self.path_cols].tolist())
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
            img_size = self.my_hparams["img_size"] or constants.IMG_SIZE
            metadata_overwrite["seg_mask"] = torch.full(img_size, True,
                                                        dtype=torch.bool)

        # Record if has mask or not
        metadata_overwrite["has_seg_mask"] = has_mask
        return metadata_overwrite


    def get_single_image(self, index):
        """
        Loads a single image with metadata.

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
        row = self.df.iloc[index]
        img_path = os.path.join(*row[self.path_cols].tolist())
        assert os.path.exists(img_path), f"No image at path specified! \n\t{img_path}"

        # Load image
        X = read_image(img_path, IMAGE_MODES[self.my_hparams.get("mode", 3)])

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

        # Store only filename, if specified. Otherwise, store the full path
        filename = img_path if self.my_hparams.get("full_path") else os.path.basename(img_path)

        # Prepare metadata
        metadata = {
            "filename": filename,
            "id": row[self.id_col],
            "visit": row[self.visit_col],
            "seq_number": row[self.seq_number_col],
            "dset": row[self.dset_col],
            # Store dataset index as well
            "dataset_idx": index,
        }

        return X, metadata


    def get_sequence(self, index):
        """
        Load an image sequence for the patient ID & visit given.

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

        # Filter for image sequence and order by sequence number
        patient_id, visit = self.id_visit[index]
        id_mask = self.df[self.id_col] == patient_id
        visit_mask = self.df[self.visit_col] == visit
        mask = id_mask & visit_mask
        df_seq = self.df[mask]
        df_seq = df_seq.sort_values(by=self.seq_number_col)

        # Get each individual metadata
        paths = df_seq[self.path_cols].apply(lambda path_parts: os.path.join(*path_parts.tolist())).tolist()
        labels = df_seq[self.label_col].tolist()

        # Load images
        imgs = []
        filenames = []
        for path in paths:
            imgs.append(read_image(path, IMAGE_MODES[self.my_hparams.get("mode", 3)]))
            filename = path if self.my_hparams.get("full_path") else os.path.basename(path)
            filenames.append(filename)

        X = torch.stack(imgs)

        # Encode labels to integers (-1, if not found)
        encoded_labels = torch.LongTensor([self.class_to_idx.get(label, -1) for label in labels])

        # Specify metadata
        metadata = {
            "filename": filenames,
            "label": encoded_labels,
            "id": [str(patient_id)],
            "visit": [str(visit)],
            "seq_number": df_seq[self.seq_number_col].tolist(),
            "dset": df_seq[self.dset_col].tolist(),
        }

        # Early return, if not loading segmentation masks
        if not self.my_hparams.get("load_seg_mask"):
            return X, metadata

        # Get segmentation mask for each image and concatenate them
        img_indices = np.nonzero(mask)
        accum_seg_metadata = defaultdict(list)
        for img_index in img_indices:
            curr_seg_metadata = self.get_segmentation_mask(img_index)
            for key, val in curr_seg_metadata.items():
                accum_seg_metadata[key].append(val)
        accum_seg_metadata = dict(accum_seg_metadata)
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
        raw_labels = self.df[self.label_col].tolist()
        if not encoded:
            return raw_labels

        # CASE 2: Encoded
        # Encode label to integer (-1, if not found)
        encoded_labels = [int(self.class_to_idx.get(label, -1)) for label in raw_labels]

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
        if self.my_hparams.get("full_seq"):
            return len(self.id_visit)

        return len(self.df)


def load_metadata_hocus_pocus(dset, **kwargs):
    dset_to_func = {
        "ood_hp_real": load_real_metadata,
        "ood_hp_noise": load_noise_metadata,
    }
    assert dset in dset_to_func, f"[Load Metadata (HocusPocus)] Invalid dset: {dset}!"
    load_func = dset_to_func[dset]

    # Load metadata
    df_metadata = load_func(**kwargs)
    return df_metadata
