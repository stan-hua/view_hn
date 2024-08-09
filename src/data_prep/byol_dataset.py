"""
byol_dataset.py

Description: Contains module to load data for Bootstrap Your Own Latent (BYOL)
             self-supervised pretraining.
"""

# Non-standard libraries
import pandas as pd
import torch
from lightly.data import LightlyDataset
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

# Custom libraries
from src.data_prep import ssl_collate_fn, utils
from src.data_prep.dataset import (UltrasoundDataModule,
                                   UltrasoundDatasetDataFrame)


################################################################################
#                             Data Module Classes                              #
################################################################################
class BYOLDataModule(UltrasoundDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities in the self-supervised setting.
    """
    def __init__(self, dataloader_params=None, df=None, img_dir=None,
                 full_seq=False, mode=3,
                 same_label=False,
                 custom_collate=None,
                 augment_training=True,
                 **kwargs):
        """
        Initialize BYOLDataModule object.

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
        same_label : bool, optional
            If True, positive samples are images with the same label, by default
            False.
        custom_collate : str, optional
            One of (None, "same_label"). "same_label" pairs images of the same
            label. Defaults to None, which is the regular SimCLR collate
            function.
        augment_training : bool
            If True, add random augmentations during training, by default True.
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
        # Set default DataLoader parameters for self-supervised task
        default_dataloader_params = {"batch_size": 128,
                                     "shuffle": True,
                                     "num_workers": 7,
                                     "pin_memory": True}
        if dataloader_params:
            default_dataloader_params.update(dataloader_params)

        # Pair together same-label images, if specified
        self.same_label = same_label
        # If same-label sampling, ensure correct collate function is used
        if self.same_label:
            custom_collate = "same_label"

        # Ensure custom collate function is as expected
        assert custom_collate in (None, "same_label"), \
            "Invalid `custom_collate` provided! (%s)" % (custom_collate,)
        self.custom_collate = custom_collate

        # NOTE: Sampler conflicts with shuffle=True
        if full_seq:
            default_dataloader_params["shuffle"] = False

        # Pass UltrasoundDataModule arguments
        super().__init__(
            default_dataloader_params, df, img_dir, full_seq, mode,
            augment_training=False,
            **kwargs)
        self.val_dataloader_params["batch_size"] = \
            default_dataloader_params["batch_size"]

        # If specified, turn off augmentations during SSL
        if not augment_training:
            self.augmentations = torch.nn.Identity()

        # Determine collate function
        # 1. Pairs same-label images
        if self.same_label and self.custom_collate == "same_label":
            self.collate_fn = ssl_collate_fn.BYOLSameLabelCollateFunction(
                self.augmentations)
        else:
        # 2. Pairs same-image pairs (different augmentation)
            self.collate_fn = ssl_collate_fn.SimCLRCollateFunction(
                self.augmentations)


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

        # Transform to LightlyDataset
        # NOTE: `transforms` only contains basic image pre-processing steps
        train_dataset = LightlyDataset.from_torch_dataset(
            train_dataset,
            transform=self.transforms)

        # Choose sampling method
        sampler = None
        if self.us_dataset_kwargs.get("full_seq"):
            sampler = BatchSampler(SequentialSampler(train_dataset),
                                   batch_size=1,
                                   drop_last=False)

        # Create DataLoader with parameters specified
        return DataLoader(train_dataset,
                          drop_last=True,
                          collate_fn=self.collate_fn,
                          sampler=sampler,
                          **self.train_dataloader_params)


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

        # Transform to LightlyDataset
        # NOTE: `transforms` only contains basic image pre-processing steps
        val_dataset = LightlyDataset.from_torch_dataset(
            val_dataset,
            transform=self.transforms)

        # Choose sampling method
        sampler = None
        if self.us_dataset_kwargs.get("full_seq"):
            sampler = BatchSampler(SequentialSampler(val_dataset),
                                   batch_size=1,
                                   drop_last=False)

        # Create DataLoader with parameters specified
        return DataLoader(val_dataset,
                          drop_last=True,
                          collate_fn=self.collate_fn,
                          sampler=sampler,
                          **self.val_dataloader_params)


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

        # Transform to LightlyDataset
        # NOTE: `transforms` only contains basic image pre-processing steps
        test_dataset = LightlyDataset.from_torch_dataset(
            test_dataset,
            transform=self.transforms)

        # Choose sampling method
        sampler = None
        if self.us_dataset_kwargs.get("full_seq"):
            sampler = BatchSampler(SequentialSampler(test_dataset),
                                   batch_size=1,
                                   drop_last=False)

        # Create DataLoader with parameters specified
        return DataLoader(test_dataset,
                          drop_last=True,
                          collate_fn=self.collate_fn,
                          sampler=sampler,
                          **self.val_dataloader_params)
