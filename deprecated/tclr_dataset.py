"""
tclr_dataset.py

Description: Contains module to load data for Temporal Contrastive Learning
             (TCLR) self-supervised pretraining.

Note: Model trained on this dataset attempts to take advantage of temporal
      information in patient ultrasound image sequences.
"""

# Standard libraries
import logging

# Non-standard libraries
import pandas as pd
import torchvision.transforms.v2 as T
from lightly.data import LightlyDataset
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

# Custom libraries
from src.data_prep import ssl_collate_fn, utils
from src.data_prep.dataset import (UltrasoundDataModule,
                                   UltrasoundDatasetDataFrame)


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)


################################################################################
#                             Data Module Classes                              #
################################################################################
# TODO: Update train dataloader to stop using split_to_paths
class TCLRDataModule(UltrasoundDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities in the self-supervised setting.
    """
    def __init__(self, dataloader_params=None, df=None, img_dir=None,
                 mode=3,
                 seq_length=18,
                 **kwargs):
        """
        Initialize TCLRDataModule object.

        Note
        ----
        Either df or img_dir must be exclusively specified to load in data.

        By default, does not split data.

        Filters training set for ultrasound image sequences with at least
        `seq_length` images.

        Parameters
        ----------
        dataloader_params : dict, optional
            Used to override default parameters for DataLoaders, by default None
        df : pd.DataFrame, optional
            Contains paths to image files and labels for each image, by default
            None
        img_dir : str, optional
            Path to directory containing ultrasound images, by default None
        mode : int, optional
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 3.
        seq_length : int, optional
            Fixed number of frames for a sampled ultrasound image sequence, by
            default 18.
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
                                     "num_workers": 7}
        if dataloader_params:
            default_dataloader_params.update(dataloader_params)

        # Extra SSL flags
        self.seq_length = seq_length

        # Ensure processing data in full sequence
        if "full_seq" in kwargs:
            kwargs["full_seq"] = True

        # Pass UltrasoundDataModule arguments
        super().__init__(default_dataloader_params,
                         df=df, img_dir=img_dir,
                         mode=mode,
                         **kwargs)
        self.val_dataloader_params["batch_size"] = \
            default_dataloader_params["batch_size"]

        # Random augmentations
        self.transforms = T.Compose([
            T.RandomAdjustSharpness(1.25, p=0.25),
            T.RandomApply([T.GaussianBlur(1, 0.1)], p=0.5),
            T.RandomRotation(15),
            T.RandomResizedCrop(self.img_size, scale=(0.5, 1)),
        ])

        # Collate function to create 2 augmented versions of each clip
        self.collate_fn = ssl_collate_fn.TCLRCollateFunction(self.transforms)


    def train_dataloader(self):
        """
        Returns DataLoader for training set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for training data
        """
        df_train = pd.DataFrame({
            "filename": self.split_to_paths["train"],
            "label": self.split_to_labels["train"]
        })

        # Get patient ID, visit number and sequence number, from orig. table
        df_train = utils.left_join_filtered_to_source(
            df_train, self.df,
            index_cols="filename")

        # Ensure US image sequences to have exactly `seq_length` frames
        df_train = utils.restrict_seq_len(df_train, n=self.seq_length)

        # Instantiate UltrasoundDatasetDataFrame
        train_dataset = UltrasoundDatasetDataFrame(
            df_train,
            img_dir=self.img_dir,
            full_seq=True,
            img_size=self.img_size,
            mode=self.mode,
            label_part=self.label_part,
        )

        # Transform to LightlyDataset
        train_dataset = LightlyDataset.from_torch_dataset(
            train_dataset,
            transform=self.transforms)

        # Create DataLoader with parameters specified
        return DataLoader(train_dataset,
                          drop_last=True,
                          collate_fn=self.collate_fn,
                          **self.train_dataloader_params)


    def val_dataloader(self):
        """
        Returns DataLoader for validation set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for validation data
        """
        df_val = pd.DataFrame({
            "filename": self.split_to_paths["val"],
            "label": self.split_to_labels["val"]
        })

        # Get patient ID, visit number and sequence number, from orig. table
        df_val = utils.left_join_filtered_to_source(
            df_val, self.df,
            index_cols="filename")

        # Ensure US image sequences to have exactly `seq_length` frames
        df_val = utils.restrict_seq_len(df_val, n=self.seq_length)

        # Instantiate UltrasoundDatasetDataFrame
        val_dataset = UltrasoundDatasetDataFrame(
            df_val,
            img_dir=self.img_dir,
            full_seq=True,
            img_size=self.img_size,
            mode=self.mode,
            label_part=self.label_part,
        )

        # Transform to LightlyDataset
        val_dataset = LightlyDataset.from_torch_dataset(
            val_dataset,
            transform=self.transforms)

        # Create DataLoader with parameters specified
        return DataLoader(val_dataset,
                          drop_last=True,
                          collate_fn=self.collate_fn,
                          **self.val_dataloader_params)
