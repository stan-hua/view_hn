"""
moco_dataset.py

Description: Contains functions/classes to load a dataset in PyTorch for
             self-supervised pretraining.
"""

# Non-standard libraries
import pandas as pd
import torchvision.transforms as T
from lightly.data import LightlyDataset
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

# Custom libraries
from src.data_prep import ssl_collate_fn, utils
from src.data_prep.dataset import (UltrasoundDataModule,
                                   UltrasoundDatasetDataFrame)


################################################################################
#                             Data Module Classes                              #
################################################################################
class MoCoDataModule(UltrasoundDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities in the self-supervised setting.
    """
    def __init__(self, dataloader_params=None, df=None, img_dir=None,
                 full_seq=False, mode=3,
                 same_label=False,
                 **kwargs):
        """
        Initialize MoCoDataModule object.

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
            If True, positive samples are same-patient images with the same
            label, by default False.
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

        # Extra SSL flags
        self.same_label = same_label
        # NOTE: If same label, each batch must be images from the same sequence
        if self.same_label:
            full_seq = True
            # NOTE: Sampler conflicts with shuffle=True
            default_dataloader_params["shuffle"] = False

        # Pass UltrasoundDataModule arguments
        super().__init__(default_dataloader_params, df, img_dir, full_seq, mode,
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

        # Determine collate function
        if self.same_label:
            self.collate_fn = ssl_collate_fn.SameLabelCollateFunction(
                self.transforms)
        else:
            # Creates two augmentation from the same image
            self.collate_fn = ssl_collate_fn.SimCLRCollateFunction(
                self.transforms)


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

        # Add metadata for patient ID, visit number and sequence number
        utils.extract_data_from_filename(df_train)

        # Instantiate UltrasoundDatasetDataFrame
        train_dataset = UltrasoundDatasetDataFrame(df_train, self.img_dir,
                                                   self.full_seq,
                                                   img_size=self.img_size,
                                                   mode=self.mode,
                                                   label_part=self.label_part)

        # Transform to LightlyDataset
        train_dataset = LightlyDataset.from_torch_dataset(
            train_dataset,
            transform=self.transforms)

        # Choose sampling method
        sampler = None
        if self.full_seq:
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
        df_val = pd.DataFrame({
            "filename": self.dset_to_paths["val"],
            "label": self.dset_to_labels["val"]
        })

        # Add metadata for patient ID, visit number and sequence number
        utils.extract_data_from_filename(df_val)

        val_dataset = UltrasoundDatasetDataFrame(df_val, self.img_dir,
                                                 self.full_seq,
                                                 img_size=self.img_size,
                                                 mode=self.mode,
                                                 label_part=self.label_part)

        # Transform to LightlyDataset
        val_dataset = LightlyDataset.from_torch_dataset(
            val_dataset,
            transform=self.transforms)

        # Choose sampling method
        sampler = None
        if self.full_seq:
            sampler = BatchSampler(SequentialSampler(val_dataset),
                                   batch_size=1,
                                   drop_last=False)

        # Create DataLoader with parameters specified
        return DataLoader(val_dataset,
                          drop_last=True,
                          collate_fn=self.collate_fn,
                          sampler=sampler,
                          **self.val_dataloader_params)
