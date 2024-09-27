"""
tcl_dataset.py

Description: Contains module to load data for Twin Contrastive Learning (TCL)
             for noisy labels.
"""

# Non-standard libraries
import torch
import torchvision.transforms.v2 as T
from lightly.data import LightlyDataset
from torch.utils.data import DataLoader

# Custom libraries
from src.data_prep import ssl_collate_fn, utils
from src.data_prep.dataset import (UltrasoundDataModule,
                                   UltrasoundDatasetDataFrame)


################################################################################
#                                  Constants                                   #
################################################################################
# Default parameters for data loader
DEFAULT_DATALOADER_PARAMS = {
    "batch_size": 128,
    "shuffle": True,
    "num_workers": 7,
}


################################################################################
#                             Data Module Classes                              #
################################################################################
class TCLDataModule(UltrasoundDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities in the self-supervised setting.
    """

    def __init__(self, df=None, img_dir=None, mode=3,
                 augment_training=True,
                 **kwargs):
        """
        Initialize TCLDataModule object.

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
        mode : int, optional
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 3.
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
        # Raise error, if imbalanced sampler specified
        if kwargs.get("imbalanced_sampler"):
            raise RuntimeError("Imbalanced sampler is not supported for BYOL!")

        # NOTE: Sampler conflicts with shuffle=True
        if kwargs.get("full_seq"):
            raise RuntimeError("US sequences is not supported by TCL")

        # Pass UltrasoundDataModule arguments
        super().__init__(
            df,
            img_dir=img_dir,
            mode=mode,
            augment_training=False,
            default_dl_params=DEFAULT_DATALOADER_PARAMS,
            **kwargs)

        # Overwrite augmentations
        self.augmentations = {}

        # CASE 1: If augmenting, instantiate weak and strong transforms
        if augment_training:
            weak_transform = T.Compose(list(utils.prep_weak_augmentations(
                img_size=self.img_size).values()))
            strong_transform = T.Compose(list(utils.prep_strong_augmentations(
                img_size=self.img_size,
                crop_scale=kwargs.get("crop_scale", 0.5)).values()))
        # CASE 2: If not augmenting, create placeholders
        else:
            weak_transform = torch.nn.Identity()
            strong_transform = torch.nn.Identity()

        # Create collate function
        self.collate_fn = ssl_collate_fn.TCLCollateFunction(
            weak_transform, strong_transform,
        )


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

        return DataLoader(test_dataset, **self.val_dataloader_params)
