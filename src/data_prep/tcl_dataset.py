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
from src.data_prep.dataset import UltrasoundDataModule, UltrasoundDatasetDataFrame


################################################################################
#                                  Constants                                   #
################################################################################
# Default parameters for data module
TCL_DEFAULT_DM_HPARAMS = {
    "augment_training": True
}


################################################################################
#                             Data Module Classes                              #
################################################################################
class TCLDataModule(UltrasoundDataModule):
    """
    TCLDataModule class.

    Note
    ----
    Used to create training/validation/test dataloaders for TCL
    """

    def __init__(self, hparams, df=None, **overwrite_params):
        """
        Initialize TCLDataModule object.

        Note
        ----
        Either df or img_dir must be exclusively specified to load in data.

        By default, does not split data.

        Parameters
        ----------
        hparams : dict
            Data hyperparameters for UltrasoundDataModule
        df : pd.DataFrame, optional
            Contains paths to image files and labels for each image, by default
            None
        **overwrite_params : Any
            Keyword arguments to overwrite default hyperparameters
        """
        # Add default hyperparameters
        hparams = hparams.copy() or {}
        hparams.update({k:v for k,v in TCL_DEFAULT_DM_HPARAMS.items() if k not in hparams})

        super().__init__(hparams, df, augment_training=False, **overwrite_params)

        # Raise error, if imbalanced sampler specified
        if self.my_hparams.get("imbalanced_sampler"):
            raise RuntimeError("Imbalanced sampler is not supported for TCL!")
        # Raise error, if using full sequences
        if self.my_hparams.get("full_seq"):
            raise RuntimeError("Full sequence is not supported for TCL!")

        # If specified, turn off augmentations during SSL
        if not self.my_hparams.get("augment_training"):
            self.augmentations = {"identity": torch.nn.Identity()}

        # Determine collate function
        # CASE 1: If augmenting, instantiate weak and strong transforms
        if self.my_hparams.get("augment_training"):
            weak_transform = T.Compose(list(utils.prep_weak_augmentations(
                img_size=self.my_hparams["img_size"]).values()))
            strong_transform = T.Compose(list(utils.prep_strong_augmentations(
                img_size=self.my_hparams["img_size"],
                crop_scale=self.my_hparams.get("crop_scale", 0.5)).values()))
        # CASE 2: If not augmenting, create placeholders
        else:
            weak_transform = torch.nn.Identity()
            strong_transform = torch.nn.Identity()

        # Create collate function
        self.collate_fn = ssl_collate_fn.TCLCollateFunction(weak_transform, strong_transform)


    def create_id_dataloaders(self, split):
        """
        Overwrite in-distribution data loader to load data for self-supervised
        learning.

        Parameters
        ----------
        split : str
            Data split

        Returns
        -------
        dict
            Contains dataloader for in-distribution data
        """
        assert split in ["train", "val", "test"]

        # Prepare dataloader parameters
        base_dl_params = self.create_dl_params(split)

        # Get labeled data
        df_metadata = self.df[self.df["split"] == split]
        label_col = self.my_hparams.get("label_col", "label")
        na_mask = df_metadata[label_col].isna()
        df_labeled = df_metadata[~na_mask].reset_index(drop=True)

        # Create dataset for labeled data
        labeled_dataset = UltrasoundDatasetDataFrame(
            df_labeled,
            hparams=self.my_hparams,
            transforms=self.transforms,
        )

        # Transform to LightlyDataset
        # NOTE: `transforms` only contains basic image pre-processing steps
        labeled_dataset = LightlyDataset.from_torch_dataset(
            labeled_dataset,
            transform=self.transforms)

        # Store train dataloader
        name_to_loader = {}

        # Create DataLoader with parameters specified
        name_to_loader["id"] = DataLoader(
            labeled_dataset,
            drop_last=True, collate_fn=self.collate_fn,
            **base_dl_params
        )
        return dataloader
