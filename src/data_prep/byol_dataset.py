"""
byol_dataset.py

Description: Contains module to load data for Bootstrap Your Own Latent (BYOL)
             self-supervised pretraining.
"""

# Non-standard libraries
import torchvision.transforms.v2 as T
from lightly.data import LightlyDataset
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

# Custom libraries
from src.data_prep import ssl_collate_fn
from src.data_prep.dataset import UltrasoundDataModule, UltrasoundDatasetDataFrame


################################################################################
#                                  Constants                                   #
################################################################################
# Default parameters for data module
BYOL_DEFAULT_DM_HPARAMS = {
    "same_label": False,
    "custom_collate": None,
    "augment_training": True
}


################################################################################
#                             Data Module Classes                              #
################################################################################
class BYOLDataModule(UltrasoundDataModule):
    """
    BYOLDataModule class.

    Note
    ----
    Used to create training/validation/test dataloaders for BYOL
    """

    def __init__(self, hparams, df=None, **overwrite_params):
        """
        Initialize BYOLDataModule object.

        Note
        ----
        Either df or img_dir must be exclusively specified to load in data.

        By default, does not split data.

        Parameters
        ----------
        hparams : dict
            Data hyperparameters additional to UltrasoundDataModule include:
            same_label : bool, optional
                If True, positive samples are same-patient images with the same
                label, by default False.
            custom_collate : str, optional
                One of (None, "same_label"). "same_label" pairs images of the same
                label. Defaults to None, which is the regular SimCLR collate
                function.
            augment_training : bool
                If True, add random augmentations during training, by default True.
        df : pd.DataFrame, optional
            Contains paths to image files and labels for each image, by default
            None
        **overwrite_params : Any
            Keyword arguments to overwrite default hyperparameters
        """
        # Add default hyperparameters
        hparams = hparams.copy() or {}
        hparams.update({k:v for k,v in BYOL_DEFAULT_DM_HPARAMS.items() if k not in hparams})

        super().__init__(hparams, df, augment_training=False, **overwrite_params)

        # Raise error, if imbalanced sampler specified
        if self.my_hparams.get("imbalanced_sampler"):
            raise RuntimeError("Imbalanced sampler is not supported for BYOL!")

        # If same-label sampling, ensure correct collate function is used
        if self.my_hparams.get("same_label"):
            self.my_hparams["custom_collate"] = "same_label"

        # If using full sequence, turn off shuffling
        if self.my_hparams.get("full_seq"):
            self.my_hparams["shuffle"] = False

        # If specified, turn off augmentations during SSL
        if not self.my_hparams.get("augment_training"):
            self.augmentations = {"identity": torch.nn.Identity()}

        # Determine collate function
        augment_list = T.Compose(list(self.augmentations.values()))
        # 1. Pairs same-label images
        if self.same_label and self.custom_collate == "same_label":
            self.collate_fn = ssl_collate_fn.BYOLSameLabelCollateFunction(augment_list)
        else:
        # 2. Pairs same-image pairs (different augmentation)
            self.collate_fn = ssl_collate_fn.SimCLRCollateFunction(augment_list)


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

        # Choose sampling method
        sampler = None
        if self.my_hparams.get("full_seq"):
            sampler = BatchSampler(SequentialSampler(labeled_dataset),
                                   batch_size=1,
                                   drop_last=False)

        # Store train dataloader
        name_to_loader = {}

        # Create DataLoader with parameters specified
        name_to_loader["id"] = DataLoader(
            labeled_dataset,
            drop_last=True, collate_fn=self.collate_fn, sampler=sampler,
            **base_dl_params
        )
        return dataloader
