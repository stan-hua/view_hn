"""
dataset.py

Description: Contains functions/classes to load a dataset in PyTorch for
             self-supervised pretraining.
"""

# Non-standard libraries
from torch.utils.data import BatchSampler, SequentialSampler

# Custom libraries
from src.data_prep.dataset import *


################################################################################
#                             Data Module Classes                              #
################################################################################
class SelfSupervisedUltrasoundDataModule(UltrasoundDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities in the self-supervised setting.
    """
    def __init__(self, dataloader_params=None, df=None, img_dir=None,
                 full_seq=False, mode=3,
                 same_label=False,
                 **kwargs):
        """
        Initialize SelfSupervisedUltrasoundDataModule object.

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
        # Set default DataLoader parameters for self-supervised taska
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
            self.collate_fn = SameLabelCollateFunction(self.transforms)
        else:
            # Creates two augmentation from the same image
            self.collate_fn = SimCLRCollateFunction(self.transforms)


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


################################################################################
#                           CollateFunction Classes                            #
################################################################################
class SimCLRCollateFunction(lightly.data.collate.BaseCollateFunction):
    """
    SimCLRCollateFunction. Used to create paired image augmentations with custom
    batch data.
    """
    def forward(self, batch):
        """
        For each image in the batch, creates two random augmentations on the
        same image and pairs them.

        Parameters
        ----------
            batch: tuple of (torch.Tensor, dict)
                Tuple of image tensors and metadata dict (containing filenames,
                etc.)

        Returns
        -------
            tuple of ((torch.Tensor, torch.Tensor), dict).
                The two tensors consist of corresponding images transformed from
                the original batch.
        """
        batch_size = len(batch)

        # Accumulate lists for keys in metadata dicts
        metadata_accum = {}
        for item in batch:
            metadata = item[1]
            for key, val in metadata.items():
                if key not in metadata_accum:
                    metadata_accum[key] = []
                metadata_accum[key].append(val)

        # Perform random augmentation on each image twice
        X_transformed = [self.transform(batch[i % batch_size][0]).unsqueeze_(0)
            for i in range(2 * batch_size)]

        # Tuple of paired transforms
        X_transformed_paired = (
            torch.cat(X_transformed[:batch_size], 0),
            torch.cat(X_transformed[batch_size:], 0)
        )

        return X_transformed_paired, metadata_accum


class SameLabelCollateFunction(lightly.data.collate.BaseCollateFunction):
    """
    SameLabelCollateFunction. Used to create two augmentations for each images,
    and pair images with the same label.
    """

    def forward(self, batch):
        """
        For each image in the batch, creates two random augmentations on each
        image and pairs same-label images.

        Note
        ----
        Because of two augmentations per image, there is no possibility that
        a label having only 1 image (in the batch) will have no pair. In the
        worst case, it will pair with a random augmentation of itself.

        Parameters
        ----------
            batch: tuple of (torch.Tensor, dict)
                Tuple of image tensors and metadata dict (containing filenames,
                etc.)

        Returns
        -------
            tuple of ((torch.Tensor, torch.Tensor), dict).
                The two tensors with corresponding images that have the same
                label.
        """
        # Accumulate lists for keys in metadata dicts
        metadata_accum = {}
        for item in batch:
            metadata = item[1]
            for key, val in metadata.items():
                if key not in metadata_accum:
                    metadata_accum[key] = []

                if not isinstance(val, str):
                    try:
                        metadata_accum[key].extend(val)
                    except:
                        metadata_accum[key].append(val)
                else:
                    metadata_accum[key].append(val)

        # Convert metadata lists to arrays
        for key, val_list in metadata_accum.items():
            metadata_accum[key] = np.array(val_list)

        # Group by label
        # NOTE: Precondition that label exists
        # NOTE: Duplicate by 2 to account for the two augmentations
        labels = np.concatenate([metadata_accum["label"],
                                 metadata_accum["label"]])
        label_to_indices = {
            label: np.argwhere(labels == label).squeeze() \
                for label in np.unique(labels)
        }

        # Create indices to pair images of the same label
        first_idx = []
        second_idx = []
        for _, indices in label_to_indices.items():
            # Pair randomly selected images
            # Shuffle indices
            n = len(indices)
            chosen_indices = np.random.choice(indices, size=n, replace=False)
            first_idx.extend(chosen_indices[:int(n/2)])
            second_idx.extend(chosen_indices[int(n/2):])

        # Get images
        # If batch size is 1, but contains multiple US images
        if len(batch) == 1 and len(batch[0][0]) > 1:
            imgs = batch[0][0]
        else:
            imgs = [data[0] for data in batch]

        # Perform random augmentation on each image twice
        batch_size = len(imgs)
        X_transformed = [
            self.transform(imgs[i % batch_size]).unsqueeze_(0)
            for i in range(batch_size * 2)]

        # Tuple of paired transforms
        X_transformed_paired = (
            torch.cat([X_transformed[idx] for idx in first_idx], 0),
            torch.cat([X_transformed[idx] for idx in second_idx], 0)
        )

        return X_transformed_paired, metadata_accum
