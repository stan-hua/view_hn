"""
segment_dataset.py

Description: Used to load ultrasound images with segmentation masks
"""

# Standard libraries
import glob
import logging
import os

# Non-standard libraries
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode
from maskedtensor import masked_tensor

# Custom libraries
from src.data import constants
from src.data_prep import utils


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Torchvision Grayscale/RGB constants
IMAGE_MODES = {1: ImageReadMode.GRAY, 3: ImageReadMode.RGB}


################################################################################
#                                   Classes                                    #
################################################################################
class SegmentedUSModule(pl.LightningDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities.
    """
    def __init__(self, mask_img_dir, src_img_dir,
                 dataloader_params=None,
                 reverse_mask=False,
                 full_seq=False, mode=3, **kwargs):
        """
        Initialize SegmentedUSModule object.

        Parameters
        ----------
        mask_img_dir : str
            Path to flat directory containing image segmentation masks.
        src_img_dir : str
            Path to flat directory containing ultrasound images.
        dataloader_params : dict, optional
            Used to override default parameters for DataLoaders, by default None
        reverse_mask : bool, optional
            If True, reverses segmentation mask, by default False.
        full_seq : bool, optional
            If True, each item has all ordered images for one full
            ultrasound sequence (determined by patient ID and visit). If False,
            treats each image under a patient as independent, by default False.
        mode : int
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 3.
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

        # Used to instantiate SegmentedUltrasoundDataset
        self.mask_img_dir = mask_img_dir
        self.src_img_dir = src_img_dir
        self.reverse_mask = reverse_mask
        self.dataset = None
        self.full_seq = full_seq
        self.mode = mode
        self.img_size = kwargs.get("img_size", 258)

        ########################################################################
        #                        DataLoader Parameters                         #
        ########################################################################
        # Default parameters for data loader
        default_data_params = {'batch_size': 1,
                               'shuffle': False,
                               'num_workers': 0,
                               'pin_memory': True}

        # Parameters for training/validation DataLoaders
        self.train_dataloader_params = default_data_params
        if dataloader_params:
            self.train_dataloader_params.update(dataloader_params)

        # NOTE: Shuffle is turned off during validation/test
        # NOTE: Batch size is set to 1 during validation/test
        self.val_dataloader_params = self.train_dataloader_params.copy()
        self.val_dataloader_params['batch_size'] = 1
        self.val_dataloader_params['shuffle'] = False


    def setup(self, stage='fit'):
        """
        Prepares data for model training/validation/testing
        """
        LOGGER.warning("SegmentedUSModule does not implement setup!")


    def train_dataloader(self):
        """
        Returns DataLoader for training set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for training data
        """
        # Instantiate SegmentedUSDataset
        train_dataset = SegmentedUSDataset(self.mask_img_dir, self.src_img_dir,
                                           reverse_mask = self.reverse_mask,
                                           full_seq=self.full_seq,
                                           img_size=self.img_size,
                                           mode=self.mode)

        # Create DataLoader with parameters specified
        return DataLoader(train_dataset, **self.train_dataloader_params,
                          collate_fn=custom_collate_fn)


def custom_collate_fn(batch):
    """
    Custom collate function for Masked Tensors.

    Note
    ----
    torch.stack isn't 

    Parameters
    ----------
    batch : list of (MaskedTensors, dict) tuples
        Batch where each item is a pair of image (MaskedTensor) and metadata
        dictionary

    Returns
    -------
    _type_
        _description_
    """
    return batch[0]


class SegmentedUSDataset(torch.utils.data.Dataset):
    """
    Dataset to load images from a directory.
    """
    def __init__(self, mask_img_dir, src_img_dir, reverse_mask=False,
                 full_seq=False, img_size=None, mode=3):
        """
        Initialize KidneyDatasetDir object.

        Note
        ----
        Filenames in images `mask_img_dir` and `src_img_dir` must correspond.

        Parameters
        ----------
        mask_img_dir : str
            Path to flat directory containing image segmentation masks.
        src_img_dir : str
            Path to flat directory containing ultrasound images.
        reverse_mask : bool, optional
            If True, reverses segmentation mask, by default False.
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
        """
        assert mode in (1, 3)
        self.mode = IMAGE_MODES[mode]
        self.reverse_mask = reverse_mask

        # Get all images in flat directory
        src_paths = glob.glob(os.path.join(src_img_dir, "*"))
        mask_paths = glob.glob(os.path.join(mask_img_dir, "*"))
        # Filter for files present in both directories
        src_filenames = [os.path.basename(path) for path in src_paths]
        mask_filenames = [os.path.basename(path) for path in mask_paths]
        both_filenames = set(src_filenames).intersection(set(mask_filenames))

        self.src_paths = np.array([os.path.join(src_img_dir, filename) \
                                   for filename in both_filenames])
        self.mask_paths = np.array([os.path.join(mask_img_dir, filename) \
                                    for filename in both_filenames])

        # Check that paths are not empty
        if not both_filenames:
            raise RuntimeError("No corresponding image src/mask segmentations "
                               "found in src/mask directories provided!")

        # Get all patient IDs
        self.ids = np.array(utils.get_from_paths(self.src_paths))

        # Get hospital visit number
        self.visits = utils.get_from_paths(self.src_paths, "visit")

        # Get number in US sequence
        self.seq_numbers = utils.get_from_paths(self.src_paths, "seq_number")

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
        transforms = []
        if img_size:
            transforms.append(T.Resize(img_size))

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
        src_img_path = self.src_paths[index]
        mask_img_path = self.mask_paths[index]

        # Load images
        X = self.load_masked_image(src_img_path, mask_img_path)

        # Get metadata from filename
        filename = os.path.basename(src_img_path)
        patient_id = self.ids[index]
        visit = self.visits[index]
        seq_number = self.seq_numbers[index]
        # NOTE: ID naming is used to identify hospital
        hospital = "Stanford" if filename.startswith("SU2") else "SickKids"

        metadata = {"filename": filename, "id": patient_id,
                    "visit": visit, "seq_number": seq_number,
                    "hospital": hospital}

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
        src_paths = self.src_paths[mask]
        mask_paths = self.mask_paths[mask]
        seq_numbers = self.seq_numbers[mask]

        # 3. Order by sequence number
        sort_idx = np.argsort(seq_numbers)
        src_paths = src_paths[sort_idx]
        mask_paths = mask_paths[sort_idx]
        seq_numbers = seq_numbers[sort_idx]

        # 4. Load images
        imgs = []
        for src_path, mask_path in tuple(zip(src_paths, mask_paths)):
            imgs.append(self.load_masked_image(src_path, mask_path)) 
        X = torch.stack(imgs)

        # 4.1 If only 1 image for a sequence, pad first dimension
        if len(imgs) == 1:
            X = torch.unsqueeze(X, 0)

        # 5. Metadata
        filenames = [os.path.basename(path) for path in src_paths]
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
        return len(self.src_paths)


    def load_masked_image(self, src_img_path, mask_img_path):
        """
        Loads a masked image given the path to the source image and the image
        mask.

        Parameters
        ----------
        src_img_path : str
            Path to the source image
        mask_img_path : str
            Path to the image mask

        Returns
        -------
        maskedtensor.maskedtensor
            Masked tensor (image with mask)
        """
        assert os.path.exists(src_img_path), "No src image at path specified!"
        assert os.path.exists(mask_img_path), "No mask image at path specified!"

        # Load source image
        X_src = read_image(src_img_path, self.mode)
        X_src = self.transforms(X_src)
        # Normalize between [0, 1]
        X_src = X_src / 255.

        # Load mask
        X_mask = read_image(mask_img_path, self.mode)
        X_mask = self.transforms(X_mask)
        # Convert to Boolean Tensor
        X_mask = X_mask.to(torch.BoolTensor())
        # If specified, reverse mask
        if self.reverse_mask:
            X_mask = ~X_mask
        
        # Create masked image tensor
        X = masked_tensor(X_src, X_mask, False)

        return X
