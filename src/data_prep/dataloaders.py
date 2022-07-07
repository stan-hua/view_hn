"""
dataloaders.py

Description: Contains functions/classes to load dataset in PyTorch.
"""
# Standard libraries
from abc import abstractmethod
import glob
import os

# Non-standard libraries
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode

################################################################################
#                                  Constants                                   #
################################################################################
SEED = 42
IMAGE_MODES = {1: ImageReadMode.GRAY, 3: ImageReadMode.RGB}

################################################################################
#                                  Functions                                   #
################################################################################
def load_dataset_from_dir(dir):
    """
    Loads image dataset from directory of images.

    Parameters
    ----------
    dir : str
        Path to directory containing ultrasound images.
    
    Returns
    -------
    torch.utils.data.Dataset
        Contains images and metadata from filename
    """
    return UltrasoundDatasetDir(dir, img_size=None)


def load_dataset_from_dataframe(df, dir=None):
    """
    Loads image dataset from dataframe of image paths and labels.

    Parameters
    ----------
    df : pd.DataFrame
        Contains column with absolute/relative path to images, and labels
    dir : str, optional
        Path to directory containing ultrasound images, by default None.
    
    Returns
    -------
    torch.utils.data.Dataset
        Contains images, metadata from filename, and labels from dataframe
    """
    return UltrasoundDatasetDataFrame(df, dir=dir)


################################################################################
#                             Data Module Classes                              #
################################################################################
class UltrasoundDataModule(pl.LightningDataModule):
    """
    Top-level object used to access all data preparation and loading
    functionalities.
    """
    def __init__(self, dataloader_params=None, df=None, dir=None,
                 train_split=1.0, debug=True, **kwargs):
        """
        Initialize UltrasoundDataModule object.

        Note
        ----
        Either df or dir must be specified to load in data.

        Parameters
        ----------
        dataloader_params : dict, optional
            Used to overrite default parameters for DataLoaders, by default None
        split : bool
            If True, splits the data into training and validation
        df : pd.DataFrame, optional
            Contains paths to image files and labels for each image, by default
            None
        dir : str, optional
            Path to directory containing ultrasound images, by default None
        train_split : float, optional
            Percentage of data to leave for training, by default 1.0
        debug : bool, optional
            If True, saves data to object for viewing, by default True
        **kwargs : dict
            Keyword arguments
        """
        super().__init__()
        assert dataloader_params is None or isinstance(dataloader_params, dict)

        self.debug = debug

        # Used to instantiate UltrasoundDataset
        self.df = df
        self.dir = dir
        self.dataset = None

        # For splitting dataset into training/validation sets
        self.train_split = train_split
        self.train_dataset = None
        self.val_dataset = None

        # Default parameters for data loader
        default_data_params = {'batch_size': 32,
                               'shuffle': False,
                               'num_workers': 4,
                               'pin_memory': True}

        # Parameters for training/validation DataLoaders
        self.train_dataloader_params = default_data_params
        if dataloader_params:
            self.train_dataloader_params.update(dataloader_params)

        self.val_dataloader_params = self.train_dataloader_params.copy()
        self.val_dataloader_params['batch_size'] = 1
        self.val_dataloader_params['shuffle'] = False

    def setup(self, stage='fit'):
        """
        Prepares data for model fitting/testing
        """ 
        # Load dataset
        if self.df is not None:
            self.dataset = UltrasoundDatasetDataFrame(self.df)
        else:
            self.dataset = UltrasoundDatasetDir(self.dir)

        if stage == 'fit':
            if self.train_split < 1:
                n = len(self.dataset)
                n_train = int(n * self.train_split)
                n_val = int(n * (1 - self.train_split))

                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.dataset, [n_train, n_val],
                    torch.Generator().manual_seed(SEED))
            else:
                self.train_dataset = self.dataset

    def train_dataloader(self):
        """
        Sets up DataLoader for training set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for training data. Returns None, if training dataset is
            unavailable.
        """
        if self.train_dataset is None:
            print("Training dataset was not set up!")
            return

        return DataLoader(self.train_dataset, **self.train_dataloader_params)

    def val_dataloader(self):
        """
        Sets up DataLoader for validation set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for validation data. Returns None, if validation dataset
            is unavailable.
        """
        if self.val_dataloader is None:
            print("Validation dataset was not set up!")
            return

        return DataLoader(self.val_dataset, **self.val_dataloader_params)

    def test_dataloader(self):
        """
        Sets up DataLoader for test set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for test data. Returns None, if test dataset is
            unavailable.
        """
        raise NotImplementedError


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
        X = read_image(img_path, self.mode)
        X = self.transforms(X)

        # Get metadata from filename
        filename = os.path.basename(img_path)
        filename_parts = filename.split("_")
        patient_id = filename_parts[0]
        us_num = int(filename_parts[-1].replace(".jpg", ""))
        # NOTE: ID naming is used to identify hospital
        hospital = "Stanford" if filename.startswith("SU2") else "SickKids"

        metadata = {"filename": filename, "id": patient_id, "us_num": us_num,
                    "hospital": hospital}

        return X, metadata

    def __len__(self):
        return len(self.paths)


class UltrasoundDatasetDir(UltrasoundDataset):
    """
    Dataset to load images from a directory.
    """
    def __init__(self, dir, img_size=None, mode=1):
        """
        Initialize KidneyDatasetDir object.

        Parameters
        ----------
        dir : str
            Path to flat directory containing ultrasound images.
        img_size : int or tuple of ints, optional
            If int provided, resizes found images to (img_size x img_size), by
            default None.
        mode : int
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 1.
        """
        assert mode in (1, 3)
        self.mode = IMAGE_MODES[mode]

        # Get all images in flat directory
        self.paths = glob.glob(os.path.join(dir, "*"))

        # Define image loading and transforms
        transforms = []
        if img_size:
            transforms.append(T.Resize(img_size))

        self.transforms = T.Compose(transforms)


class UltrasoundDatasetDataFrame(UltrasoundDataset):
    """
    Dataset to load images and labels from a DataFrame.
    """
    def __init__(self, df, dir=None, img_size=None, mode=1):
        """
        Initialize KidneyDatasetDataFrame object.

        Note
        ----
        Expects path column to be "filename", and label column to be "label".

        Parameters
        ----------
        df : pd.DataFrame
            Contains path to images and labels.
        dir : str, optional
            If provided, uses paths in dataframe as relative paths find
            ultrasound images, by default None.
        img_size : int or tuple of ints, optional
            If int provided, resizes found images to (img_size x img_size), by
            default None.
        mode : int
            Number of channels (mode) to read images into (1=grayscale, 3=RGB),
            by default 1.
        """
        assert mode in (1, 3)
        self.mode = IMAGE_MODES[mode]

        # Get paths to images. Add directory to path, if given.
        if dir:
            df["filename"] = df["filename"].map(lambda x: os.path.join(dir, x))
        self.paths = df["filename"].tolist()
        
        # Get labels
        self.labels = df["label"].tolist()

        # Define image loading and transforms
        transforms = []
        if img_size:
            transforms.append(T.Resize(img_size))

        self.transforms = transforms

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
            include path to image and label.
        """
        X, metadata = super()[index]
        metadata["label"] = self.labels[index]

        return X, metadata

