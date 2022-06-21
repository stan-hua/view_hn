"""
dataloaders.py

Description: Contains functions/classes to load dataset in PyTorch.
"""
# Standard libraries
from abc import abstractmethod
import glob
import os

# Non-standard libraries
import torch
import torchvision.transforms as T


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
#                                   Classes                                    #
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
        X = self.transforms(img_path)

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
    def __init__(self, dir, img_size=None):
        """
        Initialize KidneyDatasetDir object.

        Parameters
        ----------
        dir : str
            Path to flat directory containing ultrasound images.
        img_size : int or tuple of ints, optional
            If int provided, resizes found images to (img_size x img_size), by
            default None.
        """
        # Get all images in flat directory
        self.paths = glob.glob(os.path.join(dir, "*"))

        # Define image loading and transforms
        transforms = [T.ToPILImage()]
        if img_size:
            transforms.append(T.Resize(img_size))
        transforms.append(T.ToTensor())

        self.transforms = transforms


class UltrasoundDatasetDataFrame(UltrasoundDataset):
    """
    Dataset to load images and labels from a DataFrame.
    """
    def __init__(self, df, dir=None, img_size=None):
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
        """
        # Get paths to images. Add directory to path, if given.
        if dir:
            df["filename"] = df["filename"].map(lambda x: os.path.join(dir, x))
        self.paths = df["filename"].tolist()
        
        # Get labels
        self.labels = df["label"].tolist()

        # Define image loading and transforms
        transforms = [T.ToPILImage()]
        if img_size:
            transforms.append(T.Resize(img_size))
        transforms.append(T.ToTensor())

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
            include path to image, patient ID, and hospital.
        """
        X, metadata = super()[index]
        metadata["view"] = self.labels[index]

        return X, metadata

