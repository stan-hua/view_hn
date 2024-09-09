"""
sampler.py

Description: Contains implementations of sampling methods
"""

# Standard libraries
from typing import Callable

# Non-standard libraries
import pandas as pd
import torch
import torch.utils.data
import torchvision


################################################################################
#                                   Classes                                    #
################################################################################
# NOTE: Sourced from https://github.com/ufoym/imbalanced-dataset-sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class OtherDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    OtherDatasetSampler class.

    Note
    ----
    For a dataset with a catch-all "Other"/unlabeled class, load the first half
    of images randomly sampled from the non-"Other" classes, and load the latter
    half of images from the "Other" classes

    e.g., batch_size = 6
     - the first 3 images are labeled A, B, C and the last 3 are labeled "Other"
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        other_label=None,
    ):
        """
        Initialize OthersDatasetBatchSampler object.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset class
        batch_size : int
            Batch size
        shuffle : bool
            If True, shuffle labeled and Other samples before batching
        other_label : int, optional
            Label of "Other" class. If not provided, assumed to be last index
        """
        self.batch_size = batch_size
        self.num_samples = len(dataset)

        # Get (encoded) labels
        df = pd.DataFrame()
        df["label"] = dataset.get_labels(encoded=True)

        # If Other index is not provided, assume it's the last
        if other_label is None:
            other_label = sorted(df["label"].unique().tolist())[-1]

        # If specified, shuffle
        if shuffle:
            df = df.sample(frac=1)

        # Split indices into other and non-other images
        other_mask = (df["label"] == other_label)
        self.label_indices = df[~other_mask].index.to_numpy()
        self.other_indices = df[other_mask].index.to_numpy()

        # Set number of samples based on labeled data
        self.num_samples = len(self.label_indices) // self.batch_size


    def __iter__(self):
        # Compute number of batches of "other" labels
        num_samples_other = len(self.other_indices) // self.batch_size

        # Iterate over batches
        # NOTE: Drops last by default
        for batch_idx in range(self.num_samples):
            # Get labeled samples
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            curr_label_indices = self.label_indices[start_idx:end_idx]

            # Get "Other" samples
            # NOTE: Handle case where there are more Other labeled samples than labeled
            if batch_idx >= num_samples_other:
                curr_idx = batch_idx % num_samples_other
                start_idx = curr_idx * self.batch_size
                end_idx = start_idx + self.batch_size
            other_label_indices = self.other_indices[start_idx:end_idx]
            batch_indices = list(curr_label_indices) + list(other_label_indices)
            yield batch_indices


    def __len__(self):
        return self.num_samples
