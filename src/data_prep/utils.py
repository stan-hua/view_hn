"""
utils.py

Description: Contains helper functions for a variety of data/label preprocessing
             functions.
"""

# Standard libraries
import glob
import os

# Non-standard libraries
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Custom libraries
from src.data import constants


################################################################################
#                               Metadata Related                               #
################################################################################
def load_metadata(path=constants.METADATA_FILE, extract=False,
                  include_unlabeled=False, dir=None):
    """
    Load metadata table with filenames and view labels.

    Note
    ----
    If <include_unlabeled> specified, <dir> must be provided.

    Parameters
    ----------
    path : str, optional
        Path to CSV metadata file, by default constants.METADATA_FILE
    extract : bool, optional
        If True, extracts patient ID, US visit, and sequence number from the
        filename, by default False.
    include_unlabeled : bool, optional
        If True, include all unlabeled images in <dir>, by default False.
    dir : str, optional
        Directory containing unlabeled (and labeled) images.

    Returns
    -------
    pandas.DataFrame
        May contain metadata (filename, view label, patient id, visit, sequence
        number).
    """
    df_metadata = pd.read_csv(path)
    df_metadata = df_metadata.rename(columns={"IMG_FILE": "filename",
                                              "revised_labels": "label"})

    # If specified, include unlabeled images in directory provided
    if include_unlabeled:
        assert dir is not None, "Please provide `dir` as an argument!"

        # Get all image paths
        all_img_paths = glob.glob(os.path.join(dir, "*"))
        df_others = pd.DataFrame({"filename": all_img_paths})
        df_others.filename = df_others.filename.map(os.path.basename)
        
        # Remove found paths to already labeled images
        labeled_img_paths = set(df_metadata.filename.tolist())
        df_others = df_others[~df_others.filename.isin(labeled_img_paths)]

        # Remove external data
        df_others = df_others[~df_others.filename.str.startswith("SU2")]

        # NOTE: Unlabeled images have label "Other"
        df_others["label"] = "Other"
        
        # Merge labeled and unlabeled data
        df_metadata = pd.concat([df_metadata, df_others], ignore_index=True)

    if extract:
        extract_data_from_filename(df_metadata)

    return df_metadata


def extract_data_from_filename(df_metadata, col="filename"):
    """
    Extract metadata from each image's filename. Assign columns in-place.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    col : str, optional
        Name of column containing filename, by default "filename"
    """
    df_metadata["basename"] = df_metadata[col].map(os.path.basename)
    df_metadata["id"] = df_metadata["basename"].map(
            lambda x: x.split("_")[0])
    df_metadata["visit"] = df_metadata["basename"].map(
        lambda x: x.split("_")[1])
    df_metadata["seq_number"] = df_metadata["basename"].map(
        lambda x: int(x.split("_")[2].replace(".jpg", "")))
    df_metadata = df_metadata.drop(columns=["basename"])


def get_from_paths(paths, item="id"):
    """
    Extract metadata from a list of paths.

    Parameters
    ----------
    paths : list
        List of image paths
    item : str
        Desired item to extract. One of "id" (patient ID), "visit" or
        "seq_number".

    Returns
    -------
    np.array
        List of extracted metadata strings
    """
    assert item in ("id", "visit", "seq_number")

    patient_ids = []
    for path in paths:
        patient_ids.append(get_from_path(path, item))

    return np.array(patient_ids)


def get_from_path(path, item="id"):
    """
    Extract metadata from an image path.

    Parameters
    ----------
    path : str
        Image path
    item : str
        Desired item to extract. One of "id" (patient ID), "visit" or
        "seq_number".

    Returns
    -------
    str
        Extracted metadata strings
    """
    assert item in ("id", "visit", "seq_number")

    item_to_idx = {
        "id": 0,
        "visit": 1,
        "seq_number": 2
    }
    idx = item_to_idx[item]

    extracted = os.path.basename(path).split("_")[idx]

    # If seq number, convert to integer
    if item == "seq_number":
        extracted = int(extracted.split(".")[0])

    return extracted


def remove_only_unlabeled_seqs(df_metadata):
    """
    Removes rows from fully unlabeled ultrasound sequences.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image. May contain
        unlabeled images (label as "Other")

    Returns
    -------
    pandas.DataFrame
        Metadata dataframe with completely unlabeled sequences removed
    """
    # If including unlabeled, exclude sequences that are all "Others"
    # NOTE: This leads to the same patient-visit splits as in training
    mask = df_metadata.groupby(by=["id", "visit"]).apply(
        lambda df: not all(df.label == "Other"))
    mask = mask[mask]
    mask.name = "bool_mask"

    # Join to filter 
    df_metadata = df_metadata.set_index(["id", "visit"])
    df_metadata = df_metadata.join(mask, how="inner")

    # Remove added column
    df_metadata = df_metadata.reset_index()
    df_metadata = df_metadata.drop(columns=["bool_mask"])

    return df_metadata


################################################################################
#                                Data Splitting                                #
################################################################################
def split_by_ids(patient_ids, train_split=0.8, seed=constants.SEED):
    """
    Splits list of patient IDs into training and val/test set.

    Note
    ----
    Split may not be exactly equal to desired train_split due to imbalance of
    patients.

    Parameters
    ----------
    patient_ids : np.array or array-like
        List of patient IDs (IDs can repeat).
    train_split : float, optional
        Proportion of total data to leave for training, by default 0.8
    seed : int, optional
        If provided, sets random seed to value, by default constants.SEED

    Returns
    -------
    tuple of (np.array, np.array)
        Contains (train_indices, val_indices), which are arrays of indices into
        patient_ids to specify which are used for training or validation/test.
    """
    # Get expected # of items in training set
    n = len(patient_ids)
    n_train = int(n * train_split)

    # Add soft lower/upper bounds (5%) to expected number. 
    # NOTE: Assume it won't become negative
    n_train_min = int(n_train - (n * 0.05))
    n_train_max = int(n_train + (n * 0.05))

    # Create mapping of patient ID to number of occurrences
    id_to_len = {}
    for _id in patient_ids:
        if _id not in id_to_len:
            id_to_len[_id] = 0
        id_to_len[_id] += 1

    # Shuffle unique patient IDs
    unique_ids = list(id_to_len.keys())
    shuffled_unique_ids = shuffle(unique_ids, random_state=seed)

    # Randomly choose patients to add to training set until full
    train_ids = set()
    n_train_curr = 0
    for _id in shuffled_unique_ids:
        # Add patient if number of training samples doesn't exceed upper bound
        if n_train_curr + id_to_len[_id] <= n_train_max:
            train_ids.add(_id)
            n_train_curr += id_to_len[_id]

        # Stop when there is roughly enough in the training set 
        if n_train_curr >= n_train_min:
            break

    # Create indices
    train_idx = []
    val_idx = []
    for idx, _id in enumerate(patient_ids):
        if _id in train_ids:
            train_idx.append(idx)
        else:
            val_idx.append(idx)

    # Convert to arrays
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    return train_idx, val_idx


def cross_validation_by_patient(patient_ids, num_folds=5):
    """
    Create train/val indices for Cross-Validation with exclusive patient ids
    betwen training and validation sets.

    Parameters
    ----------
    patient_ids : np.array or array-like
        List of patient IDs (IDs can repeat).
    num_folds : int
        Number of folds for cross-validation

    Returns
    -------
    list of <num_folds> tuples of (np.array, np.array)
        Each tuple in the list corresponds to a fold's (train_ids, val_ids)
    """
    folds = []

    training_folds = []
    remaining_ids = patient_ids

    # Get patient IDs for training set in each folds
    while num_folds > 1:
        proportion = 1 / num_folds
        train_idx, rest_idx = split_by_ids(remaining_ids, proportion)

        training_folds.append(np.unique(remaining_ids[train_idx]))
        remaining_ids = remaining_ids[rest_idx]
        
        num_folds -= 1

    # The last remaining IDs are the patient IDs of the last fold
    training_folds.append(np.unique(remaining_ids))

    # Create folds
    fold_idx = list(range(len(training_folds)))
    for i in fold_idx:
        # Get training set indices
        uniq_train_ids = set(training_folds[i])
        train_idx = np.where([_id in uniq_train_ids for _id in patient_ids])[0]

        # Get validation set indices
        val_indices = fold_idx.copy()
        val_indices.remove(i)
        val_patient_ids = np.concatenate(
            np.array(training_folds, dtype=object)[val_indices])
        uniq_val_ids = set(val_patient_ids)
        val_idx = np.where([_id in uniq_val_ids for _id in patient_ids])[0]

        folds.append((train_idx, val_idx))

    return folds


################################################################################
#                                Miscellaneous                                 #
################################################################################
# TODO: Finish implementing this
def disc_to_cont(lst):
    """
    Given an ordered list of discrete integers, convert to a continuous
    increasing list of floats. Assumes uniformly distributed within and across
    boundaries.

    Parameters
    ----------
    lst : list or array-like
        List of N ordered discrete integers with no gaps between integers
    
    Returns
    -------
    np.array
        Array of N continuous decimal numbers
    """
    raise NotImplementedError()

    assert len(lst) > 3

    # Convert to array if not already
    if not isinstance(lst, np.array):
        lst = np.array(lst)

    # Get mapping of discrete value to count
    uniq_vals = sorted(np.unique(lst))
    count = {}
    for discrete_val in uniq_vals:
        count[discrete_val] = (lst == discrete_val).sum()

    # Interpolate values between boundaries
    cont_vals = []

    for i in range(0, len(uniq_vals) -1):
        curr_val = uniq_vals[i]
        next_val = uniq_vals[i + 1]
        cont_vals.append(np.linspace(curr_val, next_val, count[curr_val] + 1)[:-1])
