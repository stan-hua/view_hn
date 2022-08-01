"""
utils.py

Description: Contains helper functions for a variety of data/label preprocessing
             functions.
"""

# Non-standard libraries
import numpy as np
import pandas as pd

# Custom libraries
from src.data import constants

################################################################################
#                                  Functions                                   #
################################################################################
def load_metadata(path=constants.METADATA_FILE, extract=False):
    """
    Load metadata table with filenames and view labels.

    Parameters
    ----------
    path : str, optional
        Path to CSV metadata file, by default constants.METADATA_FILE
    extract : bool, optional
        If True, extracts patient ID, US visit, and sequence number from the
        filename, by default False.

    Returns
    -------
    pandas.DataFrame
        May contain metadata (filename, view label, patient id, visit, sequence
        number).
    """
    df_metadata = pd.read_csv(path)
    df_metadata = df_metadata.rename(columns={"IMG_FILE": "filename",
                                              "revised_labels": "label"})

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
    df_metadata["id"] = df_metadata[col].map(
            lambda x: int(x.split("_")[0]))
    df_metadata["visit"] = df_metadata[col].map(
        lambda x: int(x.split("_")[1]))
    df_metadata["seq_number"] = df_metadata[col].map(
        lambda x: int(x.split("_")[2].replace(".jpg", "")))


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
