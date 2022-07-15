"""
eda.py

Description: Contains functions to perform exploratory data analysis on dataset.
"""

# Standard libraries
import logging

# Non-standard libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate

# Custom libraries
from src.data import constants


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)


################################################################################
#                                    Plots                                     #
################################################################################
def plot_hist_num_images_per_patient(counts):
    """
    Plots histogram of number of images per patient.

    Parameters
    ----------
    counts : list or dict
        Number of images per patient, where each list item represents a unique
        patient. Or dict mapping from patient ID to number of occurences.
    
    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = sns.histplot(counts, binwidth=20, kde=True)
    ax.set(xlabel='# Images in Dataset', ylabel='Num. Patients')

    return ax


def plot_hist_of_view_labels(df_metadata):
    """
    Plot distribution of view labels.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = sns.histplot(df_metadata["view"])
    ax.set(xlabel='View Label', ylabel='Num. of Images')
    plt.xticks(rotation=30)

    return ax


################################################################################
#                                    Tables                                    #
################################################################################
def show_dist_of_ith_view(df_metadata, i=0):
    """
    Prints to the console a summary table of view label distribution of i-th
    view in each sequence.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    i : int, optional
        Index of item in each sequence to check distribution of view label, by
        default 0.
    """
    # Group by unique ultrasound sequences
    df_seqs = df_metadata.groupby(by=["id", "visit"])

    # Get counts for each view at index specified
    view_counts = df_seqs.apply(lambda df: df.sort_values(
        by=["seq_number"]).iloc[i]["view"]).value_counts()
    view_counts = view_counts.to_frame().rename(
        columns={0: f"Num. Seqs. w/ View at index {i}"})

    # Print to console
    print(tabulate(view_counts, headers="keys", tablefmt="psql"))


################################################################################
#                              One-Time Questions                              #
################################################################################
def are_labels_strictly_ordered(df_metadata):
    """
    Check if each sequence has unidirectional labels.

    Note
    ----
    Unidirectional if it follows one of two sequences:
        1. Saggital (R), Transverse (R), Bladder, Transverse (L), Saggital (L)
        2. Saggital (L), Transverse (L), Bladder, Transverse (R), Saggital (R)

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.

    Returns
    -------
    bool
        If True, labels never cross back. Otherwise, False.
    """
    def _crosses_back(df):
        """
        Given a unique US sequence for one patient, checks if the sequence of
        labels is unidirectional.

        Parameters
        ----------
        df : pd.DataFrame
            All sequences for one patient.
        """
        views = df.sort_values(by=["seq_number"])["view"]

        # NOTE: prev_prev_view and prev_view only changee when curr_view changes
        # 'view' label
        prev_prev_view = None
        prev_view = None

        for curr_view in views:
            # First two views are for setting up checks
            if prev_view is None:
                prev_view = curr_view
                continue
            elif prev_prev_view is None:
                prev_prev_view = prev_view
                prev_view = curr_view
                continue
            # Ignore if current view is the same as previous view
            elif curr_view != prev_view:
                continue

            # If current view is equal to 2 views ago, then crossed back
            if curr_view == prev_prev_view:
                return True
            
            # If not, update views
            prev_prev_view = prev_view
            prev_view = curr_view
        
        return False

    # Group by unique ultrasound sequences
    df_seqs = df_metadata.groupby(by=["id", "visit"])
    
    # Check if any US sequence is not unidirectional
    crossed_back = df_seqs.apply(_crosses_back)

    return crossed_back.any()


################################################################################
#                               Helper Functions                               #
################################################################################
def load_metadata(filepath=constants.METADATA_FILE):
    """
    Load metadata table with filenames and view labels.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV metadata file, by default constants.METADATA_FILE

    Returns
    -------
    pandas.DataFrame
        Contains metadata (filename, view label, patient id, visit, sequence
        number).
    """
    df_metadata = pd.read_csv(filepath)
    df_metadata = df_metadata.rename(columns={"IMG_FILE": "filename",
                                              "revised_labels": "view"})
    df_metadata["id"] = df_metadata.filename.map(lambda x: int(x.split("_")[0]))
    df_metadata["visit"] = df_metadata.filename.map(
        lambda x: int(x.split("_")[1]))
    df_metadata["seq_number"] = df_metadata.filename.map(
        lambda x: int(x.split("_")[2].replace(".jpg", "")))
    
    return df_metadata


if __name__ == '__main__':
    df_metadata = load_metadata()
    plot_hist_of_view_labels(df_metadata)
    plt.tight_layout()
    plt.show()
