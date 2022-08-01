"""
eda.py

Description: Contains functions to perform exploratory data analysis on dataset.
"""

# Standard libraries
import cv2
import imageio
import logging
import os

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

# Custom libraries
from src.data import constants
from src.data_prep.utils import load_metadata


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
    ax = sns.histplot(df_metadata["label"])
    ax.set(xlabel='View Label', ylabel='Num. of Images')
    plt.xticks(rotation=30)

    return ax


def patient_imgs_to_gif(df_metadata, patient_idx=0, dir=None,
                        save_path=None):
    """
    Saves a patient's US sequence to GIF. Patient is specified by their
    positional index in the metadata table.

    Note
    ----
    If predicted label (pred) is provided in df_metadata, includes predicted
    label in the image.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image, and it may include
        the prediction for each image under "pred".
    patient_idx : int
        Relative positional index of unnique patient-visit in df_metadata, by
        default 0
    dir : str, optional
        Path to directory containing images, by default None
    save_path : str, optional
        Path and filename to save gif as, by default
        constants.DIR_FIGURES+"/predictions/us_patient_{id}_visit_{visit}.gif"
    """
    # Adds directory if specified
    if dir:
        df_metadata["filename"] = dir + "/" + df_metadata["filename"]

    # Get unique patient-visit sequences
    unique_seqs = np.unique(df_metadata[["id", "visit"]].to_numpy(), axis=0)

    # Error checking on input patient_idx
    if patient_idx > len(unique_seqs):
        error_msg = "Chosen patient-visit index exceeds maximum index. " \
                    f"Please choose an index in [0, {len(unique_seqs)})"
        raise RuntimeError(error_msg)

    # Filter for a patient (using patient ID and visit as keys)
    patient_id, visit = unique_seqs[patient_idx]
    id_mask = (df_metadata["id"] == patient_id)
    visit_mask = (df_metadata["visit"] == visit)
    df_metadata = df_metadata[id_mask & visit_mask]

    img_paths = df_metadata["filename"].tolist()
    labels = df_metadata["label"].tolist()
    if "pred" in df_metadata.columns:
        preds = df_metadata["pred"].tolist()
    else:
        preds = None

    # Read each image in the sequence. Add label and prediction, if given
    images = []
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        height, width = img.shape[0], img.shape[1]

        # Add label to image with a background box
        label = labels[i]
        box_height = 45 if preds is None else 90
        cv2.rectangle(img, (0,height-box_height), (width,height), (0,0,0), -1)
        cv2.putText(img=img, text=f"Label: {label}",
                    org=(0, height-int((box_height*0.6))),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255,255,255), thickness=2)

        # Add prediction to image
        if preds is not None:
            pred = preds[i]
            cv2.putText(img=img, text=f"Prediction: {pred}",
                        org=(0, height-15),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1, color=(255,255,255), thickness=2)

        images.append(img)

    if save_path is None:
        # Make dir if not exists
        dir = constants.DIR_FIGURES + "/predictions/"
        if not os.path.exists(dir):
            os.mkdir(dir)

        save_path = dir + f"/us_patient_{patient_id}_visit_{visit}.gif"

    imageio.mimsave(save_path, images, fps=2)


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
        by=["seq_number"]).iloc[i]["label"]).value_counts()
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
        df : pandas.DataFrame
            All sequences for one patient.
        """
        views = df.sort_values(by=["seq_number"])["label"].tolist()

        # Keep track of what views have been seen and previous view
        seen = set()
        prev = None

        for view in views:
            # If not same as last, but was seen previously, then crossed back
            if view != prev and view in seen:
                return True

            seen.add(view)
            prev = view

        return False

    # Group by unique ultrasound sequences
    df_seqs = df_metadata.groupby(by=["id", "visit"])
    
    # Check if any US sequence is not unidirectional
    crossed_back = df_seqs.apply(_crosses_back)

    return not crossed_back.any()


if __name__ == '__main__':
    ############################################################################
    #                      Plot Distribution of Views                          #
    ############################################################################
    df_metadata = load_metadata(extract=True)
    plot_hist_of_view_labels(df_metadata)
    plt.tight_layout()
    plt.show()

    ############################################################################
    #                    Plot US Images with Prediction                        #
    ############################################################################
    df_test_metadata = pd.read_csv(constants.DIR_RESULTS + "/test_set_results(five_view).csv")
    for _ in range(10):
        idx = np.random.randint(0, 84)
        patient_imgs_to_gif(df_test_metadata, patient_idx=idx)
