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
from mpl_toolkits.axes_grid1 import ImageGrid
from tabulate import tabulate

# Custom libraries
from src.data import constants
from src.data_prep.dataset import SelfSupervisedUltrasoundDataModule
from src.data_prep.utils import load_metadata


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Map label to encoded integer
CLASS_TO_IDX = {"Saggital_Left": 0, "Transverse_Left": 1, "Bladder": 2,
                "Transverse_Right": 3, "Saggital_Right": 4, "Other": 5}
IDX_TO_CLASS = {v: u for u, v in CLASS_TO_IDX.items()}


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
        Relative positional index of unique patient-visit in df_metadata, by
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


def gridplot_images(imgs, filename, title=None):
    """
    Plot example images on a grid plot

    Parameters
    ----------
    example_imgs : np.array
        Images to visualize
    filename : str
        Path to save figure to
    title : str
        Plot title, by default None
    """
    # Determine number of images to plot
    num_imgs_sqrt = int(np.sqrt(len(imgs)))
    num_imgs = num_imgs_sqrt ** 2

    # Create grid plot
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(num_imgs_sqrt, num_imgs_sqrt),
        axes_pad=0.01,      # padding between axes
    )

    for ax, img_arr in zip(grid, imgs[:num_imgs]):
        # Set x and y axis to be invisible
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        # If first dimension is the channels, move to end
        if img_arr.shape[0] in (1, 3):
            img_arr = np.moveaxis(img_arr, 0, -1)

        # Add image to grid plot
        ax.imshow(img_arr, cmap='gray', vmin=0, vmax=255)

    # Set title
    fig.suptitle(title)

    # Save images
    plt.tight_layout()
    plt.savefig(constants.DIR_FIGURES + "/eda/" + filename)


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
    print_table(view_counts)


def get_unique_label_sequences(df_metadata):
    """
    Prints the unique view progression over each US sequence, where each view
    label is encoded as an integer.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    """
    def _get_label_sequence(df):
        """
        Given a unique US sequence for one patient, get the order of contiguous
        labels in the sequence.

        Parameters
        ----------
        df : pandas.DataFrame
            One full US sequence for one patient.

        Returns
        -------
        list
            Unique label section within the sequence provided
        """
        views = df.sort_values(by=["seq_number"])["label"].tolist()

        # Keep track of order of views
        prev = None
        seq = []

        for view in views:
            if view != prev:
                seq.append(view)
                prev = view

        return seq

    df_metadata = df_metadata.copy()

    # Encode labels as integers
    df_metadata.label = df_metadata.label.map(CLASS_TO_IDX).astype(str)

    # Get unique label sequences per patient
    df_seqs = df_metadata.groupby(by=["id", "visit"])
    label_seqs = df_seqs.apply(_get_label_sequence)
    label_seqs = label_seqs.map(lambda x: "".join(x))
    label_seq_counts = label_seqs.value_counts().reset_index().rename(
        columns={"index": "Label Sequence", 0: "Count"})

    # Print to stdout
    print_table(label_seq_counts, show_index=False)


def get_transition_matrix(df_metadata):
    """
    Print transition matrix

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    """
    def transition_matrix(transitions):
        """
        Given a unique US sequence for one patient, get the transition matrix
        between views encoded as integers.

        Note
        ----
        Adapted from Stack Overflow:
            The following code takes a list such as
            [1,1,2,6,8,5,5,7,8,8,1,1,4,5,5,0,0,0,1,1,4,4,5,1,3,3,4,5,4,1,1]
            with states labeled as successive integers starting with 0
            and returns a transition matrix, M,
            where M[i][j] is the probability of transitioning from i to j

        Link: https://stackoverflow.com/questions/46657221/
              generating-markov-transition-matrix-in-python

        Parameters
        ----------
        transitions : list
            List of transitions with N states, represented by numbers from
            0 to N-1.
        """
        states = list(range(6))
        # Instantiate matrix
        matrix = {i: {j: 0 for j in states} for i in states}

        # Get number of times each transition occurs from state i to j
        for i, j in zip(transitions, transitions[1:]):
            matrix[i][j] += 1

        # Convert counts to probabilities
        for i in matrix.keys():
            # Get number of  transitions from state i
            n = sum(matrix[i].values())
            if n == 0:
                continue

            # Convert each transition count into a probability
            for j in matrix[i]:
                matrix[i][j] = matrix[i][j] / n

        return matrix

    df_metadata = df_metadata.copy()

    # Encode labels as integers
    df_metadata["encoded_label"] = df_metadata.label.map(CLASS_TO_IDX)

    # Get transition matrix per patient
    df_seqs = df_metadata.groupby(by=["id", "visit"])
    all_matrices = df_seqs.apply(
        lambda df: transition_matrix(df.encoded_label.tolist()))

    # Convert transition matrix dicts to numpy arrays
    all_matrices = all_matrices.map(lambda d: pd.DataFrame(d).T.to_numpy())\
        .to_numpy()
    all_matrices = np.stack(all_matrices)

    # Get weighted average of transition matrix across all sequences
    len_per_seq = df_seqs.apply(len).to_numpy()
    weights = len_per_seq / len_per_seq.sum()
    trans_matrix = pd.DataFrame(np.average(all_matrices,
                                           axis=0, weights=weights))

    # Rename columns and rows
    trans_matrix.index = trans_matrix.index.map(IDX_TO_CLASS)
    trans_matrix.columns = trans_matrix.columns.map(IDX_TO_CLASS)

    print_table(trans_matrix)

    return trans_matrix


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
            One full US sequence for one patient.

        Returns
        -------
        bool
            If True, labels are not unidirectional (crosses back).
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


def plot_ssl_augmentations():
    """
    Plot example images of data augmented during self-supervised model training.
    """
    # Instantiate data module
    df_metadata = load_metadata(extract=True)
    data_module = SelfSupervisedUltrasoundDataModule(
        df=df_metadata, dir=constants.DIR_IMAGES)

    # Sample 1 batch of images
    example_imgs = None
    for (x_q, _), _ in data_module.train_dataloader():
        example_imgs = x_q.numpy()
        break

    # Plot example images
    gridplot_images(example_imgs, filename="example_ssl_augmentations.png")


################################################################################
#                               Helper Functions                               #
################################################################################
def print_table(df, show_cols=True, show_index=True):
    """
    Prints table to stdout in a pretty format.

    Parameters
    ----------
    df : pandas.DataFrame
        A table
    show_cols : bool
        If True, prints column names, by default True.
    show_index : bool
        If True, prints row index, by default True.
    """
    print(tabulate(df, tablefmt="psql",
                   headers="keys" if show_cols else None,
                   showindex=show_index))


if __name__ == '__main__':
    ############################################################################
    #                         Plot Example Images                              #
    ############################################################################
    plot_ssl_augmentations()

    ############################################################################
    #                      Plot Distribution of Views                          #
    ############################################################################
    df_metadata = load_metadata(extract=True, include_unlabeled=True,
                                dir=constants.DIR_IMAGES)
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

    ############################################################################
    #                 Print Examples of Label Progression                      #
    ############################################################################
    df_metadata = load_metadata(extract=True)
    get_unique_label_sequences(df_metadata)

    ############################################################################
    #                          Transition Matrix                               #
    ############################################################################
    df_metadata = load_metadata(extract=True, include_unlabeled=True,
                                dir=constants.DIR_IMAGES)
    get_transition_matrix(df_metadata)
