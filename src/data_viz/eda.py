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
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms.v2 as T
from skimage.exposure import equalize_hist
from torchvision.io import read_image, ImageReadMode

# Custom libraries
from src.data import constants
from src.data_prep.moco_dataset import MoCoDataModule
from src.data_prep.utils import load_metadata, load_sickkids_metadata
from src.data_viz import utils as viz_utils


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Map label to encoded integer
CLASS_TO_IDX = {"Sagittal_Left": 0, "Transverse_Left": 1, "Bladder": 2,
                "Transverse_Right": 3, "Sagittal_Right": 4, "Other": 5}
IDX_TO_CLASS = {v: u for u, v in CLASS_TO_IDX.items()}

# SickKids training set patient IDs
SK_TRAIN_IDS = [
    "1001","1002","1004","1005","1008","1009","1012","1019","1020","1032",
    "1038","1039","1041","1047","1050","1055","1059","1066","1069","1070",
    "1075","1076","1077","1078","1081","1089","1092","1093","1100","1104",
    "1105","1107","1110","1113","1115",
]
SK_VAL_IDS = [
    "1003","1010","1011","1021","1022","1035","1044","1045","1048","1056",
    "1065","1087","1088","1098","1099","1103","1114",
]
SK_TEST_IDS = [
    "1001","1002","1004","1005","1008","1009","1012","1019","1020","1032",
    "1038","1039","1041","1047","1050","1055","1059","1066","1069","1070",
    "1075","1076","1077","1078","1081","1089","1092","1093","1100","1104",
    "1105","1107","1110","1113","1115"
]



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


def plot_pixels_along_axis(imgs, ax1=None, ax2=None):
    """
    Plot average pixel intensities across x-axis and y-axis, separately.

    Parameters
    ----------
    imgs : np.array
        Array of grayscale images (B, H, W)
    ax1 : plt.Axes
        Matplotlib axes to plot across x-axis onto
    ax2 : plt.Axes
        Matplotlib axes to plot across x-axis onto
    """
    # Create axes, if not provided
    if ax1 is None or ax2 is None:
        _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # Get height and width
    _, H, W = imgs.shape

    # Plot horizontal intensity distribution
    ax1.bar(
        x=list(range(256)), height=imgs.mean(axis=(0, 1)),
        align="edge",
        width=1,
        color="#a8ddb5",
    )
    ax1.set_ylim(0, 255)
    ax1.set_xlim(0, W)
    ax1.set_ylabel("Pixel Intensity (0-255)")
    ax1.set_xlabel("")
    ax1.set_title("along x-axis")

    # Plot vertical intensity distribution
    ax2.barh(
        y=list(range(256)), width=imgs.mean(axis=(0, 2)),
        align="edge",
        height=1,
        color="#43a2ca",
    )
    ax2.set_xlim(0, 255)
    ax2.set_ylim(0, H)
    ax2.invert_yaxis()
    ax2.set_xlabel("Pixel Intensity (0-255)")
    ax2.set_ylabel("")
    ax2.set_title("along y-axis")


def plot_pixels_histogram_by_label(df_metadata, fname_prefix):
    """
    Plot pixel-wise, width-wise (horizontal) and length-wise (vertical) pixel
    distributions, and save to EDA directory

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    fname_prefix : str
        Filename prefix

    Returns
    -------
    dict
        Contains mean and standard deviation across all pixels
    """
    # Filename prefix
    fname_prefix = fname_prefix + "-" if fname_prefix else ""
    LOGGER.info(f"Now plotting pixel histogram for `{fname_prefix}`...")

    # Filter for data with existing images
    exists_mask = df_metadata["filename"].map(os.path.exists)
    df_metadata = df_metadata[exists_mask]

    # Get image paths
    img_paths = df_metadata["filename"].tolist()

    # Define resize operation
    resize_op = T.Resize(constants.IMG_SIZE)
    normalize_op = T.Normalize([128], [75])

    # Load all images
    imgs = []
    for img_path in img_paths:
        img = read_image(img_path, ImageReadMode.GRAY)
        # Resize image
        img = resize_op(img)
        # Perform histogram equalization
        img = T.functional.equalize(img).to(float)
        # TODO: Consider normalizing by SickKids Train after
        img = normalize_op(img)

        # Normalize between 0 and 1, then multiply by 255
        img = img - img.min()
        img /= img.max()
        img *= 255

        # Remove channel dimension
        img = img.squeeze(dim=0)
        imgs.append(img)
    imgs = torch.stack(imgs)

    # Convert images to numpy
    imgs = imgs.numpy()

    # Store stats
    img_stats = {}
    # Compute pixel-level stats
    img_stats["mean"] = imgs.mean()
    img_stats["std"] = imgs.std()

    # Create figure
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Pixel Intensities", size="x-large")

    # Get unique labels
    labels = sorted(df_metadata["label"].unique())

    # Plot the pixel intensities (row/column) for each label
    subfigs = fig.subfigures(nrows=len(labels))
    for idx, subfig in enumerate(subfigs):
        # Get images corresponding to label
        label = labels[idx]
        mask = (df_metadata["label"] == label)
        curr_imgs = imgs[mask]

        # Create row title
        subfig.suptitle(f"Label: `{label}` | N = {sum(mask)}")

        # Plot pixel intensities across row and across column
        ax1, ax2 = subfig.subplots(nrows=1, ncols=2)
        plot_pixels_along_axis(curr_imgs, ax1, ax2)

    # Ensure save directory exists
    if not os.path.exists(constants.DIR_FIGURES_EDA):
        os.makedirs(constants.DIR_FIGURES_EDA)
    save_path = os.path.join(constants.DIR_FIGURES_EDA, f"{fname_prefix}pixel_intensities.png")

    # Scale figure size based on the number of labels
    fig.set_size_inches(8, 3*len(labels))

    # Save figure
    fig.savefig(save_path)

    return img_stats


def plot_img_histogram_per_hospital():
    """
    Plot image histogram for each hospital
    """
    shared_kwargs = {"prepend_img_dir": True, "extract": True}

    # Store all dataset stats
    dset_stats = {}

    # Load SickKids metadata
    df_sk_metadata = load_metadata("sickkids", **shared_kwargs)

    # 1. SickKids Train Set
    df_sk_metadata_train = df_sk_metadata[df_sk_metadata["id"].isin(SK_TRAIN_IDS)]
    dset_stats["sk_train"] = plot_pixels_histogram_by_label(df_sk_metadata_train, "sickkids_train")

    # 2. SickKids Validation Set
    df_sk_metadata_val = df_sk_metadata[df_sk_metadata["id"].isin(SK_VAL_IDS)]
    dset_stats["sk_val"] = plot_pixels_histogram_by_label(df_sk_metadata_val, "sickkids_val")

    # 3. SickKids Test Set
    df_sk_metadata_test = df_sk_metadata[df_sk_metadata["id"].isin(SK_TEST_IDS)]
    dset_stats["sk_test"] = plot_pixels_histogram_by_label(df_sk_metadata_test, "sickkids_test")

    # 4. Other Test Sets
    for dset in ("stanford", "stanford_non_seq", "sickkids_silent_trial", "uiowa", "chop"):
        df_curr_metadata = load_metadata(dset, **shared_kwargs)
        dset_stats[dset] = plot_pixels_histogram_by_label(df_curr_metadata, dset)

    print(dset_stats)


def patient_imgs_to_gif(df_metadata, patient_idx=0, img_dir=None,
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
    img_dir : str, optional
        Path to directory containing images, by default None
    save_path : str, optional
        Path and filename to save gif as, by default
        constants.DIR_FIGURES+"/predictions/us_patient_{id}_visit_{visit}.gif"
    """
    # Adds directory if specified
    if img_dir:
        df_metadata["filename"] = img_dir + "/" + df_metadata["filename"]

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
        # Make img_dir if not exists
        img_dir = constants.DIR_FIGURES + "/predictions/"
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        save_path = img_dir + f"/us_patient_{patient_id}_visit_{visit}.gif"

    imageio.mimsave(save_path, images, fps=2)


def plot_hn_dist_by_side(df_metadata, title="(Side vs HN)"):
    """
    Creates count plot for number of images with HN vs w/o HN (stratified by
    kidney side)

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image. This must include
        side and HN.
    title : str, optional
        Optional title for plot, by default (Side vs HN)

    Returns
    -------
    matplotlib.axes.Axes
    """
    sns.countplot(
        data=df_metadata[~df_metadata.hn.isna()],
        x="side", hue="hn", order=["Left", "Right"])
    plt.xlabel("Side")
    plt.ylabel("Count")
    plt.title(title)

    return plt.gca()


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
    viz_utils.print_table(view_counts)


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
    viz_utils.print_table(label_seq_counts, show_index=False)


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

    viz_utils.print_table(trans_matrix)

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
    df_metadata = load_metadata("sickkids", prepend_img_dir=True, extract=True)
    dataloader_params = {"batch_size": 9}
    data_module = MoCoDataModule(
        dataloader_params,
        df=df_metadata, img_dir=constants.DIR_IMAGES,
        crop_scale=0.3,
    )

    # Sample 1 batch of images
    for (x_q, x_k), _ in data_module.train_dataloader():
        src_imgs = x_q.numpy()
        augmented_imgs = x_k.numpy()
        break

    # Plot example images
    viz_utils.gridplot_images(
        src_imgs,
        filename="before_ssl_augmentations.png",
        save_dir=constants.DIR_FIGURES_EDA)
    viz_utils.gridplot_images(
        augmented_imgs,
        filename="after_ssl_augmentations.png",
        save_dir=constants.DIR_FIGURES_EDA)


if __name__ == '__main__':
    ############################################################################
    #                         Plot Example Images                              #
    ############################################################################
    plot_ssl_augmentations()

    ############################################################################
    #                        Plot Pixel Histograms                             #
    ############################################################################
    # Plot image histogram
    plot_img_histogram_per_hospital()

    ############################################################################
    #                      Plot Distribution of Views                          #
    ############################################################################
    df_metadata = load_sickkids_metadata(
        extract=True,
        include_unlabeled=True,
        include_test_set=True,
    )

    # Plot distribution of view labels
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
    df_metadata = load_sickkids_metadata(extract=True)
    get_unique_label_sequences(df_metadata)

    ############################################################################
    #                          Transition Matrix                               #
    ############################################################################
    df_metadata = load_sickkids_metadata(extract=True, include_unlabeled=True,
                                img_dir=constants.DIR_IMAGES)
    get_transition_matrix(df_metadata)
