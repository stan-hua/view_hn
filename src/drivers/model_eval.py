"""
model_eval.py

Description: Used to evaluate a trained model's performance on the testing set.
"""

# Standard libraries
import logging
import os
import random
from colorama import Fore, Style

# Non-standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as T
import yaml
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

# Custom libraries
from src.data import constants
from src.data_prep.dataset import UltrasoundDataModule
from src.data_prep import utils
from src.data_viz.eda import print_table
from src.models.efficientnet_pl import EfficientNetPL
from src.models.efficientnet_lstm_pl import EfficientNetLSTM
from src.models.linear_classifier import LinearClassifier
from src.models.linear_lstm import LinearLSTM


# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure seaborn color palette
sns.set_palette("Paired")

# Add progress_apply to pandas
tqdm.pandas()

################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Flag to use GPU or not
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Map label to encoded integer (for visualization)
CLASS_TO_IDX = {"Saggital_Left": 0, "Transverse_Left": 1, "Bladder": 2,
                "Transverse_Right": 3, "Saggital_Right": 4, "Other": 5}
IDX_TO_CLASS = {v: u for u, v in CLASS_TO_IDX.items()}

# Type of models
MODEL_TYPES = ("five_view", "binary", "five_view_seq", "five_view_seq_w_other",
               "five_view_seq_relative",
               "five_view_moco", "five_view_moco_seq")

################################################################################
#                               Paths Constants                                #
################################################################################
# Model type to checkpoint file
MODEL_TYPE_TO_CKPT = {
    "five_view": "/five_view/0/epoch=6-step=1392.ckpt",
    "five_view_seq": "cnn_lstm_8/0/epoch=31-step=5023.ckpt",
    "five_view_seq_w_other": "/five_view/0/epoch=6-step=1392.ckpt",
    "five_view_seq_relative": "relative_side_grid_search(2022-09-21)/relative_side(2022-09-20_22-09)/0/epoch=7-step=1255.ckpt",
    "binary" : "/binary_classifier/0/epoch=12-step=2586.ckpt",
    # Checkpoint of MoCo Linear Classifier
    "five_view_moco": "moco_linear_eval_4/0/last.ckpt",
    # Checkpoint of MoCo LinearLSTM Classifier
    "five_view_moco_seq": "moco_linear_lstm_eval_0/0/epoch=12-step=129.ckpt",
}

# Table to store/retrieve predictions and labels for test set
TEST_PRED_PATH = constants.DIR_RESULTS + "/test_set_results(%s).csv"


################################################################################
#                             Inference - Related                              #
################################################################################
def get_test_set_metadata(df_metadata, hparams, img_dir=constants.DIR_IMAGES):
    """
    Get metadata table containing (filename, label) for each image in the test
    set.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    hparams : dict
        Hyperparameters used in model training run, used to load exact test set.
    img_dir : str, optional
        Path to directory containing metadata, by default constants.DIR_IMAGES

    Returns
    -------
    pandas.DataFrame
        Metadata of each image in the test set
    """
    # NOTE: If data splitting parameters are not given, assume defaults
    for split_params in ("train_val_split", "train_test_split"):
        if split_params not in hparams:
            hparams[split_params] = 0.75

    # Set up data
    dm = UltrasoundDataModule(df=df_metadata, img_dir=img_dir, **hparams)
    dm.setup()

    # Get filename and label of test set data from data module
    df_test = pd.DataFrame({
        "filename": dm.dset_to_paths["test"],
        "label": dm.dset_to_labels["test"]})

    # Extract other metadata from filename
    df_test["orig_filename"] = df_test.filename.map(lambda x: x.split("\\")[-1])
    utils.extract_data_from_filename(df_test, col="orig_filename")
    df_test = df_test.drop(columns="orig_filename")

    return df_test


def predict_on_images(model, filenames, img_dir=constants.DIR_IMAGES):
    """
    Performs inference on images specified. Returns predictions, probabilities
    and raw model output.

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch model.
    filenames : np.array or array-like
        Filenames (or full paths) to images to infer on.
    img_dir : str, optional
        Path to directory containing images, by default constants.DIR_IMAGES

    Returns
    -------
    tuple of np.array
        (prediction for each image, probability of each prediction,
         raw model output)
    """
    # Set to evaluation mode
    model.eval()

    # Predict on each images one-by-one
    with torch.no_grad():
        preds = []
        probs = []
        outs = []

        for filename in tqdm(filenames):
            img_path = filename if img_dir is None else f"{img_dir}/{filename}"

            # Load image as expected by model
            img = read_image(img_path, ImageReadMode.RGB)
            img = img / 255.
            img = transform_image(img)
            img = np.expand_dims(img, axis=0)

            # Convert to tensor and send to device
            img = torch.FloatTensor(img).to(DEVICE)

            # Perform inference
            out = model(img)
            pred = torch.argmax(out, dim=1)
            pred = int(pred.detach().cpu())

            # Get probability
            prob = torch.nn.functional.softmax(out, dim=1)
            prob = prob.detach().cpu().numpy().max()
            probs.append(prob)

            # Get maximum activation
            out = float(out.max().detach().cpu())
            outs.append(out)

            # Convert from encoded label to label name
            pred_label = constants.IDX_TO_CLASS[pred]
            preds.append(pred_label)

    return np.array(preds), np.array(probs), np.array(outs)


def predict_on_sequences(model, filenames, img_dir=constants.DIR_IMAGES):
    """
    Performs inference on a full ultrasound sequence specified. Returns
    predictions, probabilities and raw model output.

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch model.
    filenames : np.array or array-like
        Filenames (or full paths) to images from one unique sequence to infer.
    img_dir : str, optional
        Path to directory containing images, by default constants.DIR_IMAGES

    Returns
    -------
    tuple of np.array
        (prediction for each image, probability of each prediction,
         raw model output)
    """
    # Set to evaluation mode
    model.eval()

    # Predict on each images one-by-one
    with torch.no_grad():
        imgs = []
        for filename in filenames:
            img_path = filename if img_dir is None else f"{img_dir}/{filename}"

            # Load image as expected by model
            img = read_image(img_path, ImageReadMode.RGB)
            img = img / 255.
            img = transform_image(img)
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)

        # Convert to tensor and send to device
        imgs = torch.FloatTensor(imgs).to(DEVICE)

        # Perform inference
        outs = model(imgs)
        outs = outs.detach().cpu()
        preds = torch.argmax(outs, dim=1).numpy()

        # Get probability
        probs = torch.nn.functional.softmax(outs, dim=1)
        probs = torch.max(probs, dim=1).values.numpy()

        # Get maximum activation
        outs = torch.max(outs, dim=1).values.numpy()

        # Convert from encoded label to label name
        preds = np.vectorize(constants.IDX_TO_CLASS.__getitem__)(preds)

    return preds, probs, outs


################################################################################
#                            Analysis of Inference                             #
################################################################################
def plot_confusion_matrix(df_pred, filter_confident=False, relative_side=False):
    """
    Plot confusion matrix based on model predictions.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    filter_confident : bool, optional
        If True, filters for most confident prediction in each view label for
        each unique sequence, before creating the confusion matrix.
    relative_side : bool, optional
        If True, assumes predicted labels are encoded for relative sides, by
        default False.
    """
    # If flagged, get for most confident pred. for each view label per seq.
    if filter_confident:
        df_pred = filter_most_confident(df_pred)

    # Gets all labels based on side encoding
    all_labels = constants.CLASSES["relative" if relative_side else ""]

    # Plots confusion matrix
    cm = confusion_matrix(df_pred["label"], df_pred["pred"],
                          labels=all_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=all_labels)
    disp.plot()
    plt.tight_layout()
    plt.show()


def plot_pred_probability_by_views(df_pred, relative_side=False):
    """
    Plot average probability of prediction per view label.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    relative_side : bool, optional
        If True, assumes predicted labels are encoded for relative sides, by
        default False.
    """
    df_pred = df_pred.copy()

    # Filter for correct test predictions
    df_pred.binary_label = df_pred.label.map(
        lambda x: "Bladder" if x == "Bladder" else "Other")
    df_pred = df_pred[df_pred.binary_label == df_pred.pred]

    # Average prediction probability over each view
    df_prob_by_view = df_pred.groupby(by="label").mean()["prob"].reset_index()
    df_prob_by_view = df_prob_by_view.rename(columns=
        {"label": "View", "prob": "Probability"})
    
    # Gets all labels based on side encoding
    all_labels = constants.CLASSES["relative" if relative_side else ""]
    # Bar plot
    sns.barplot(data=df_prob_by_view, x="View", y="Probability",
                order=all_labels)
    plt.show()


def check_misclassifications(df_pred, filter=True, local=True,
                             relative_side=False):
    """
    Given the most confident test set predictions, determine the percentage of
    misclassifications that are due to:
        1. Swapping sides (e.g., Saggital Right mistaken for Saggital Left)
        2. Adjacent views

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    filter : bool, optional
        If True, filters for most confident prediction in consecutive label
        groups or for each of the 5 main views per sequence, by default True.
    local : bool, optional
        If True, gets the most confident prediction within each group of
        consecutive images with the same label. Otherwise, aggregates by view
        label to find the most confident view label predictions,
        by default True.
    relative_side : bool, optional
        If True, assumes predicted labels are encoded for relative sides, by
        default False.
    """
    # Filter for most confident predictions for each expected view label
    if filter:
        df_pred = filter_most_confident(df_pred, local=local)

    # Get misclassified instances
    df_misclassified = df_pred[(df_pred.label != df_pred.pred)]

    # Get label adjacency dict based on side label
    label_adjacency_dict = constants.LABEL_ADJACENCY["relative" if \
        relative_side else ""]

    # 1. Proportion and count of misclassification from adjacent views
    prop_adjacent = df_misclassified.apply(
        lambda row: row.pred in label_adjacency_dict[row.label],
        axis=1).mean()
    num_adjacent = df_misclassified.apply(
        lambda row: row.pred in label_adjacency_dict[row.label],
        axis=1).sum()

    # 2. Proportion and count of misclassification from wrong body side
    # Filter out Bladder images (so remains are Sagittal/Transverse labels/pred)
    df_swapped = df_misclassified[(df_misclassified.label != "Bladder")]
    df_swapped = df_swapped[(df_swapped.pred != "Bladder")]
    prop_swapped = df_swapped.apply(
        lambda row: row.label.split("_")[0] == row.pred.split("_")[0],
        axis=1).mean()
    num_swapped = df_swapped.apply(
        lambda row: row.label.split("_")[0] == row.pred.split("_")[0],
        axis=1).sum()

    # Format output
    df_results = pd.Series({
        "Wrong Side": prop_swapped,
        "Adjacent Label": prop_adjacent,
        "Other": 1 - (prop_swapped + prop_adjacent)
    })

    df_results = pd.DataFrame({
        "Proportion": [prop_swapped, prop_adjacent,
                       1-(prop_swapped+prop_adjacent)],
        "Count": [num_swapped, num_adjacent,
                  len(df_misclassified) - num_swapped - num_adjacent]
    }, index=["Wrong Side", "Adjacent Label", "Other"])

    # Print to command line
    print_table(df_results)


def plot_prob_over_sequence(df_pred, correct_only=False, update_seq_num=False):
    """
    Plots average probability of prediction over each number in the sequence.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    correct_only : bool, optional
        If True, filters for correctly predicted samples before plotting, by
        default False.
    update_seq_num : bool, optional
        If True, accounts for missing images between sequence numbers by
        creating new (raw) sequence number based on existing images, by default
        False.
    """
    col = "seq_number"

    # If specified, create new sequence numbers
    if update_seq_num:
        df_pred = get_new_seq_numbers(df_pred)
        col = "seq_number_new"

    # Filter correct if specified
    if correct_only:
        df_pred = df_pred[df_pred.label == df_pred.pred]

    # Create plot
    df = df_pred.groupby(by=[col])["prob"].mean().reset_index()
    sns.barplot(data=df, x=col, y="prob")
    plt.xlabel("Number in the US Sequence")
    plt.ylabel("Prediction Probability")
    plt.xlim(0, max(df_pred[col])+1)
    plt.xticks(np.arange(0, max(df[col])+1, 5.0))
    plt.tight_layout()
    plt.show()


def plot_acc_over_sequence(df_pred, update_seq_num=False):
    """
    Plots accuracy of predictions over each number in the sequence.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    update_seq_num : bool, optional
        If True, accounts for missing images between sequence numbers by
        creating new (raw) sequence number based on existing images, by default
        False.
    """
    col = "seq_number"

    # If specified, create new sequence numbers
    if update_seq_num:
        df_pred = get_new_seq_numbers(df_pred)
        col = "seq_number_new"

    # Create plot
    df = df_pred.groupby(by=[col]).apply(
        lambda df: (df.pred == df.label).mean())
    df.name = "accuracy"
    df = df.reset_index()
    sns.barplot(data=df, x=col, y="accuracy")
    plt.xlabel("Number in the US Sequence")
    plt.ylabel("Accuracy")
    plt.xlim(0, max(df_pred[col])+1)
    plt.xticks(np.arange(0, max(df[col])+1, 5))
    plt.tight_layout()
    plt.show()


def plot_image_count_over_sequence(df_pred, update_seq_num=False):
    """
    Plots number of imgaes for each number in the sequence.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    """
    col = "seq_number"

    # If specified, create new sequence numbers
    if update_seq_num:
        df_pred = get_new_seq_numbers(df_pred)
        col = "seq_number_new"

    # Create plot
    df_count = df_pred.groupby(by=[col]).apply(lambda df: len(df))
    df_count.name = "count"
    df_count = df_count.reset_index()
    sns.barplot(data=df_count, x=col, y="count")
    plt.xlabel("Number in the US Sequence")
    plt.ylabel("Number of Images")
    plt.xlim(0, max(df_pred[col])+1)
    plt.xticks(np.arange(0, max(df_pred[col])+1, 5))
    plt.tight_layout()
    plt.show()


def check_others_pred_progression(df_pred):
    """
    Given test set predictions for full sequences that include "Other" labels,
    show (predicted) label progression for unlabeled images among real labels.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    """
    def _get_label_sequence(df):
        """
        Given a unique US sequence for one patient, get the order of contiguous
        labels in the sequence, where there is blocks of unlabeled image, show
        progression of predicted labels.

        Parameters
        ----------
        df : pandas.DataFrame
            One full US sequence for one patient.

        Returns
        -------
        list
            Unique label section within the sequence provided
        """
        label_views = df.sort_values(by=["seq_number"])["label"].tolist()
        pred_views = df.sort_values(by=["seq_number"])["pred"].tolist()

        # Encoded label for "Other"
        other_label = str(CLASS_TO_IDX["Other"])
        
        # Keep track of order of views
        prev_label = None
        prev_pred = None
        seq = []

        for i, view in enumerate(label_views):
            # If not 'Other', show regular label progression
            if view != other_label and view != prev_label:
                seq.append(view)
                prev_pred = None
            # If 'Other', show prediction progression until out of 'Other'
            elif view == other_label and pred_views[i] != prev_pred:
                prev_pred = pred_views[i]

                # Color predicted view, so it's distinguishable in stdout
                seq.append(f"{Fore.MAGENTA}{prev_pred}{Style.RESET_ALL}")

            prev_label = view
        return seq

    df_pred = df_pred.copy()

    # Encode labels as integers
    df_pred.label = df_pred.label.map(CLASS_TO_IDX).astype(str)
    df_pred.pred = df_pred.pred.map(CLASS_TO_IDX).astype(str)

    # Get unique label sequences per patient
    df_seqs = df_pred.groupby(by=["id", "visit"])
    label_seqs = df_seqs.apply(_get_label_sequence)
    label_seqs = label_seqs.map(lambda x: "".join(x))
    label_seq_counts = label_seqs.value_counts().reset_index().rename(
        columns={"index": "Label Sequence", 0: "Count"})

    # Print to stdout
    print_table(label_seq_counts, show_index=False)


def check_rel_side_pred_progression(df_pred):
    """
    Given test set predictions for full sequences where side labels are relative
    to the first side, show label/predicted label progression.

    Note
    ----
    Highlight locations where error in toggling between first/second occurs.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    """
    def _get_label_sequence(df):
        """
        Given a unique US sequence for one patient, get the order of contiguous
        correct/incorrect relative side labels in the sequence.

        Parameters
        ----------
        df : pandas.DataFrame
            One full US sequence for one patient.

        Returns
        -------
        list
            Unique label section within the sequence provided
        """
        # Get true/predicted (relative) side label
        side_labels = df.sort_values(by=["seq_number"])["label"].tolist()
        side_preds = df.sort_values(by=["seq_number"])["pred"].tolist()
        
        # Keep track of order of views
        prev_label = None
        prev_pred = None
        seq = []

        for i, side_label in enumerate(side_labels):
            # Get prediction
            side_pred = side_preds[i]

            # If side changed
            if side_label != prev_label:
                if side_pred == side_label:
                    # If side changed + correct pred., color green
                    seq.append(f"{Fore.GREEN}{side_label}{Style.RESET_ALL}")
                    # Reset prev. prediction (in case new error pops up)
                    prev_pred = None
                elif side_pred != prev_pred:
                    # If side changed + new incorrect pred., color yellow
                    prev_pred = side_pred
                    seq.append(f"{Fore.YELLOW}{prev_pred}{Style.RESET_ALL}")
            else:
                if side_pred not in (side_label, prev_pred):
                    # If side same + new incorrect pred., color red
                    prev_pred = side_pred
                    seq.append(f"{Fore.MAGENTA}{prev_pred}{Style.RESET_ALL}")

            prev_label = side_label
        return seq

    df_pred = df_pred.copy()

    # Extract and encode relative side labels
    side_to_idx = {"First": "1", "Second": "2", "None": "-",}
    df_pred.label = df_pred.label.map(
        lambda x: side_to_idx[utils.extract_from_label(x, "side")]).astype(str)
    df_pred.pred = df_pred.pred.map(
        lambda x: side_to_idx[utils.extract_from_label(x, "side")]).astype(str)

    # Get unique label sequences per patient
    df_seqs = df_pred.groupby(by=["id", "visit"])
    label_seqs = df_seqs.apply(_get_label_sequence)
    label_seqs = label_seqs.map(lambda x: "".join(x))
    label_seq_counts = label_seqs.value_counts().reset_index().rename(
        columns={"index": "Label Sequence", 0: "Count"})

    # Print to stdout
    print_table(label_seq_counts, show_index=False)


def print_confusion_matrix(df_pred, unique_labels=constants.CLASSES[""]):
    """
    Prints confusion matrix with proportion and count

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    unique_labels : list, optional
        List of unique labels, by default constants.CLASSES[""]
    """
    def get_counts_and_prop(df):
        """
        Given all instances for one label, get the proportions and counts for
        each of the predicted labels.

        Parameters
        ----------
        df : pandas.DataFrame
            Test set prediction for 1 label.
        """
        df_counts = df["pred"].value_counts()
        num_samples = len(df)

        return df_counts.map(lambda x: f"{round(x/num_samples, 2)} ({x})")

    # Check that test labels include all labels
    assert set(df_pred["label"].unique()) == set(unique_labels), \
        "Not all given labels are present!"

    # Accumulate counts and proportion of predicted labels for each label
    df_labels = df_pred.groupby(by=["label"]).apply(get_counts_and_prop)

    # Rename columns
    df_labels = df_labels.reset_index().rename(
        columns={"pred": "proportion", "level_1": "pred"})

    # Reformat table to become a confusion matrix
    df_cm = df_labels.pivot(index="label", columns="pred", values="proportion")

    # Remove axis names
    df_cm = df_cm.rename_axis(None).rename_axis(None, axis=1)

    # Reorder column and index by given labels
    df_cm = df_cm.loc[:, unique_labels].reindex(unique_labels)

    print_table(df_cm)


################################################################################
#                               Helper Functions                               #
################################################################################
def transform_image(img):
    """
    Transforms image, as done during training.

    Note
    ----
    Transforms include: 
        - Resize the image to (256, 256)

    Parameters
    ----------
    img : np.array
        An image

    Returns
    -------
    np.array
        Transformed image
    """
    transforms = []
    transforms.append(T.Resize(constants.IMG_SIZE))
    transforms = T.Compose(transforms)

    return transforms(img)


def get_hyperparameters(hparam_dir=None, filename="hparams.yaml"):
    """
    Load hyperparameters from model training directory. If not provided, return
    default hyperparameters.

    Parameters
    ----------
    hparam_dir : str
        Path to model training directory containing hyperparameters.
    filename : str
        Filename of YAML file with hyperparameters, by default "hparams.yaml"

    Returns
    -------
    dict
        Hyperparameters
    """
    if hparam_dir:
        with open(os.path.join(hparam_dir, filename), "r") as stream:
            try:
                hparams = yaml.full_load(stream)
                return hparams
            except yaml.YAMLError as exc:
                LOGGER.critical(exc)
                LOGGER.critical("Using default hyperparameters...")

    # If above does not succeed, use default hyperparameters
    hparams = {
        "img_size": constants.IMG_SIZE,
        "train": True,
        "test": True,
        "train_test_split": 0.75,
        "train_val_split": 0.75
    }

    return hparams


def get_local_groups(values):
    """
    Identify local groups of consecutive labels/predictions in a list of values.
    Return a list of increasing integers which identify these groups.

    Note
    ----
    Input of [1, 1, 2, 1, 1] would return [1, 1, 2, 3, 3]. Groupings would be:
        - Group 1: (1, 1)
        - Group 2: (2)
        - Group 3: (1, 1)
    

    Parameters
    ----------
    values: list
        List of values in some sequential order
    col : str
        Name of column to check for consecutive values

    Returns
    -------
    list
        Increasing integer values, which represent each local group of
        images with the same label.
    """
    curr_val = 0
    prev_val = None
    local_groups = []

    for val in values:
        if val != prev_val:
            curr_val += 1
            prev_val = val
        
        local_groups.append(curr_val)

    return local_groups


def filter_most_confident(df_pred, local=False):
    """
    Given predictions for all images across multiple US sequences, filter the
    prediction with the highest confidence (based on output activation).

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    local : bool, optional
        If True, gets the most confident prediction within each group of
        consecutive images with the same label. Otherwise, aggregates by view
        label to find the most confident view label predictions,
        by default False.

    Returns
    -------
    pandas.DataFrame
        Filtered predictions
    """
    df_pred = df_pred.copy()

    if not local:
        # Get most confident pred per view per sequence (ignoring seq. number)
        df_seqs = df_pred.groupby(by=["id", "visit", "label"])
        df_filtered = df_seqs.apply(lambda df: df[df.out == df.out.max()])
    else:
        # Get most confident pred per group of consecutive labels per sequence
        # 0. Sort by id, visit and sequence number
        df_pred = df_pred.sort_values(by=["id", "visit", "seq_number"])

        # 1. Identify local groups of consecutive labels
        local_grps_per_seq = df_pred.groupby(by=["id", "visit"]).apply(
            lambda df: get_local_groups(df.label.values))
        df_pred["local_group"] = np.concatenate(local_grps_per_seq.values)

        df_seqs = df_pred.groupby(by=["id", "visit", "local_group"])
        df_filtered = df_seqs.apply(lambda df: df[df.out == df.out.max()])

    return df_filtered


def extract_from_label(df, col="label", extract="plane"):
    """
    Creates a new column called "<col>_<extract>", made from extracting data
    from a column of label/predicted label strings.

    Parameters
    ----------
    df : pd.DataFrame
        Contains <col> column with strings of the form <plane>_<side>
    col : str, optional
        Name of column of strings, by default "label"
    extract : str, optional
        What to extract. One of "plane" or "side", by default "plane"
    """
    def _extract_str(x, extract="plane"):
        parts = x.split("_")
        if extract == "side" and len(parts) > 1:
            return parts[1]
        return parts[0]

    df[f"{col}_{extract}"] = df[col].map(lambda x: _extract_str(x, extract))


def get_new_seq_numbers(df_pred):
    """
    Since sequence numbers are not all present (due to unlabeled images), create
    new sequence numbers from order of existing images in the sequence.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.

    Returns
    -------
    pd.DataFrame
        df_pred with added column "seq_number_new"
    """
    # Sort by sequence number
    df_pred = df_pred.sort_values(by=["id", "visit", "seq_number"],
                                  ignore_index=True)

    # Get new sequence numbers (rank)
    new_seq_numbers = np.concatenate(df_pred.groupby(by=["id", "visit"]).apply(
        lambda df: list(range(len(df)))).to_numpy())

    df_pred["seq_number_new"] = new_seq_numbers

    return df_pred


def show_example_side_predictions(df_pred, n=5, relative_side=False):
    """
    Given label sequences and their corresponding side predictions (encoded as
    single characters), print full label and prediction sequences with incorrect
    predictions colored red.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    n : int
        Number of random unique sequences to show
    relative_side : bool
        If True, predicted labels must be relative side (e.g., Transverse_First)
    """
    if relative_side:
        side_to_idx = side_to_idx = {"First": "1", "Second": "2", "None": "-",}
    else:
        side_to_idx = side_to_idx = {"Left": "1", "Right": "2", "None": "-",}
    labels = df_pred.groupby(by=["id", "visit"]).apply(
        lambda df: "".join(df.sort_values(by=["seq_number"]).label.map(
            lambda x: side_to_idx[utils.extract_from_label(x, "side")]).tolist()))
    preds = df_pred.groupby(by=["id", "visit"]).apply(
        lambda df: "".join(df.sort_values(by=["seq_number"]).pred.map(
            lambda x: side_to_idx[utils.extract_from_label(x, "side")]).tolist()))

    for _ in range(n):
        # Choose random sequence
        idx = random.randint(0, len(labels)-1)
        labels_str = labels.iloc[idx]
        preds_str = preds.iloc[idx]

        # Color prediction sequence by correct/wrong
        colored_pred_str = ""
        for i in range(len(labels_str)):
            pred = preds_str[i]
            color_code = Fore.GREEN if labels_str[i] == preds_str[i] else \
                Fore.MAGENTA
            colored_pred_str += f"{color_code}{pred}{Style.RESET_ALL}"

        print("")
        print(labels_str)
        print(colored_pred_str)


def get_model_class(model_type):
    """
    Return model class and parameters, needed to load model class from
    checkpoint.

    Parameters
    ----------
    model_type : str
        Name of model. Must be in MODEL_TYPES

    Returns
    -------
    tuple of (cls, dict)
        The first is reference to a class. Can be used to instantiate a model.
        The second is a dict of parameters needed in cls.load_from_checkpoint.
    """
    assert model_type in MODEL_TYPES

    # Parameters to pass into load_from_checkpoint
    load_from_ckpt_params = {}

    # Flags based on model name
    moco = ("moco" in model_type)
    sequential = ("seq" in model_type)

    # Choose model class
    if moco:
        if sequential:
            model_cls = LinearLSTM
        else:
            model_cls = LinearClassifier

        # Backbone used in MoCo pretraining
        load_from_ckpt_params["backbone"] = torch.nn.Sequential(
            EfficientNet.from_name("efficientnet-b0", include_top=False))
    elif sequential:
        model_cls = EfficientNetLSTM
    else:
        model_cls = EfficientNetPL

    return model_cls, load_from_ckpt_params


################################################################################
#                                  Main Flows                                  #
################################################################################
def main_test_set(model_cls, checkpoint_path,
                  load_from_ckpt_params=None,
                  save_path=TEST_PRED_PATH, sequential=False,
                  include_unlabeled=False, seq_number_limit=None,
                  relative_side=False):
    """
    Performs inference on test set, and saves results

    Parameters
    ----------
    model_cls : class reference
        Used to load specific model from checkpoint path
    checkpoint_path : str
        Path to file containing EfficientNetPL model checkpoint
    load_from_ckpt_params : dict, optional
        Additional keyword parameters to pass into load_from_checkpoint
    save_path : str, optional
        Path to file to save test results to, by default TEST_PRED_PATH
    sequential : bool, optional
        If True, feed full image sequences into model, by default False.
    include_unlabeled : bool, optional
        If True, include unlabeled images in test set, by default False.
    relative_side : bool, optional
        If True, converts side (Left/Right) to order in which side appeared
        (First/Second/None). Requires <extract> to be True, by default False.
    seq_number_limit : int, optional
        If provided, filters out test set samples with sequence numbers higher
        than this value, by default None.
    """
    # 2. Get metadata, specifically for the test set
    # 2.0 Get image filenames and labels
    df_metadata = utils.load_metadata(
        extract=True,
        include_unlabeled=include_unlabeled, img_dir=constants.DIR_IMAGES,
        relative_side=relative_side)

    # 2.1 Get hyperparameters for run
    model_dir = os.path.dirname(checkpoint_path)
    hparams = get_hyperparameters(model_dir)

    # 2.2 Load test metadata
    df_test_metadata = get_test_set_metadata(df_metadata, hparams)

    # If provided, filter out high sequence number images
    if seq_number_limit:
        mask = (df_test_metadata["seq_number"] <= seq_number_limit)
        df_test_metadata = df_test_metadata[mask]

    # 2.3 Sort, so no mismatch occurs due to groupby sorting
    df_test_metadata = df_test_metadata.sort_values(by=["id", "visit"],
                                                    ignore_index=True)

    # 3. Load existing model and send to device
    model = model_cls.load_from_checkpoint(
        checkpoint_path,
        **(load_from_ckpt_params if load_from_ckpt_params else {}))
    model = model.to(DEVICE)

    # 4. Get predictions on test set
    if not sequential:
        filenames = df_test_metadata.filename.tolist()
        preds, probs, outs = predict_on_images(model=model, filenames=filenames,
                                               img_dir=None)
    else:
        # Perform inference one sequence at a time
        ret = df_test_metadata.groupby(by=["id", "visit"]).progress_apply(
            lambda df: predict_on_sequences(
                model=model, filenames=df.filename.tolist(), img_dir=None)
        )

        # Flatten predictions, probs and model outputs from all sequences
        preds = np.concatenate(ret.map(lambda x: x[0]).to_numpy())
        probs = np.concatenate(ret.map(lambda x: x[1]).to_numpy())
        outs = np.concatenate(ret.map(lambda x: x[2]).to_numpy())

    # 5. Save predictions on test set
    df_test_metadata["pred"] = preds
    df_test_metadata["prob"] = probs
    df_test_metadata["out"] = outs
    df_test_metadata.to_csv(save_path, index=False)


if __name__ == '__main__':
    # NOTE: Chosen checkpoint and model type
    MODEL_TYPE = MODEL_TYPES[-1]
    ckpt_path = constants.DIR_RESULTS + MODEL_TYPE_TO_CKPT[MODEL_TYPE]

    # Flag for relative side encoding
    relative_side = ("relative" in MODEL_TYPE)

    # Add model type to save path
    test_save_path = TEST_PRED_PATH % (MODEL_TYPE,)

    # Inference on test set
    if not os.path.exists(test_save_path):
        # Get model class based on model type
        model_cls, load_ckpt_params = get_model_class(MODEL_TYPE)

        # Perform inference
        main_test_set(model_cls, ckpt_path, load_ckpt_params,
                      save_path=test_save_path,
                      sequential=("seq" in MODEL_TYPE),
                      include_unlabeled=("other" in MODEL_TYPE),
                      seq_number_limit=40 if "short" in MODEL_TYPE else None,
                      relative_side=relative_side)

    # Load test metadata
    df_pred = pd.read_csv(test_save_path)

    # 5-View (Not including 'Other' label)
    if "five_view" in MODEL_TYPE and "other" not in MODEL_TYPE:
        # Print reasons for misclassification of most confident predictions
        check_misclassifications(df_pred, relative_side=relative_side)

        # Plot confusion matrix for most confident predictions
        plot_confusion_matrix(df_pred, filter_confident=True,
                              relative_side=relative_side)
        # Plot confusion matrix for all predictions
        plot_confusion_matrix(df_pred, filter_confident=False,
                              relative_side=relative_side)
        # Print confusion matrix for all predictions
        print_confusion_matrix(
            df_pred, constants.CLASSES["relative" if relative_side else ""])

        # Plot probability of predicted labels over the sequence number
        plot_prob_over_sequence(df_pred, update_seq_num=True, correct_only=True)

        # Plot accuracy over sequence number
        plot_acc_over_sequence(df_pred, update_seq_num=True)

        # Plot number of images over sequence number
        plot_image_count_over_sequence(df_pred, update_seq_num=True)

        # Relative side labels
        if relative_side:
            check_rel_side_pred_progression(df_pred)

        # Show randomly chosen side predictions for full sequences
        show_example_side_predictions(df_pred, relative_side=relative_side)

    # Bladder vs. Other models
    if "binary" in MODEL_TYPE:
        # Plot binary avg. prediction probabilites by view
        plot_pred_probability_by_views(df_pred)

    # 5-View (Includin 'Other' label)
    if "five_view" in MODEL_TYPE and "other" in MODEL_TYPE:
        check_others_pred_progression(df_pred)
