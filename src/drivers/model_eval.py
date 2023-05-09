"""
model_eval.py

Description: Used to evaluate a trained model's performance.
"""

# Standard libraries
import argparse
import logging
import os
import random
from colorama import Fore, Style
from collections import OrderedDict

# Non-standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as T
from arch.bootstrap import IIDBootstrap
from scipy.stats import pearsonr
from sklearn import metrics as skmetrics
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

# Custom libraries
from src.data import constants
from src.data_prep import utils
from src.data_viz import plot_umap
from src.data_viz import utils as viz_utils
from src.drivers import embed, load_model, load_data


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
CLASS_TO_IDX = {"Sagittal_Left": 0, "Transverse_Left": 1, "Bladder": 2,
                "Transverse_Right": 3, "Sagittal_Right": 4, "Other": 5}
IDX_TO_CLASS = {v: u for u, v in CLASS_TO_IDX.items()}

# Plot theme (light/dark)
THEME = "dark"

# Flag to calculate metrics for each evaluation set
CALCULATE_METRICS = True

# Flag to create embeddings and plot UMAP for each evaluation set
EMBED = False


################################################################################
#                               Paths Constants                                #
################################################################################
# Model type to checkpoint file
MODEL_TYPE_TO_CKPT = OrderedDict({
    "binary" : "/binary_classifier/0/epoch=12-step=2586.ckpt",

    "five_view": "/five_view/0/epoch=6-step=1392.ckpt",
    "five_view_seq": "cnn_lstm_8/0/epoch=31-step=5023.ckpt",
    "five_view_seq_w_other": "/five_view/0/epoch=6-step=1392.ckpt",
    "five_view_seq_relative": "relative_side_grid_search(2022-09-21)/relative_side"
                              "(2022-09-20_22-09)/0/epoch=7-step=1255.ckpt",

    # Checkpoint of MoCo Linear Classifier
    "five_view_moco": "moco_linear_eval_4/0/last.ckpt",
    # Checkpoint of MoCo LSTM + Linear Classifier
    "five_view_moco_seq": "moco_linear_lstm_eval_0/0/epoch=12-step=129.ckpt",

    # Checkpoint of MoCo (Stanford train) Linear Classifier
    "five_view_moco_su": "moco_linear_eval_su_to_sk_1/0/epoch=0-step=396.ckpt",
    # Checkpoint of MoCo (Stanford train) LSTM + Linear Classifier
    "five_view_moco_seq_su": "moco_linear_lstm_eval_su_to_sk_1/0/"
                             "epoch=9-step=99.ckpt",

    # Checkpoint of MoCo (SickKids All) Linear Classifier
    "five_view_moco_sk_all": "moco_linear_eval_sk_all/0/"
                             "epoch=24-step=9924.ckpt",
    # Checkpoint MoCo (SickKids All) LSTM + Linear
    "five_view_moco_seq_sk_all": "moco_linear_lstm_eval_sk_all/0/"
                                 "epoch=22-step=229.ckpt",
})

# Type of models
MODEL_TYPES = list(MODEL_TYPE_TO_CKPT.keys())


################################################################################
#                                Initialization                                #
################################################################################
def init(parser):
    """
    Initialize ArgumentParser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser object
    """
    arg_help = {
        "exp_name": "Name/s of experiment/s (to evaluate)",
        "dset": "List of dataset split or test dataset name to evaluate",
    }
    parser.add_argument("--exp_name", required=True,
                        nargs='+',
                        help=arg_help["exp_name"])
    parser.add_argument("--dset", default=[constants.DEFAULT_EVAL_DSET],
                        nargs='+',
                        help=arg_help["dset"])


################################################################################
#                             Inference - Related                              #
################################################################################
@torch.no_grad()
def predict_on_images(model, filenames, img_dir=constants.DIR_IMAGES,
                      mask_bladder=False,
                      **hparams):
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
    mask_bladder : bool, optional
        If True, mask out predictions on bladder/middle side, leaving only
        kidney labels. Defaults to False.
    hparams : dict, optional
        Keyword arguments for experiment hyperparameters

    Returns
    -------
    pandas.DataFrame
        For each image, contains
            - prediction for each image,
            - probability of each prediction,
            - raw model output
    """
    # Get mapping of index to class
    label_part = hparams.get("label_part")
    idx_to_class = constants.LABEL_PART_TO_CLASSES[label_part]["idx_to_class"]

    # Set to evaluation mode
    model.eval()

    # Predict on each images one-by-one
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

        # If specified, remove Bladder/None as a possible prediction
        # NOTE: Assumes model predicts bladder/none as the 3rd index
        if mask_bladder:
            out = out[:, :2]

        # Get index of predicted label
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
        pred_label = idx_to_class[pred]
        preds.append(pred_label)

    # Pack into dataframe
    df_preds = pd.DataFrame({
        "pred": preds,
        "prob": probs,
        "out": outs,
    })

    return df_preds


@torch.no_grad()
def predict_on_sequences(model, filenames, img_dir=constants.DIR_IMAGES,
                         mask_bladder=False,
                         **hparams):
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
    mask_bladder : bool, optional
        If True, mask out predictions on bladder/middle side, leaving only
        kidney labels. Defaults to False.
    hparams : dict, optional
        Keyword arguments for experiment hyperparameters

    Returns
    -------
    pandas.DataFrame
        For each frame in the sequence, contains
            - prediction for each image,
            - probability of each prediction,
            - raw model output
    """
    # Get mapping of index to class
    label_part = hparams.get("label_part")
    idx_to_class = constants.LABEL_PART_TO_CLASSES[label_part]["idx_to_class"]

    # Set to evaluation mode
    model.eval()

    # Predict on each images one-by-one
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

    # If specified, remove Bladder/None as a possible prediction
    # NOTE: Assumes model predicts bladder/none as the 3rd index
    if mask_bladder:
        outs = outs[:, :2]

    # Compute index of predicted label
    preds = torch.argmax(outs, dim=1).numpy()

    # Get probability
    probs = torch.nn.functional.softmax(outs, dim=1)
    probs = torch.max(probs, dim=1).values.numpy()

    # Get maximum activation
    outs = torch.max(outs, dim=1).values.numpy()

    # Convert from encoded label to label name
    preds = np.vectorize(idx_to_class.__getitem__)(preds)

    # Pack into dataframe
    df_preds = pd.DataFrame({
        "pred": preds,
        "prob": probs,
        "out": outs,
    })

    return df_preds


@torch.no_grad()
def multi_predict_on_sequences(model, filenames, img_dir=constants.DIR_IMAGES,
                               **hparams):
    """
    Performs inference on a full ultrasound sequence specified with a
    multi-output model. Returns predictions, probabilities and raw model output.

    Parameters
    ----------
    model : torch.nn.Module
        A trained multi-output model (EfficientNetLSTMMulti).
    filenames : np.array or array-like
        Filenames (or full paths) to images from one unique sequence to infer.
    img_dir : str, optional
        Path to directory containing images, by default constants.DIR_IMAGES
    hparams : dict, optional
        Keyword arguments for experiment hyperparameters

    Returns
    -------
    pandas.DataFrame
        For each frame in the sequence and for each side, contains
            - prediction for each image,
            - probability of each prediction,
            - raw model output
    """
    # Set to evaluation mode
    model.eval()

    # Predict on each images one-by-one
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

    # Perform inference to get both side/plane raw output
    out_dict = model(imgs)

    # Accumulate prediction, probability and raw output for each label
    label_to_results = {}
    for label_part in out_dict.keys():
        outs = out_dict[label_part].detach().cpu()
        preds = torch.argmax(outs, dim=1).numpy()

        # Get probability
        probs = torch.nn.functional.softmax(outs, dim=1)
        probs = torch.max(probs, dim=1).values.numpy()

        # Get maximum activation
        outs = torch.max(outs, dim=1).values.numpy()

        # Get mapping of index to class
        idx_to_class = \
            constants.LABEL_PART_TO_CLASSES[label_part]["idx_to_class"]

        # Convert from encoded label to label name
        preds = np.vectorize(idx_to_class.__getitem__)(preds)

        # Store results
        label_to_results[f"{label_part}_out"] = outs
        label_to_results[f"{label_part}_prob"] = probs
        label_to_results[f"{label_part}_pred"] = preds

    return pd.DataFrame(label_to_results)


################################################################################
#                            Analysis of Inference                             #
################################################################################
def plot_confusion_matrix(df_pred, filter_confident=False, ax=None, **hparams):
    """
    Plot confusion matrix based on model predictions.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions and labels. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    filter_confident : bool, optional
        If True, filters for most confident prediction in each view label for
        each unique sequence, before creating the confusion matrix.
    ax : matplotlib.Axis, optional
        Axis to plot confusion matrix on
    hparams : dict, optional
        Keyword arguments for experiment hyperparameters
    """
    # If flagged, get for most confident pred. for each view label per seq.
    if filter_confident:
        df_pred = filter_most_confident(df_pred)

    # Gets all labels
    if hparams.get("relative_side"):
        all_labels = constants.CLASSES["relative"]
    else:
        label_part = hparams.get("label_part")
        all_labels = constants.LABEL_PART_TO_CLASSES[label_part]["classes"]

    # Plots confusion matrix
    cm = skmetrics.confusion_matrix(df_pred["label"], df_pred["pred"],
                                    labels=all_labels)
    disp = skmetrics.ConfusionMatrixDisplay(cm, display_labels=all_labels)
    disp.plot(ax=ax)

    # Set title
    if ax:
        title = "Confusion Matrix (%s)" % \
            ("Most Confident" if filter_confident else "All")
        ax.set_title(title)


def print_confusion_matrix(df_pred, **hparams):
    """
    Prints confusion matrix with proportion and count

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions and labels. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    hparams : dict, optional
        Keyword arguments for experiment hyperparameters
    """
    def get_counts_and_prop(df):
        """
        Given all instances for one label, get the proportions and counts for
        each of the predicted labels.

        Parameters
        ----------
        df : pandas.DataFrame
            Model prediction for 1 label.
        """
        df_counts = df["pred"].value_counts()
        num_samples = len(df)

        return df_counts.map(lambda x: f"{round(x/num_samples, 2)} ({x})")

    # Gets all labels
    if hparams.get("relative_side"):
        all_labels = constants.CLASSES["relative"]
    else:
        label_part = hparams.get("label_part")
        all_labels = constants.LABEL_PART_TO_CLASSES[label_part]["classes"]

    # Check that all labels are found
    assert set(df_pred["label"].unique()) == set(all_labels), \
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

    # Fill in columns, if not predicted (at all)
    for label in all_labels:
        if label not in df_cm.columns:
            df_cm[label] = "0.0 (0)"

    # Reorder column and index by given labels
    df_cm = df_cm.loc[:, all_labels].reindex(all_labels)

    viz_utils.print_table(df_cm)


def plot_pred_probability_by_views(df_pred, relative_side=False):
    """
    Plot average probability of prediction per view label.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    relative_side : bool, optional
        If True, assumes predicted labels are encoded for relative sides, by
        default False.
    """
    df_pred = df_pred.copy()

    # Filter for correct predictions
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
    Given the most confident model predictions, determine the percentage of
    misclassifications that are due to:
        1. Swapping sides (e.g., Saggital Right mistaken for Saggital Left)
        2. Adjacent views

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
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
    viz_utils.print_table(df_results)


def plot_prob_over_sequence(df_pred, correct_only=False, update_seq_num=False):
    """
    Plots average probability of prediction over each number in the sequence.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
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
        Model predictions. Each row contains a label,
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
        Model predictions. Each row contains a label,
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
    Given model predictions for full sequences that include "Other" labels,
    show (predicted) label progression for unlabeled images among real labels.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
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
    viz_utils.print_table(label_seq_counts, show_index=False)


def check_rel_side_pred_progression(df_pred):
    """
    Given model predictions for full sequences where side labels are relative
    to the first side, show label/predicted label progression.

    Note
    ----
    Highlight locations where error in toggling between first/second occurs.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
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
    viz_utils.print_table(label_seq_counts, show_index=False)


def compare_prediction_similarity(df_pred_1, df_pred_2):
    """
    For each label, print Pearson correlation of:
        (1) predicted label with model 1 vs model 2
        (2) side predicted
        (3) view predicted

    Parameters
    ----------
    df_pred_1 : pandas.DataFrame
        Inference on the same dset by model 1
    df_pred_2 : pandas.DataFrame
        Inference on the same dset by model 2
    """
    # Correlation of exact label
    print("""
################################################################################
#                         Exact Prediction Correlation                         #
################################################################################
    """)
    for label in df_pred_1.label.unique():
        pred_mask_1 = (df_pred_1.pred == label)
        pred_mask_2 = (df_pred_2.pred == label)
        print(f"{label}: {pearsonr(pred_mask_1, pred_mask_2)}")

    # Extract side and plane (view) from predicted label
    extract_from_label(df_pred_1, col="pred", extract="side")
    extract_from_label(df_pred_1, col="pred", extract="plane")
    extract_from_label(df_pred_2, col="pred", extract="side")
    extract_from_label(df_pred_2, col="pred", extract="plane")

    # Correlation of side
    print("""
################################################################################
#                          Side Predicted Correlation                          #
################################################################################
    """)
    for side in ("Left", "Right", "None"):
        pred_mask_1 = (df_pred_1.pred_side == side)
        pred_mask_2 = (df_pred_2.pred_side == side)
        print(f"{side}: {pearsonr(pred_mask_1, pred_mask_2)}")

    # Correlation of view
    print("""
################################################################################
#                         Plane Predicted Correlation                          #
################################################################################
    """)
    for plane in ("Saggital", "Transverse", "Bladder"):
        pred_mask_1 = (df_pred_1.pred_plane == plane)
        pred_mask_2 = (df_pred_2.pred_plane == plane)
        print(f"{plane}: {pearsonr(pred_mask_1, pred_mask_2)}")


def plot_rolling_accuracy(df_pred, window_size=15, max_seq_num=75,
                          update_seq_num=False, save_path=None):
    """
    Given predictions on ultrasound image sequences, plot accuracy on
    overlapping windows of size `window_size`, based on sequence number.

    Note
    ----
    Windows are done on sequence numbers (e.g., sequence numbers [0-14])

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    window_size : int, optional
        Window size for sequence numbers, by default 15
    max_seq_num : int, optional
        Maximum sequence number to consider, by default 75
    update_seq_num : bool, optional
        If True, accounts for missing images between sequence numbers by
        creating new (raw) sequence number based on existing images, by default
        False.
    save_path : str, optional
        If provided, saves figure to path, by default None.

    Returns
    -------
    matplotlib.axes.Axes
        Axis plot
    """
    seq_num_col = "seq_number" if not update_seq_num else "seq_number_new"
    # If specified, update sequence numbers to relative sequence numbers
    if update_seq_num:
        df_pred = get_new_seq_numbers(df_pred)

    # Specify maximum sequencec number
    true_max_seq_num = df_pred[seq_num_col].max()
    max_seq_num = min(max_seq_num, true_max_seq_num)

    # Calculate accuracies of windows
    window_accs = []
    sample_sizes = []
    for i in range(0, max_seq_num-window_size):
        df_window = df_pred[(df_pred[seq_num_col] >= i) &
                            (df_pred[seq_num_col] < i+window_size)]
        window_accs.append(round((df_window.label == df_window.pred).mean(), 4))
        sample_sizes.append(len(df_window))

    # Create bar plot
    plt.figure(figsize=(10, 7))
    container = plt.bar(list(range(len(window_accs))), window_accs,
                        color="#377eb8",
                        alpha=0.7)

    # Add number of samples (used to calculate accuracy) as bar labels
    bar_labels = [f"N:{n}" if idx % 3 == 0 else "" for idx, n in
                    enumerate(sample_sizes)]
    plt.bar_label(container,
                    labels=bar_labels,
                    label_type="edge",
                    fontsize="small",
                    alpha=0.75,
                    padding=1)

    # Set y bounds
    plt.ylim(0, 1)

    # Add axis text
    plt.xlabel("Starting Sequence Number")
    plt.ylabel("Accuracy")
    plt.title("Rolling Accuracy over Sequence Number Windows of "
              f"Size {window_size}")

    plt.tight_layout()

    # Save figure if specified
    if save_path:
        plt.savefig(save_path)

    return plt.gca()


def calculate_metrics(df_pred, ci=False, **ci_kwargs):
    """
    Calculate important metrics given prediction and labels

    Parameters
    ----------
    df_pred : pd.DataFrame
        Model predictions and labels
    ci : bool, optional
        If True, add bootstrapped confidence interval, by default False.
    **ci_kwargs : dict, optional
        Keyword arguments to pass into `bootstrap_metric` if `ci` is True

    Returns
    -------
    pd.DataFrame
        Table containing metrics
    """
    # Accumulate exact metric, and confidence interval bounds (if specified)
    metrics = OrderedDict()

    # Accuracy by class
    unique_labels = sorted(df_pred["label"].unique())
    for label in unique_labels:
        df_pred_filtered = df_pred[df_pred.label == label]
        metrics[f"Label Accuracy ({label})"] = 0
        if not df_pred_filtered.empty:
            metrics[f"Label Accuracy ({label})"] = round(
                skmetrics.accuracy_score(
                    df_pred_filtered["label"],
                    df_pred_filtered["pred"]),
                4)

    # Overall accuracy
    metrics["Overall Accuracy"] = \
        round(skmetrics.accuracy_score(df_pred["label"], df_pred["pred"]), 4)
    # Bootstrap confidence interval
    if ci:
        point, (lower, upper) = bootstrap_metric(
            df_pred=df_pred,
            metric_func=skmetrics.accuracy_score,
            **ci_kwargs)
        metrics[f"Overall Accuracy"] = f"{point} ({lower}, {upper})"

    # F1 Score by class
    # NOTE: Overall F1 Score isn't calculated because it's equal to 
    #       Overall Accuracy in multi-label problems.
    # TODO: Implement wrapper function to allow bootstrapping f1_score
    f1_scores = skmetrics.f1_score(df_pred["label"], df_pred["pred"],
                                   labels=unique_labels,
                                   average=None)
    for i, f1_score in enumerate(f1_scores):
        metrics[f"Label F1-Score ({unique_labels[i]})"] = round(f1_score, 4)

    return pd.Series(metrics)


def calculate_metric_by_groups(df_pred, group_cols,
                               metric_func=skmetrics.accuracy_score):
    """
    Group predictions by patient/sequence ID, calculate metric on each group
    and get the average prediction

    Parameters
    ----------
    df_pred : pd.DataFrame
        Model predictions and labels
    group_cols : list
        List of columns in `df_pred` to group by
    metric_func : function, optional
        Reference to function that can be used to calculate a metric given the
        (label, predictions), by default sklearn.metrics.accuracy_score

    Returns
    -------
    str
        Contains mean and standard deviation of calculated metric across groups
    """
    # Calculate metric on each group
    grp_metrics = df_pred.groupby(by=group_cols).apply(
        lambda df_grp: metric_func(df_grp["label"], df_grp["pred"]))

    # Calculate average across groups
    mean = round(grp_metrics.mean(), 4)
    sd = round(grp_metrics.std(), 4)

    return f"{mean} +/- {sd}"


def calculate_hn_corr(df_pred):
    """
    Compute correlation between correct predictions and HN

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.

    Returns
    -------
    float
        Correlation between correctly predicted label and HN
    """
    # Filter for samples with HN label
    df_hn = df_pred.dropna(subset=["hn"])
    df_hn["correct"] = (df_hn.label == df_hn.pred)

    return pearsonr(df_hn["hn"], df_hn["correct"])[0]


def bootstrap_metric(df_pred,
                     metric_func=skmetrics.accuracy_score,
                     alpha=0.05,
                     n_bootstrap=12000,
                     seed=constants.SEED):
    """
    Perform BCa bootstrap on table of predictions to calculate metric with a
    bootstrapped confidence interval.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    metric_func : function, optional
        Reference to function that can be used to calculate a metric given the
        (label, predictions), by default sklearn.metrics.accuracy_score
    alpha : float, optional
        Desired significance level, by default 0.05
    n_bootstrap : int, optional
        Sample size for each bootstrap iteration
    seed : int, optional
        Random seed

    Returns
    -------
    tuple of (exact, (lower_bound, upper_bound))
        Output of `func` with 95% confidence interval ends
    """
    # Calculate exact point metric
    # NOTE: Assumes function takes in (label, pred)
    exact_metric = round(metric_func(df_pred["label"], df_pred["pred"]), 4)

    # Initialize bootstrap
    bootstrap = IIDBootstrap(
        df_pred["label"], df_pred["pred"],
        seed=seed)

    try:
        # Calculate empirical CI bounds
        ci_bounds = bootstrap.conf_int(
            func=metric_func,
            reps=n_bootstrap,
            method='bca',
            size=1-alpha,
            tail='two').flatten()
        # Round to 4 decimal places
        ci_bounds = np.round(ci_bounds, 4)
    except RuntimeError: # NOTE: May occur if all labels are predicted correctly
        ci_bounds = np.nan, np.nan

    return exact_metric, tuple(ci_bounds)


def eval_calculate_all_metrics(df_pred):
    """
    Calculates all eval. metrics in a proper table format

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.

    Returns
    -------
    pandas.DataFrame
        Table of formatted metrics
    """
    # 1. Calculate metrics
    # 1.1.1 For all samples
    df_metrics_all = calculate_metrics(df_pred, ci=True)
    df_metrics_all.name = "All"

    # 1.2 Stratify patients w/ HN and w/o HN
    if "hn" in df_pred.columns:
        df_metrics_w_hn = calculate_metrics(df_pred[df_pred.hn == 1])
        df_metrics_w_hn.name = "With HN"
        df_metrics_wo_hn = calculate_metrics(df_pred[df_pred.hn == 0])
        df_metrics_wo_hn.name = "Without HN"

    # 1.3 For images at label boundaries
    # NOTE: Disabled for now
    # df_metrics_at_boundary = calculate_metrics(
    #     df_pred[df_pred["at_label_boundary"]])
    # df_metrics_at_boundary.name = "At Label Boundary"

    # 1.4 Most confident view (per label) in each sequence
    # NOTE: Disabled for now
    # df_confident = filter_most_confident(df_pred, local=False)
    # df_metrics_confident = calculate_metrics(df_confident, ci=True)
    # df_metrics_confident.name = "Most Confident"

    # 1.5 Accuracy, grouped by patient
    df_metrics_all["Accuracy (By Patient)"] = \
        calculate_metric_by_groups(df_pred, ["id"])

    # 1.6 Accuracy, grouped by sequence
    df_metrics_all["Accuracy (By Seq)"] = \
        calculate_metric_by_groups(df_pred, ["id", "visit"])

    # Filler columns
    filler = df_metrics_all.copy()
    filler[:] = ""
    filler.name = ""

    # 2. Combine
    df_metrics = pd.concat([
        df_metrics_all,
        # filler,
        # df_metrics_w_hn, df_metrics_wo_hn,
        # filler,
        # df_metrics_at_boundary,
        # filler,
        # df_metrics_confident,
        ], axis=1)

    return df_metrics


def eval_create_plots(df_pred, hparams, inference_dir,
                      dset=constants.DEFAULT_EVAL_DSET):
    """
    Visual evaluation of model performance

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    hparams : dict
        Keyword arguments for experiment hyperparameters
    inference_dir : str
        Path to experiment-specific inference directory
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.
    """
    # 0. Reset theme
    viz_utils.set_theme(THEME)

    # 1. Create confusion matrix plot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # 1.1 Plot confusion matrix for most confident predictions
    plot_confusion_matrix(df_pred, filter_confident=True, ax=ax1, **hparams)
    # 1.2 Plot confusion matrix for all predictions
    plot_confusion_matrix(df_pred, filter_confident=False, ax=ax2, **hparams)
    plt.tight_layout()
    plt.savefig(os.path.join(inference_dir, f"{dset}_confusion_matrix.png"))

    # 1.3 Print confusion matrix for all predictions
    # NOTE: May error if the same label is predicted for all samples
    try:
        print_confusion_matrix(df_pred, **hparams)
    except:
        pass

    # 2. Print reasons for misclassification of most confident predictions
    if not hparams.get("label_part"):
        check_misclassifications(df_pred,
                                 relative_side=hparams.get("relative_side"))

    # 3. Show randomly chosen side predictions for full sequences
    show_example_side_predictions(df_pred,
                                  relative_side=hparams.get("relative_side"),
                                  label_part=hparams.get("label_part"))


def calculate_exp_metrics(exp_name, dset, hparams=None, mask_bladder=False):
    """
    Given that inference was performed, compute metrics for experiment model
    and dataset.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str
        Name of evaluation split or test dataset
    hparams : dict, optional
        Experiment hyperparameters, by default None
    mask_bladder : bool, optional
        If True, bladder predictions are masked out, by default False
    """
    # 0. Overwrite `mask_bladder`, based on dset
    if dset in constants.DSETS_MISSING_BLADDER:
        LOGGER.info(f"Hospital missing bladder labels found ({dset})! "
                    "Overwriting `mask_bladder`")
        mask_bladder = True

    # 1. Get experiment hyperparameters (if not provided)
    hparams = hparams if hparams \
        else load_model.get_hyperparameters(exp_name=exp_name)

    # 2. Load inference
    save_path = create_save_path(
        exp_name,
        dset=dset,
        mask_bladder=mask_bladder)
    df_pred = pd.read_csv(save_path)
    # 2.0 Ensure no duplicates (or sequential data only)
    # NOTE: Because Stanford had duplicate metadata, there were duplicate preds
    if not mask_bladder:
        df_pred = df_pred.drop_duplicates(subset=["id", "visit", "seq_number"])
    # 2.1 Add side/plane label, if not present
    for label_part in constants.LABEL_PARTS:
        if label_part not in df_pred.columns:
            df_pred[label_part] = utils.get_labels_for_filenames(
                df_pred["filename"].tolist(), label_part=label_part)
    # 2.2 Add HN labels, if not already exists. NOTE: Needs side label to work
    if "hn" not in df_pred.columns:
        df_pred = utils.extract_hn_labels(df_pred)
    # 2.3 Specify which images are at label boundaries
    # NOTE: Disabled for now
    # df_pred["at_label_boundary"] = utils.get_label_boundaries(df_pred)

    # If multi-output, evaluate each label part, separately
    label_parts = constants.LABEL_PARTS if hparams.get("multi_output") \
        else [hparams.get("label_part")]
    for label_part in label_parts:
        temp_exp_name = exp_name

        # If multi-output, make temporary changes to hparams
        orig_label_part = hparams.get("label_part")
        if hparams.get("multi_output"):
            # Change experiment name to create different folders
            temp_exp_name += f"__{label_part}"
            # Force to be a specific label part
            hparams["label_part"] = label_part
            df_pred["label"] = df_pred[label_part]
            df_pred["pred"] = df_pred[f"{label_part}_pred"]
            df_pred["prob"] = df_pred[f"{label_part}_prob"]
            df_pred["out"] = df_pred[f"{label_part}_out"]

        # Add suffix, if predictions mask bladder
        if mask_bladder:
            temp_exp_name += "__mask_bladder"

        # Experiment-specific inference directory, to store figures
        inference_dir = os.path.join(constants.DIR_INFERENCE, temp_exp_name)
        if not os.path.isdir(inference_dir):
            os.mkdir(inference_dir)

        # 4. Calculate metrics
        df_metrics = eval_calculate_all_metrics(df_pred)
        df_metrics.to_csv(os.path.join(inference_dir,
                                        f"{dset}_metrics.csv"))

        # 5. Create plots for visual evaluation
        eval_create_plots(df_pred, hparams, inference_dir, dset=dset)

        # Revert temporary changes
        hparams["label_part"] = orig_label_part
        df_pred = df_pred.drop(columns=["label", "pred", "prob", "out"])


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
        Model predictions. Each row contains a label,
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
    df[f"{col}_{extract}"] = df[col].map(
        lambda x: utils.extract_from_label(x, extract))


def get_new_seq_numbers(df_pred):
    """
    Since sequence numbers are not all present (due to unlabeled images), create
    new sequence numbers from order of existing images in the sequence.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Model predictions. Each row contains a label,
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


def show_example_side_predictions(df_pred, n=5, relative_side=False,
                                  label_part=None):
    """
    Given label sequences and their corresponding side predictions (encoded as
    single characters), print full label and prediction sequences with incorrect
    predictions colored red.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    n : int
        Number of random unique sequences to show
    relative_side : bool
        If True, predicted labels must be relative side (e.g., Transverse_First)
    """
    # If relative side label
    if relative_side:
        side_to_idx = {"First": "1", "Second": "2", "None": "-",}
    else:
        side_to_idx = {"Left": "1", "Right": "2", "None": "-",}

    # How to determine side index
    side_func = lambda x: side_to_idx[utils.extract_from_label(x, "side")]
    if label_part == "side":
        side_func = lambda x: side_to_idx[x]
    else:
        LOGGER.warning(f"Invalid `label_part` provided! {label_part}")
        return

    # Apply function above to labels/preds
    labels = df_pred.groupby(by=["id", "visit"]).apply(
        lambda df: "".join(
            df.sort_values(by=["seq_number"]).label.map(side_func).tolist()))
    preds = df_pred.groupby(by=["id", "visit"]).apply(
        lambda df: "".join(
            df.sort_values(by=["seq_number"]).pred.map(side_func).tolist()))

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


def create_save_path(exp_name, dset=constants.DEFAULT_EVAL_DSET, **extra_flags):
    """
    Create file path to dset predictions, based on experiment name and keyword
    arguments

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.
    **extra_flags : dict, optional
        Keyword arguments, specifying extra flags used during inference

    Returns
    -------
    str
        Expected path to dset predictions
    """
    # Add mask bladder, if dset doesn't contain bladders
    if dset in constants.DSETS_MISSING_BLADDER:
        extra_flags["mask_bladder"] = True

    # Add true flags to the experiment name
    for flag, val in extra_flags.items():
        if val:
            exp_name += f"__{flag}"

    # Create inference directory, if not exists
    inference_dir = os.path.join(constants.DIR_INFERENCE, exp_name)
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    # Expected path to dset inference
    fname = f"{dset}_set_results.csv"
    save_path = os.path.join(inference_dir, fname)

    return save_path


def calculate_per_seq_silhouette_score(exp_name, label_part="side",
                                       exclude_labels=("None",),
                                       dset=constants.DEFAULT_EVAL_DSET):
    """
    Calculate a per - ultrasound sequence Silhouette score.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    label_part : str, optional
        If specified, either `side` or `plane` is extracted from each label
        and used as the given label, by default "side"
    exclude_labels : list or array-like, optional
        List of labels whose matching samples will be excluded when calculating
        the Silhouette score, by default ("None",)
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.

    Returns
    -------
    float
        Mean Silhouette Score across unique ultrasound sequences
    """
    # Load embeddings
    df_embeds = embed.get_embeds(exp_name, dset=dset)
    df_embeds = df_embeds.rename(columns={"paths": "files"})    # legacy name

    # Extract metadata from image file paths
    df_metadata = pd.DataFrame({"filename": df_embeds["files"]})
    df_metadata = utils.extract_data_from_filename(df_metadata, col="filename")

    filenames = df_metadata["filename"].map(os.path.basename).to_numpy()

    # Get view labels (if any) for all extracted images
    view_labels = utils.get_labels_for_filenames(
        filenames, label_part=label_part)
    df_metadata["label"] = view_labels

    # Isolate UMAP embeddings (all patients)
    df_embeds_only = df_embeds.drop(columns=["files"])

    # Mask out excluded labels
    if exclude_labels is not None:
        mask = ~df_metadata["label"].isin(exclude_labels)
        df_embeds_only = df_embeds_only[mask]
        df_metadata = df_metadata[mask]

    # Calculate per-sequence Silhouette score
    scores = []
    for p_id, visit in list(df_metadata.groupby(by=["id", "visit"]).groups):
        mask = ((df_metadata["id"] == p_id) & (df_metadata["visit"] == visit))
        # Ignore, if number of labels < 2
        if df_metadata["label"][mask].nunique() < 2:
            continue
        scores.append(skmetrics.silhouette_samples(
            X=df_embeds_only[mask],
            labels=df_metadata["label"][mask],
            metric="cosine").mean())

    return np.mean(scores)


################################################################################
#                                  Main Flows                                  #
################################################################################
def infer_dset(exp_name,
               dset=constants.DEFAULT_EVAL_DSET,
               seq_number_limit=None,
               mask_bladder=False,
               overwrite_existing=False,
               **overwrite_hparams):
    """
    Perform inference on dset, and saves results

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.
    seq_number_limit : int, optional
        If provided, filters out dset split samples with sequence numbers higher
        than this value, by default None.
    mask_bladder : bool, optional
        If True, mask out predictions on bladder/middle side, leaving only
        kidney labels. Defaults to False.
    overwrite_existing : bool, optional
        If True and prediction file already exists, overwrite existing, by
        default False.
    overwrite_hparams : dict, optional
        Keyword arguments to overwrite experiment hyperparameters

    Raises
    ------
    RuntimeError
        If `exp_name` does not lead to a valid training directory
    """
    # 0. Create path to save predictions
    pred_save_path = create_save_path(exp_name, dset=dset,
                                      seq_number_limit=seq_number_limit,
                                      mask_bladder=mask_bladder)
    # Early return, if prediction already made
    if os.path.isfile(pred_save_path) and not overwrite_existing:
        return

    # 0. Get experiment directory, where model was trained
    model_dir = load_model.get_exp_dir(exp_name)

    # 1 Get experiment hyperparameters
    hparams = load_model.get_hyperparameters(model_dir)
    hparams.update(overwrite_hparams)

    # 2. Load existing model and send to device
    model = load_model.load_pretrained_from_exp_name(
        exp_name, overwrite_hparams=overwrite_hparams)
    model = model.to(DEVICE)

    # 3. Load data
    # NOTE: Ensure data is loaded in the non-SSL mode
    dm = load_data.setup_data_module(hparams, self_supervised=False)

    # 3.1 Get metadata (for specified split)
    df_metadata = load_data.get_dset_metadata(dm, hparams, dset=dset)
    # 3.2 If provided, filter out high sequence number images
    if seq_number_limit:
        mask = (df_metadata["seq_number"] <= seq_number_limit)
        df_metadata = df_metadata[mask]
    # 3.3 Sort, so no mismatch occurs due to groupby sorting
    df_metadata = df_metadata.sort_values(by=["id", "visit"], ignore_index=True)

    # 4. Predict on dset split
    # If multi-output model
    if hparams.get("multi_output"):
        if not hparams["full_seq"]:
            raise NotImplementedError("Multi-output model is not implemented "
                                      "for single-images!")
        # Perform inference one sequence at a time
        df_preds = df_metadata.groupby(by=["id", "visit"]).progress_apply(
            lambda df: multi_predict_on_sequences(
                model=model, filenames=df.filename.tolist(), img_dir=None,
                **hparams)
        )
        # Remove groupby index
        df_preds = df_preds.reset_index(drop=True)
    else:
        # If temporal model
        if hparams["full_seq"]:
            # Perform inference one sequence at a time
            df_preds = df_metadata.groupby(by=["id", "visit"]).progress_apply(
                lambda df: predict_on_sequences(
                    model=model, filenames=df.filename.tolist(), img_dir=None,
                    mask_bladder=mask_bladder,
                    **hparams)
            )
            # Remove groupby index
            df_preds = df_preds.reset_index(drop=True)
        else:
            filenames = df_metadata.filename.tolist()
            df_preds = predict_on_images(
                model=model,
                filenames=filenames,
                img_dir=None,
                mask_bladder=mask_bladder,
                **hparams)

    # Join to metadata. NOTE: This works because of earlier sorting
    df_metadata = pd.concat([df_metadata, df_preds], axis=1)

    # 5. Save predictions on dset split
    df_metadata.to_csv(pred_save_path, index=False)


def embed_dset(exp_name, dset=constants.DEFAULT_EVAL_DSET, **overwrite_hparams):
    """
    Extract embeddings on dset.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.
    overwrite_hparams : dict, optional
        Keyword arguments to overwrite experiment hyperparameters

    Raises
    ------
    RuntimeError
        If `exp_name` does not lead to a valid training directory
    """
    # 0. Create path to save embeddings
    embed_save_path = embed.get_save_path(exp_name, dset=dset)

    # Early return, if embeddings already made
    if os.path.isfile(embed_save_path):
        return

    # 0. Get experiment directory, where model was trained
    model_dir = load_model.get_exp_dir(exp_name)

    # 1 Get experiment hyperparameters
    hparams = load_model.get_hyperparameters(model_dir)
    hparams.update(overwrite_hparams)
    # NOTE: Ensure full image path is saved in embedding file
    hparams["full_path"] = True

    # 2. Load existing model and send to device
    model = load_model.load_pretrained_from_exp_name(
        exp_name, overwrite_hparams=overwrite_hparams)
    model = model.to(DEVICE)

    # 3. Load data
    # NOTE: Ensure data is loaded in the non-SSL mode
    dm = load_data.setup_data_module(hparams, self_supervised=False)

    # 4. Create a DataLoader
    if dset == "test":
        dataloader = dm.test_dataloader()
    elif dset == "val":
        dataloader = dm.val_dataloader()
    else:
        dataloader = dm.train_dataloader()

    # 5. Extract embeddings and save them
    embed.extract_embeds(
        model,
        save_embed_path=embed_save_path,
        img_dataloader=dataloader,
        device=DEVICE)


def analyze_dset_preds(exp_name, dset=constants.DEFAULT_EVAL_DSET,
                       mask_bladder=False):
    """
    Analyze dset split predictions.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.
    mask_bladder : bool, optional
        If True, mask out predictions on bladder/middle side, leaving only
        kidney labels. Defaults to False.

    Raises
    ------
    RuntimeError
        If `exp_name` does not lead to a valid training directory
    """
    # 0. Get experiment directory, where model was trained
    model_dir = load_model.get_exp_dir(exp_name)

    # 1 Get experiment hyperparameters
    hparams = load_model.get_hyperparameters(model_dir)

    # If specified, calculate metrics
    if CALCULATE_METRICS:
        # If 2+ dsets provided, calculate metrics on each dset individually
        dsets = [dset] if isinstance(dset, str) else dset
        for dset_ in dsets:
            try:
                calculate_exp_metrics(
                    exp_name=exp_name,
                    dset=dset_,
                    hparams=hparams,
                    mask_bladder=mask_bladder)
            except KeyError:
                # 1. Perform inference
                infer_dset(
                    exp_name=exp_name, dset=dset_,
                    mask_bladder=dset_ in constants.DSETS_MISSING_BLADDER,
                    overwrite_existing=True,
                    **load_data.create_overwrite_hparams(dset_))
                # 2. Attempt to calculate metrics again
                calculate_exp_metrics(
                    exp_name=exp_name,
                    dset=dset_,
                    hparams=hparams,
                    mask_bladder=mask_bladder)

    # If specified, create UMAP plots
    if EMBED:
        plot_umap.main(exp_name, dset=dset)

    # Close all open figures
    plt.close("all")


if __name__ == '__main__':
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Parse arguments
    ARGS = PARSER.parse_args()

    # For each experiment,
    for EXP_NAME in ARGS.exp_name:
        # Iterate over all specified eval dsets
        for DSET in ARGS.dset:
            # Specify to mask bladder, if it's a hospital w/o bladder labels
            MASK_BLADDER = DSET in constants.DSETS_MISSING_BLADDER

            # 2. Create overwrite parameters
            OVERWRITE_HPARAMS = load_data.create_overwrite_hparams(DSET)

            # 3. Perform inference
            infer_dset(exp_name=EXP_NAME, dset=DSET,
                       mask_bladder=MASK_BLADDER,
                       **OVERWRITE_HPARAMS)

            # 4. Extract embeddings
            if EMBED:
                embed_dset(exp_name=EXP_NAME, dset=DSET, **OVERWRITE_HPARAMS)

            # # 5. Evaluate predictions and embeddings
            analyze_dset_preds(exp_name=EXP_NAME, dset=DSET,
                               mask_bladder=MASK_BLADDER)

        # TODO: 6. Create UMAPs embeddings on all dsets together
        analyze_dset_preds(exp_name=EXP_NAME, dset=ARGS.dset)
