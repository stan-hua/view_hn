"""
model_eval.py

Description: Used to evaluate a trained model's performance on other view
             labeling datasets.
"""

# Standard libraries
import argparse
import logging
import os
import random
import sys
from collections import OrderedDict
from functools import partial

# Non-standard libraries
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms.v2 as T
from arch.bootstrap import IIDBootstrap
from albumentations.pytorch.transforms import ToTensorV2
from colorama import Fore, Style
from scipy.stats import pearsonr
from sklearn import metrics as skmetrics
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

# Custom libraries
from src.data import constants
from src.data_prep import utils
from src.data_viz import plot_umap
from src.data_viz import utils as viz_utils
from src.scripts import embed, load_model, load_data
from src.utils.logging import load_comet_logger

# Configure seaborn color palette
sns.set_palette("Paired")

# Add progress_apply to pandas
tqdm.pandas()

################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

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

# Flag to overwrite existing results
OVERWRITE_EXISTING = False

# Flag to force bladder masking for US image datasets
FORCE_MASK_BLADDER = False


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
        "exp_names": "Name/s of experiment/s (to evaluate)",
        "dsets": "List of dataset names to evaluate",
        "splits": "Name of data splits for each `dset` to evaluate",
        "ckpt_option": "Choice of checkpoint to load (last/best)",
        "mask_bladder": "If True, mask bladder logit, if it's an US image dataset.",
        "test_time_aug": "If True, perform test-time augmentations during inference.",
        "da_transform_name":
            "If provided, performs domain adaptation transform on test images, "
            "by default None. Must be one of ('fda', 'hm')",
        "log_to_comet": "If True, log results to Comet ML",
    }
    parser.add_argument("--exp_names", required=True,
                        nargs='+',
                        help=arg_help["exp_names"])
    parser.add_argument("--dsets", default=["sickkids"],
                        nargs='+',
                        help=arg_help["dsets"])
    parser.add_argument("--splits", default=["test"],
                        nargs='+',
                        help=arg_help["splits"])
    parser.add_argument("--ckpt_option", default="best",
                        help=arg_help["ckpt_option"])
    parser.add_argument("--mask_bladder", action="store_true",
                        help=arg_help["mask_bladder"])
    parser.add_argument("--test_time_aug", action="store_true",
                        help=arg_help["test_time_aug"])
    parser.add_argument("--da_transform_name", default=None,
                        help=arg_help["da_transform_name"])
    parser.add_argument("--log_to_comet", action="store_true",
                        help=arg_help["log_to_comet"])


################################################################################
#                             Inference - Related                              #
################################################################################
@torch.no_grad()
def predict_on_images(model, filenames, labels=None,
                      img_dir=constants.DSET_TO_IMG_SUBDIR_FULL["sickkids"],
                      mask_bladder=False,
                      test_time_aug=False,
                      da_transform=None,
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
    labels : list of str
        String label for each image
    img_dir : str, optional
        Path to directory containing images, by default constants.DSET_TO_IMG_SUBDIR_FULL["sickkids"]
    mask_bladder : bool, optional
        If True, mask out predictions on bladder/middle side, leaving only
        kidney labels. Defaults to False.
    test_time_aug : bool, optional
        If True, perform test-time augmentation.
    da_transform : A.Compose, optional
        If provided, contains Domain Adaptation transform and ensures it's
        converted back to a PyTorch tensor, by default None.
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
    class_to_idx = constants.LABEL_PART_TO_CLASSES[label_part]["class_to_idx"]

    # Set to evaluation mode
    model.eval()

    # Predict on each images one-by-one
    preds = []
    probs = []
    outs = []
    losses = []

    for idx, filename in tqdm(enumerate(filenames)):
        img_path = filename if img_dir is None else f"{img_dir}/{filename}"

        # Load image as expected by model
        # NOTE: Load as numpy, if passing into da_transform
        as_numpy = False if da_transform is None else True
        img = load_image(
            img_path,
            augment=test_time_aug,
            n=8 if test_time_aug else 1,
            as_numpy=as_numpy,
        )

        # Pass through da transform
        if da_transform is not None:
            img = da_transform(image=img)["image"]
            # Add batch size dimension back
            img = img.unsqueeze(0)

        # Convert to float32 type and send to device
        img = img.to(torch.float32).to(DEVICE)

        # Perform inference
        out = model(img)

        # If specified, remove Bladder/None as a possible prediction
        # NOTE: Assumes model predicts bladder/none as the 3rd index
        if mask_bladder:
            out = out[:, :2]

        # If test-time augmentation, averaging output across augmented samples
        out = out.mean(axis=0, keepdim=True)

        # Compute loss, if label provided
        loss = None
        if labels:
            label_idx = class_to_idx[labels[idx]]
            label = torch.LongTensor([label_idx]).to(out.device)
            loss = round(float(torch.nn.functional.cross_entropy(out, label).detach().cpu().item()), 4)
        losses.append(loss)

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
        "loss": losses,
    })

    return df_preds


# TODO: Implement test-time augmentations
@torch.no_grad()
def predict_on_sequences(model, filenames,
                         img_dir=constants.DSET_TO_IMG_SUBDIR_FULL["sickkids"],
                         mask_bladder=False,
                         test_time_aug=False,
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
        Path to directory containing images, by default constants.DSET_TO_IMG_SUBDIR_FULL["sickkids"]
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
    if test_time_aug:
        raise NotImplementedError

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
        img = transform_image(img).squeeze(0)
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
def multi_predict_on_sequences(model, filenames, img_dir=constants.DSET_TO_IMG_SUBDIR_FULL["sickkids"],
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
        Path to directory containing images, by default constants.DSET_TO_IMG_SUBDIR_FULL["sickkids"]
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
        img = transform_image(img).squeeze(0)
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
    df_labels = df_pred.groupby(by=["label"]).apply(
        get_counts_and_prop,
        include_groups=True,
    )

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
    df_count = df_pred.groupby(by=[col]).apply(len)
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
    label_seqs = df_seqs.apply(_get_label_sequence, include_groups=True)
    label_seqs = label_seqs.map("".join)
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
    label_seqs = df_seqs.apply(_get_label_sequence, include_groups=True)
    label_seqs = label_seqs.map("".join)
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
    for side in ("Left", "Right", "Bladder"):
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

    # 1. Accuracy by class
    unique_labels = sorted(df_pred["label"].unique())
    for label in unique_labels:
        df_pred_filtered = df_pred[df_pred.label == label]
        metrics[f"Label Accuracy ({label})"] = 0
        if not df_pred_filtered.empty:
            metrics[f"Label Accuracy ({label})"] = scale_and_round(calculate_accuracy(
                df_pred_filtered))

    # 2. Overall accuracy
    metrics["Overall Accuracy"] = scale_and_round(calculate_accuracy(df_pred))
    # Bootstrap confidence interval
    if ci:
        point, (lower, upper) = bootstrap_metric(
            df_pred=df_pred,
            metric_func=skmetrics.accuracy_score,
            **ci_kwargs)
        point, lower, upper = scale_and_round(point), scale_and_round(lower), scale_and_round(upper)
        metrics["Overall Accuracy"] = f"{point} [{lower}, {upper}]"

    # 3. Accuracy, grouped by patient
    metrics["Accuracy (By Patient)"] = \
        calculate_metric_by_groups(df_pred, ["id"])

    # 4. Accuracy, grouped by sequence
    metrics["Accuracy (By Seq)"] = \
        calculate_metric_by_groups(df_pred, ["id", "visit"])

    # 5. Accuracy for adjacent same-label images vs. stand-alone images
    mask_same_label_adjacent = identify_repeating_same_label_in_video(df_pred)
    metrics["Accuracy (Adjacent to Same-Label)"] = scale_and_round(calculate_accuracy(
        df_pred[mask_same_label_adjacent]))
    metrics["Accuracy (Not Adjacent to Same-Label)"] = scale_and_round(calculate_accuracy(
        df_pred[~mask_same_label_adjacent if len(mask_same_label_adjacent)
                else []]))

    # 6. F1 Score by class
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
    # Skip, if empty
    if df_pred.empty:
        return "N/A"

    # Calculate metric on each group
    grp_metrics = df_pred.groupby(by=group_cols).apply(
        lambda df_grp: metric_func(df_grp["label"], df_grp["pred"]),
        include_groups=True,
    )

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
                     label_col="label", pred_col="pred",
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
    label_col : str, optional
        Name of label column, by default "label"
    pred_col : str, optional
        Name of label column, by default "pred"
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
    exact_metric = round(metric_func(df_pred[label_col], df_pred[pred_col]), 4)

    # Initialize bootstrap
    bootstrap = IIDBootstrap(
        df_pred[label_col], df_pred[pred_col],
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
    # Accumulate metric table columns
    accum_metric_tables = []

    # 1. Calculate metrics
    # 1.1.1 For all samples
    df_metrics_all = calculate_metrics(df_pred, ci=True)
    df_metrics_all.name = "All"
    accum_metric_tables.append(df_metrics_all)

    # 0. Filler columns
    filler = df_metrics_all.copy()
    filler[:] = ""
    filler.name = ""

    # 1.2 Stratify patients w/ HN and w/o HN
    if "hn" in df_pred.columns:
        df_metrics_w_hn = calculate_metrics(df_pred[df_pred.hn == 1])
        df_metrics_w_hn.name = "With HN"
        df_metrics_wo_hn = calculate_metrics(df_pred[df_pred.hn == 0])
        df_metrics_wo_hn.name = "Without HN"

        # Update accumulator
        accum_metric_tables.append(filler)
        accum_metric_tables.append(df_metrics_w_hn)
        accum_metric_tables.append(df_metrics_wo_hn)

    # 1.3 For images at label boundaries
    # NOTE: Disabled for now
    # df_metrics_at_boundary = calculate_metrics(
    #     df_pred[df_pred["at_label_boundary"]])
    # df_metrics_at_boundary.name = "At Label Boundary"

    # 1.4 Most confident sample for each predicted view label across sequence
    df_confident = filter_most_confident(df_pred, local=False)
    df_metrics_confident = calculate_metrics(df_confident, ci=True)
    df_metrics_confident.name = "Most Confident"
    accum_metric_tables.append(df_metrics_confident)

    # 1.5 If bladder present, re-compute metrics without Bladder images
    # NOTE: Doesn't remove non-Bladder images misclassified as Bladder
    for bladder_label in ["None", "Bladder"]:
        if bladder_label not in df_pred["label"].unique():
            continue

        # Recalculate metrics
        df_pred_wo_bladder = df_pred[df_pred["label"] != bladder_label]
        df_metrics_wo_bladder = calculate_metrics(df_pred_wo_bladder, ci=True)
        df_metrics_wo_bladder.name = "Without Bladder"

        # Update accumulator
        accum_metric_tables.append(filler)
        accum_metric_tables.append(df_metrics_wo_bladder)

    # 2. Combine
    df_metrics = pd.concat(accum_metric_tables, axis=1)

    return df_metrics


def eval_create_plots(df_pred, hparams, inference_dir, dset, split,
                      comet_logger=None):
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
        Name of dataset
    split : str, optional
        Specific split of dataset. One of (train/val/test)
    comet_logger : comet_ml.ExistingExperiment
        If provided, log figures to Comet ML
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
    plt.savefig(os.path.join(inference_dir, f"{dset}-{split}_confusion_matrix.png"))

    if comet_logger is not None:
        comet_logger.log_figure(f"{dset}-{split}_confusion_matrix.png", plt.gcf(),
                                overwrite=True)

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


def calculate_exp_metrics(exp_name, dset, split, hparams=None,
                          log_to_comet=False,
                          **infer_kwargs):
    """
    Given that inference was performed, compute metrics for experiment model
    and dataset.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Name of dataset
    split : str, optional
        Specific split of dataset. One of (train/val/test)
    hparams : dict, optional
        Experiment hyperparameters, by default None
    log_to_comet : bool, optional
        If True, log metrics and graphs to Comet ML
    **infer_kwargs : Keyword arguments
        Inference keyword arguments, which includes
            mask_bladder : bool, optional
                If True, bladder predictions are masked out, by default False
            test_time_aug : bool, optional
                If True, perform test-time augmentation.
    """
    # 1. Get experiment hyperparameters (if not provided)
    hparams = hparams if hparams \
        else load_model.get_hyperparameters(exp_name=exp_name)

    # 1.1 Get comet logger, if specified
    comet_logger = None
    if log_to_comet:
        comet_logger = load_comet_logger(exp_key=hparams.get("comet_exp_key"))

    # 2. Load inference
    df_pred = load_view_predictions(exp_name, dset=dset, split=split,
                                    **infer_kwargs)

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

        # Experiment-specific inference directory, to store figures
        inference_dir = create_save_dir_by_flags(
            exp_name,
            dset=dset,
            **infer_kwargs,
        )

        # 4. Calculate metrics
        df_metrics = eval_calculate_all_metrics(df_pred)
        df_metrics.to_csv(os.path.join(inference_dir,
                                        f"{dset}-{split}_metrics.csv"))

        # 4.1 Store metrics in Comet ML, if possible
        if comet_logger is not None:
            suffix = "(mask_bladder)" if infer_kwargs.get("mask_bladder", False) else ""
            comet_logger.log_table(f"{dset}-{split}_metrics{(suffix)}.csv", df_metrics)

        # 5. Create plots for visual evaluation
        eval_create_plots(df_pred, hparams, inference_dir, dset=dset, split=split)


def store_example_classifications(exp_name, dset, split, mask_bladder=False,
                                  save_dir=constants.DIR_FIGURES_PRED):
    """
    Store correctly/incorrectly classified images in nested folders.

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
    save_dir : str, optional
        Directory to save model classifications, by default
        constants.DIR_FIGURES_PRED
    """
    # Load inference
    save_path = create_save_path(exp_name, dset=dset, split=split,
                                 mask_bladder=mask_bladder)
    df_pred = pd.read_csv(save_path)

    # Sort by video
    df_pred = df_pred.sort_values(by=["id", "visit", "seq_number"])
    # Get unique labels
    labels = df_pred["label"].unique().tolist()

    mask_same_label = identify_repeating_same_label_in_video(df_pred)

    df_pred = df_pred[~mask_same_label]

    # Create sub-folders for experiment and its correct/incorrect predictions
    exp_dir = os.path.join(save_dir, exp_name, dset)
    correct_dir = os.path.join(exp_dir, "standalone", "correct")
    incorrect_dir = os.path.join(exp_dir, "standalone", "incorrect")
    for subdir in (correct_dir, incorrect_dir):
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    # For each label, stratify by correct and incorrect predictions
    for label in labels:
        # Filter for correct/incorrectly predicted
        df_label = df_pred[df_pred.label == label]
        df_correct = df_label[df_label.label == df_label.pred]
        df_incorrect = df_label[df_label.label != df_label.pred]

        # Sample at most 9 images to plot
        df_correct = df_correct.sample(n=min(9, len(df_correct)))
        df_incorrect = df_incorrect.sample(n=min(9, len(df_incorrect)))

        # Get paths to images
        correct_paths = df_correct.filename.tolist()
        incorrect_paths = df_incorrect.filename.tolist()

        # Create gridplot from sampled images
        viz_utils.gridplot_images_from_paths(
            correct_paths,
            filename=f"{label} (correct).png",
            save_dir=correct_dir,
            title=f"Classified ({label})",
        )

        viz_utils.gridplot_images_from_paths(
            incorrect_paths,
            filename=f"{label} (misclassified).png",
            save_dir=incorrect_dir,
            title=f"Misclassified ({label})",
        )


def get_highest_loss_samples(df_pred, n=100):
    """
    Get predicted samples with the highest loss.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Each row is a prediction
    n : int, optional
        Number of samples per label
        
    Returns
    -------
    pd.DataFrame
        Table with sampled highest losses per label
    """
    df_samples = df_pred.groupby(by=["label"]).apply(
        lambda df: df.sort_values(by=["loss"], ascending=False).iloc[:n]
    ).reset_index(drop=True)
    return df_samples


################################################################################
#                               Helper Functions                               #
################################################################################
def load_image(img_path, as_numpy=False, **transform_hparams):
    """
    Load image as tensor or numpy

    Parameters
    ----------
    img_path : str
        Path to image
    as_numpy : bool, optional
        If True, return numpy array, by default False

    Returns
    -------
    torch.Tensor or np.ndarray (if as_numpy)
        Loaded and transformed image (images, if transform returns multiple images)
    """
    # Load and transform image
    img = read_image(img_path, ImageReadMode.RGB)
    img = transform_image(
        img,
        **transform_hparams
    )

    # If specified, convert to numpy and ensure channels are last
    if as_numpy:
        # Attempt to remove empty first dimension
        img = img.squeeze(0)

        # CASE 1: Multiple images returned as a result of augmentation
        if len(img.shape) == 4:
            img = img.numpy().transpose(0, 2, 3, 1)
        # CASE 2: Only single image as expected
        else:
            assert len(img.shape) == 3
            img = img.numpy().transpose(1, 2, 0)

    return img


def transform_image(img, augment=False, n=1, hparams=None):
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
    augment : bool, optional
        If True, perform augmentations during training
    n : int, optional
        If augmenting, the number of times to randomly augment the image

    Returns
    -------
    np.array
        Transformed image/s with extra first dimension, useful if n > 1.
    """
    # Get parameters (or defaults)
    hparams = hparams or {}
    img_size = hparams.get("img_size", constants.IMG_SIZE)
    crop_scale = hparams.get("crop_scale", 0.2)

    # Convert to torch
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    transforms = []
    # Add resize transform
    transforms.append(T.Resize(img_size))

    # If specified, add training augmentations
    if augment:
        transforms.append(T.Compose(list(utils.prep_strong_augmentations(
            img_size=img_size,
            crop_scale=crop_scale).values())))
    transforms = T.Compose(transforms)

    # Get RNG state (for later restoration)
    random_state = torch.get_rng_state()
    # Set random seed
    torch.seed()

    # Augment each image separately
    img_stack = torch.stack([transforms(img) for _ in range(n)])

    # Normalize between 0 and 1
    img_stack = img_stack.to(torch.float32) / 255.

    # Restore original RNG state
    torch.set_rng_state(random_state)

    return img_stack


def get_da_transform(da_transform_name, src_paths):
    """
    Return domain adaptation (DA) transform

    Note
    ----
    Ensures that it's converted to a PyTorch tensor post-transform

    Parameters
    ----------
    da_transform_name : str
        Name of DA transform
    src_paths : list
        List of source dataset paths

    Returns
    -------
    albumentations.Compose
        Composed DA transform
    """
    # Early return identity, if none
    if da_transform_name is None:
        return T.Identity()

    # Ensure valid transform is provided
    if da_transform_name not in ("fda", "hm"):
        raise RuntimeError(f"`da_transform_name` must be in {('fda', 'hm')}")

    # Load transform
    # CASE 1: Fourier Domain Adaptation
    if da_transform_name == "fda":
        da_transform = A.FDA(
            src_paths,
            beta_limit=0.1,
            p=1,
            read_fn=partial(load_image, as_numpy=True))
    # CASE 2: Histogram Matching
    elif da_transform_name == "hm":
        da_transform = A.HistogramMatching(
            src_paths,
            blend_ratio=(1, 1),
            p=1,
            read_fn=partial(load_image, as_numpy=True))

    # Compose transform
    # NOTE: Ensure converted to PyTorch tensor after
    transforms = A.Compose([da_transform, ToTensorV2()])

    return transforms


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


def identify_repeating_segments(values):
    """
    Create boolean mask, where True is given to list values adjacent to the same
    value.

    Parameters
    ----------
    values: list
        List of values

    Returns
    -------
    np.array
        Boolean mask of repeating value segments
    """
    mask = []
    prev_val = None
    segment_len = 1

    for val in values:
        # CASE 0: Part of repeating segment
        if val == prev_val:
            segment_len += 1
            continue

        # CASE 1: New value encountered
        # CASE 1.0: Default value (to skip)
        if prev_val is None:
            prev_val = val
            continue

        # CASE 1.1: Extend mask
        mask.extend([(segment_len > 1)] * segment_len)

        # Update accumulators
        segment_len = 1
        prev_val = val

    # Handle last segment
    if prev_val is not None:
        mask.extend([(segment_len > 1)] * segment_len)

    # Ensure mask is the same length as values
    assert len(mask) == len(values)

    return np.array(mask)


def filter_most_confident(df_pred, local=False, top_k=1, groupby="pred"):
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
        consecutive images with the same label. Otherwise, aggregates by
        predicted view label to find the most confident view label predictions,
        by default False.
    top_k : int, optional
        Get top K most confident view label predictions, by default 1.
    groupby : str, optional
        If `local=False`, "pred" would find the most confident view prediction
        (irrespective of the ground truth), and "label" would find the most
        confident prediction grouping by the ground-truth label

    Returns
    -------
    pandas.DataFrame
        Filtered predictions
    """
    df_pred = df_pred.copy()

    # CASE 1: For each sequence, get the strongest predicted view
    if not local:
        # Get most confident pred per view per sequence (ignoring seq. number)
        assert groupby in ("pred", "label")
        df_seqs = df_pred.groupby(by=["id", "visit", groupby])
        df_filtered = df_seqs.apply(
            lambda df: df.nlargest(top_k, "out"),
            include_groups=True).reset_index(drop=True)
    # CASE 2: 
    else:
        # Get most confident pred per group of consecutive labels per sequence
        # 0. Sort by id, visit and sequence number
        df_pred = df_pred.sort_values(by=["id", "visit", "seq_number"])

        # 1. Identify local groups of consecutive labels
        local_grps_per_seq = df_pred.groupby(by=["id", "visit"]).apply(
            lambda df: get_local_groups(df.label.values),
            include_groups=True,
        )
        df_pred["local_group"] = np.concatenate(local_grps_per_seq.values)

        df_seqs = df_pred.groupby(by=["id", "visit", "local_group"])
        df_filtered = df_seqs.apply(
            lambda df: df[df.out == df.out.max()],
            include_groups=True,
        )

    return df_filtered


def identify_repeating_same_label_in_video(df_pred, repeating=True):
    """
    Given predictions on US videos, identify images which have consecutive
    images with the same label.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    repeating : bool, optional
        If True, filter for repeating same label frames. Otherwise, filter for
        frames that are NOT adjacent to frames of the same label, by default
        True.

    Returns
    -------
    np.array
        Boolean mask to keep images that ARE or ARE NOT adjacent to
        same-label image frames
    """
    # PRECONDITION: Index must be unique
    assert not df_pred.index.duplicated().any(), "Duplicate index value found!"

    # Early return, if empty
    if df_pred.empty:
        return np.array([])

    # Create copy and ensure videos are in order
    df_pred = df_pred.sort_values(by=["id", "visit", "seq_number"])
    # Reset index (necessary for reordering)
    df_pred = df_pred.reset_index(drop=True)

    # Get boolean mask for images adjacent to images of the same label
    mask = np.concatenate(df_pred.groupby(by=["id", "visit"]).apply(
        lambda df: identify_repeating_segments(df.label.tolist())).to_numpy())

    # If specified, only get those that are NOT adjacent to same-label images
    if not repeating:
        mask = ~mask

    # Reorder mask to match index
    mask = mask[df_pred.index]

    return mask


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
        side_to_idx = {"First": "1", "Second": "2", "Bladder": "-",}
    else:
        side_to_idx = {"Left": "1", "Right": "2", "Bladder": "-",}

    # How to determine side index
    side_func = lambda x: side_to_idx[utils.extract_from_label(x, "side")]
    if label_part == "side":
        side_func = lambda x: side_to_idx[x]
    else:
        LOGGER.warning("Invalid `label_part` provided in `show_example_side_predictions`! Skipping...")
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


def create_save_dir_by_flags(exp_name, dset=constants.DEFAULT_EVAL_SPLIT,
                             **extra_flags):
    """
    Create directory to save dset predictions, based on experiment name and
    keyword arguments

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Name of dataset
    **extra_flags : dict, optional
        Keyword arguments, specifying extra flags used during inference

    Returns
    -------
    str
        Expected directory to save dset predictions
    """
    # Add mask bladder, if dset doesn't contain bladders
    if FORCE_MASK_BLADDER and dset in constants.DSETS_MISSING_BLADDER:
        extra_flags["mask_bladder"] = True

    # Add true flags to the experiment name
    for flag, val in extra_flags.items():
        # Skip, if false-y
        if not val:
            continue

        # CASE 1: String
        if isinstance(val, str):
            exp_name += f"__{val}"
        # CASE 2: Boolean or other
        else:
            exp_name += f"__{flag}"

    # Create inference directory, if not exists
    inference_dir = os.path.join(constants.DIR_INFERENCE, exp_name)
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    return inference_dir


def create_save_path(exp_name, dset, split,
                     **extra_flags):
    """
    Create file path to dset predictions, based on experiment name and keyword
    arguments

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Name of dataset
    split : str, optional
        Specific split of dataset. One of (train/val/test)
    **extra_flags : dict, optional
        Keyword arguments, specifying extra flags used during inference

    Returns
    -------
    str
        Expected path to dset predictions
    """
    # Create inference directory path
    inference_dir = create_save_dir_by_flags(exp_name, dset, **extra_flags)

    # Expected path to dset inference
    fname = f"{dset}-{split}_set_results.csv"
    save_path = os.path.join(inference_dir, fname)

    return save_path


def calculate_per_seq_silhouette_score(exp_name, dset, split,
                                       label_part="side",
                                       exclude_labels=("Bladder", "Other")):
    """
    Calculate a per - ultrasound sequence Silhouette score.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Name of dataset
    split : str, optional
        Specific split of dataset. One of (train/val/test)
    label_part : str, optional
        If specified, either `side` or `plane` is extracted from each label
        and used as the given label, by default "side"
    exclude_labels : list or array-like, optional
        List of labels whose matching samples will be excluded when calculating
        the Silhouette score, by default ("Bladder",)

    Returns
    -------
    float
        Mean Silhouette Score across unique ultrasound sequences
    """
    # Load embeddings
    df_embeds = embed.get_embeds(exp_name, dset=dset, split=split)
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


def calculate_accuracy(df_pred, label_col="label", pred_col="pred"):
    """
    Given a table of predictions with columns "label" and "pred", compute
    accuracy rounded to 4 decimal places.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    label_col : str, optional
        Name of label column, by default "label"
    pred_col : str, optional
        Name of label column, by default "pred"

    Returns
    -------
    float
        Accuracy rounded to 4 decimal places
    """
    # Early return, if empty
    if df_pred.empty:
        return "N/A"

    # Compute accuracy
    acc = skmetrics.accuracy_score(df_pred[label_col], df_pred[pred_col])

    # Round decimals
    acc = round(acc, 4)

    return acc


def load_view_predictions(exp_name, dset, split, **infer_kwargs):
    """
    Load predictions by model given by `exp_name` on dataset `dset`.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str
        Name of evaluation split or test dataset
    **infer_kwargs : Keyword arguments which includes:
        mask_bladder : bool, optional
            If True, bladder predictions are masked out, by default False
        test_time_aug : bool, optional
            If True, perform test-time augmentation.

    Returns
    -------
    pd.DataFrame
        Each row is an image with a view label (plane/side) predicted.
    """
    # 1. Specify path to inference file
    save_path = create_save_path(exp_name, dset=dset, split=split, **infer_kwargs)
    # Raise error, if predictions not found
    if not os.path.exists(save_path):
        raise RuntimeError(f"Predictions not found on dataset {dset}-{split}!\n"
                           f"`exp_name`: {exp_name}")

    # 2. Load results
    df_pred = pd.read_csv(save_path)

    # 3. Ensure no duplicates (or sequential data only)
    df_pred = df_pred.drop_duplicates(subset=["id", "visit", "seq_number", "filename"])

    # 4. Add side/plane label, if not present
    for label_part in constants.LABEL_PARTS:
        if label_part not in df_pred.columns:
            df_pred[label_part] = utils.get_labels_for_filenames(
                df_pred["filename"].tolist(), label_part=label_part)

    # 5. Add HN labels, if not already exists. NOTE: Needs side label to work
    if "hn" not in df_pred.columns:
        df_pred = utils.extract_hn_labels(df_pred)

    # 6. Specify which images are at label boundaries
    # NOTE: Disabled for now
    # df_pred["at_label_boundary"] = utils.get_label_boundaries(df_pred)

    return df_pred


def load_side_plane_view_predictions(side_exp_name, plane_exp_name, dset, split,
                                     mask_bladder=False):
    """
    Load view label predictions for side and plane model.

    Parameters
    ----------
    side_exp_name : str
        Name of side experiment
    plane_exp_name : str
        Name of plane experiment
    dset : str
        Name of evaluation dataset
    split : str
        Name of data split corresponding to dataset
    mask_bladder : bool, optional
        If True, bladder predictions are masked out, by default False

    Returns
    -------
    pd.DataFrame
        Contains view label predictions (side and plane) for dataset
    """
    # Load side predictions
    if side_exp_name != "canonical":
        df_side_preds = load_view_predictions(
            side_exp_name, dset=dset, split=split, mask_bladder=mask_bladder)
        # Rename columns
        df_side_preds = df_side_preds.rename(columns={
            "label": "side",
            "pred": "side_pred",
            "prob": "side_prob",
            "out": "side_out",
        })
        df_side_preds = df_side_preds.loc[:,~df_side_preds.columns.duplicated()]

        # Fix slashes in filenames
        df_side_preds["filename"] = df_side_preds["filename"].map(os.path.normpath)

    # Load plane predictions
    if plane_exp_name != "canonical":
        df_plane_preds = load_view_predictions(
            plane_exp_name, dset=dset, split=split, mask_bladder=mask_bladder)
        # Rename columns
        df_plane_preds = df_plane_preds.rename(columns={
            "label": "plane",
            "pred": "plane_pred",
            "prob": "plane_prob",
            "out": "plane_out",
        })
        df_plane_preds = df_plane_preds.loc[:,~df_plane_preds.columns.duplicated()]

        # Fix slashes in filenames
        df_plane_preds["filename"] = df_plane_preds["filename"].map(os.path.normpath)

    # CASE 1: Both experiments specified are valid models
    if side_exp_name != "canonical" and plane_exp_name != "canonical":
        # Merge predictions
        duplicate_suffix = "_duplicate"
        df_view_preds = df_side_preds.merge(
            df_plane_preds,
            how="inner",
            on=["filename"],
            suffixes=("", duplicate_suffix))

        # Remove duplicate columns
        cols = df_view_preds.columns.tolist()
        duplicate_cols = [col for col in cols if col.endswith(duplicate_suffix)]
        df_view_preds = df_view_preds.drop(columns=duplicate_cols)
    # CASE 2: Both are specified to use canonical labels
    elif side_exp_name == "canonical" and plane_exp_name == "canonical":
        raise RuntimeError("Case not handled!")
    # CASE 3: Use side labels
    elif side_exp_name == "canonical":
        df_view_preds = df_plane_preds
        df_view_preds["side_pred"] = df_view_preds["plane"]
        df_view_preds["side_prob"] = 1.
        df_view_preds["side_out"] = 1.
    # CASE 4: Use plane labels
    elif plane_exp_name == "canonical":
        df_view_preds = df_side_preds
        df_view_preds["plane_pred"] = df_view_preds["plane"]
        df_view_preds["plane_prob"] = 1.
        df_view_preds["plane_out"] = 1.

    return df_view_preds


def scale_and_round(x, factor=100, num_places=2):
    """
    Scale and round if value is an integer or float.

    Parameters
    ----------
    x : Any
        Any object
    factor : int
        Factor to multiply by
    num_places : int
        Number of decimals to round
    """
    if not isinstance(x, (int, float)):
        return x
    x = round(factor * x, num_places)
    return x


################################################################################
#                                  Main Flows                                  #
################################################################################
def infer_dset(exp_name, dset, split,
               ckpt_option="best",
               mask_bladder=False,
               test_time_aug=False,
               da_transform_name=None,
               overwrite_existing=OVERWRITE_EXISTING,
               **overwrite_hparams):
    """
    Perform inference on dset, and saves results

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Name of dataset
    split : str, optional
        Specific split of dataset. One of (train/val/test)
    ckpt_option : str, optional
        If "best", select based on validation loss. If "last", take last epoch.
    mask_bladder : bool, optional
        If True, mask out predictions on bladder/middle side, leaving only
        kidney labels. Defaults to False.
    test_time_aug : bool, optional
        If True, perform test-time augmentation.
    da_transform_name : str, optional
        If provided, performs domain adaptation transform on test images, by
        default None. Must be one of ("fda", "hm").
    overwrite_existing : bool, optional
        If True and prediction file already exists, overwrite existing, by
        default OVERWRITE_EXISTING.
    overwrite_hparams : dict, optional
        Keyword arguments to overwrite experiment hyperparameters

    Raises
    ------
    RuntimeError
        If `exp_name` does not lead to a valid training directory
    """
    # Assertion on `da_transform`
    assert da_transform_name in (None, "fda", "hm"), '`da_transform_name` must be one of (None, "fda", "hm")'

    # 0. Create path to save predictions
    pred_save_path = create_save_path(exp_name, dset=dset, split=split,
                                      mask_bladder=mask_bladder,
                                      test_time_aug=test_time_aug,
                                      da_transform_name=da_transform_name,
                                      ckpt_option=ckpt_option)
    # Early return, if prediction already made
    if os.path.isfile(pred_save_path) and not overwrite_existing:
        return

    # 0. Get experiment directory, where model was trained
    model_dir = load_model.get_exp_dir(exp_name)

    # 1. Get experiment hyperparameters
    hparams = load_model.get_hyperparameters(model_dir)

    # 2. If domain adaptation transform specified, load original training set
    #    images and create transform
    da_transform = None
    if da_transform_name is not None:
        # Get training data
        dm = load_data.setup_data_module(
            hparams, self_supervised=False, augment_training=False)
        df_train_metadata = dm.filter_metadata(dset="sickkids", split="train")

        # Filter for existing images
        exists_mask = df_train_metadata["filename"].map(os.path.exists)
        df_train_metadata = df_train_metadata[exists_mask]

        # Get 200 examples from each label
        src_paths = df_train_metadata.groupby(
            by=["label"])["filename"].sample(n=200).tolist()
        da_transform = get_da_transform(da_transform_name, src_paths)

    # Overwrite hyperparameters (can change dataset loaded)
    hparams.update(overwrite_hparams)

    # 4. Load existing model and send to device
    model = load_model.load_pretrained_from_exp_name(
        exp_name,
        ckpt_option=ckpt_option,
        overwrite_hparams=overwrite_hparams
    )
    model = model.to(DEVICE)

    # 3. Load data
    # NOTE: Ensure data is loaded in the non-SSL mode
    dm = load_data.setup_data_module(
        hparams, self_supervised=False, augment_training=False)

    # 3.1 Get metadata (for specified split)
    df_metadata = dm.filter_metadata(dset=dset, split=split)
    # 3.2 Sort, so no mismatch occurs due to groupby sorting
    df_metadata = df_metadata.sort_values(by=["id", "visit"], ignore_index=True)

    # 4. Predict on dset split
    # If multi-output model
    if hparams.get("multi_output"):
        if da_transform_name is not None:
            raise NotImplementedError(
                "Domain Adaptation transform is not implemented for "
                "multi-output model!")
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
            if da_transform_name is not None:
                raise NotImplementedError(
                    "Domain Adaptation transform is not implemented for "
                    "video model!")
            # CASE 1: Dataset of US videos
            if dset not in constants.DSETS_NON_SEQ:
                # Perform inference one sequence at a time
                df_preds = df_metadata.groupby(by=["id", "visit"]).\
                    progress_apply(
                        lambda df: predict_on_sequences(
                            model=model,
                            filenames=df.filename.tolist(),
                            img_dir=None,
                            mask_bladder=mask_bladder,
                            **hparams)
                    )
                # Remove groupby index
                df_preds = df_preds.reset_index(drop=True)
            # CASE 2: Dataset of US images
            else:
                df_preds = df_metadata.progress_apply(
                    lambda row: predict_on_sequences(
                        model=model,
                        filenames=[row.filename],
                        img_dir=None,
                        mask_bladder=mask_bladder,
                        **hparams),
                    axis=1,
                )
                df_preds = pd.concat(df_preds.tolist(), ignore_index=True)
        else:
            filenames = df_metadata.filename.tolist()
            df_preds = predict_on_images(
                model=model,
                filenames=filenames,
                labels=df_metadata["label"].tolist(),
                img_dir=None,
                mask_bladder=mask_bladder,
                test_time_aug=test_time_aug,
                da_transform=da_transform,
                **hparams)

    # Join to metadata. NOTE: This works because of earlier sorting
    df_metadata = pd.concat([df_metadata, df_preds], axis=1)

    # 5. Save predictions on dset split
    df_metadata.to_csv(pred_save_path, index=False)


def embed_dset(exp_name, dset, split,
               ckpt_option="best",
               overwrite_existing=OVERWRITE_EXISTING,
               **overwrite_hparams):
    """
    Extract embeddings on dset.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Name of dataset
    split : str, optional
        Specific split of dataset. One of (train/val/test)
    ckpt_option : str, optional
        If "best", select based on validation loss. If "last", take last epoch.
    overwrite_existing : bool, optional
        If True and embeddings already exists, overwrite existing, by
        default OVERWRITE_EXISTING.
    overwrite_hparams : dict, optional
        Keyword arguments to overwrite experiment hyperparameters

    Raises
    ------
    RuntimeError
        If `exp_name` does not lead to a valid training directory
    """
    # 0. Create path to save embeddings
    embed_save_path = embed.get_save_path(exp_name, dset=dset, split=split)

    # Early return, if embeddings already made
    if os.path.isfile(embed_save_path) and not overwrite_existing:
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
        exp_name,
        ckpt_option=ckpt_option,
        overwrite_hparams=overwrite_hparams
    )
    model = model.to(DEVICE)

    # NOTE: For non-video datasets, ensure each image is treated independently
    if dset in constants.DSETS_NON_SEQ:
        hparams["full_seq"] = False
        hparams["batch_size"] = 1

    # 3. Load data
    # NOTE: Ensure data is loaded in the non-SSL mode
    dm = load_data.setup_data_module(
        hparams, self_supervised=False, augment_training=False)
    dataloader = dm.get_filtered_dataloader(split=split, dset=dset)

    # 5. Extract embeddings and save them
    embed.extract_embeds(
        model,
        save_embed_path=embed_save_path,
        img_dataloader=dataloader,
        device=DEVICE)


def analyze_dset_preds(exp_name, dsets, splits,
                       log_to_comet=False,
                       **infer_kwargs):
    """
    Analyze dset split predictions.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dsets : str or list, optional
        Name of dataset
    splits : str or list, optional
        Specific split of dataset. One of (train/val/test)
    log_to_comet : bool, optional
        If True, log metrics and UMAPs to Comet ML.
    **infer_kwargs : Keyword arguments
        Inference keyword arguments, which includes
            mask_bladder : bool, optional
                If True, bladder predictions are masked out, by default False
            test_time_aug : bool, optional
                If True, perform test-time augmentation.

    Raises
    ------
    RuntimeError
        If `exp_name` does not lead to a valid training directory
    """
    # 0. Get experiment directory, where model was trained
    model_dir = load_model.get_exp_dir(exp_name)

    # 1 Get experiment hyperparameters
    hparams = load_model.get_hyperparameters(model_dir)

    # 2. If specified, calculate metrics
    if CALCULATE_METRICS:
        # If 2+ dsets provided, calculate metrics on each dset individually
        dsets = [dsets] if isinstance(dsets, str) else dsets
        splits = [splits] if isinstance(splits, str) else splits
        for idx, curr_dset in enumerate(dsets):
            curr_split = splits[idx]
            try:
                calculate_exp_metrics(
                    exp_name=exp_name,
                    dset=curr_dset,
                    split=curr_split,
                    hparams=hparams,
                    log_to_comet=log_to_comet,
                    **infer_kwargs,
                )
            except KeyError:
                # Log error
                LOGGER.error(KeyError)
                LOGGER.error("Try to perform inference again...")

    # 3. If specified, create UMAP plots
    if EMBED:
        plot_umap.main(exp_name, dset=dsets,
                       comet_exp_key=hparams.get("comet_exp_key") if log_to_comet else None)

    # Close all open figures
    plt.close("all")


def main(args):
    """
    Run main flows to:
        1. Perform inference
        2. Analyze predictions
    """
    exp_names = args.exp_names
    dsets = args.dsets
    splits = args.splits
    # If only one of dset/split is > 1, assume it's meant to be broadcast
    if len(dsets) == 1 and len(splits) > 1:
        LOGGER.info("Only 1 `dset` provided! Assuming same `dset` for all `splits`...")
        dsets = dsets * len(splits)
    if len(splits) == 1 and len(dsets) > 1:
        LOGGER.info("Only 1 `split` provided! Assuming same `split` for all `dsets`...")
        splits = splits * len(dsets)

    # For each experiment, perform inference and analyze results
    for exp_name in exp_names:
        # Iterate over all specified eval dsets
        for idx, curr_dset in enumerate(dsets):
            curr_split = splits[idx]

            # Specify to mask bladder, if it's a hospital w/o bladder labels
            if args.mask_bladder or FORCE_MASK_BLADDER:
                mask_bladder = curr_dset in constants.DSETS_MISSING_BLADDER
            else:
                mask_bladder = False

            # 2. Create overwrite parameters
            eval_hparams = load_data.create_eval_hparams(curr_dset, curr_split)

            # 3. Perform inference
            infer_kwargs = {
                "mask_bladder": mask_bladder,
                "test_time_aug": args.test_time_aug,
                "da_transform_name": args.da_transform_name,
                "ckpt_option": args.ckpt_option,
            }
            infer_dset(exp_name=exp_name, dset=curr_dset, split=curr_split,
                       **infer_kwargs,
                       **eval_hparams)

            # 4. Extract embeddings
            if EMBED:
                embed_dset(exp_name=exp_name, dset=curr_dset,
                           ckpt_option=args.ckpt_option,
                           **eval_hparams)

            # 5. Evaluate predictions and embeddings
            analyze_dset_preds(exp_name=exp_name,
                               dsets=curr_dset,
                               splits=curr_split,
                               log_to_comet=args.log_to_comet,
                               **infer_kwargs)

        # 6. Create UMAPs embeddings on all dsets together
        # NOTE: `mask_bladder` is not needed if combining all dsets
        infer_kwargs.pop("mask_bladder")
        analyze_dset_preds(exp_name=exp_name,
                           dsets=dsets,
                           splits=splits,
                           log_to_comet=args.log_to_comet,
                           **infer_kwargs)


if __name__ == '__main__':
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Parse arguments
    ARGS = PARSER.parse_args()

    # 2. Run main flows
    main(ARGS)
