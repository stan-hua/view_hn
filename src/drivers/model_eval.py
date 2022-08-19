"""
model_eval.py

Description: Used to evaluate a trained model's performance on the testing set.
"""

# Standard libraries
import logging
import os

# Non-standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as T
import yaml
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


# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure seaborn color palette
sns.set_palette("Paired")


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Flag to use GPU or not
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Type of models
MODEL_TYPES = ("five_view", "binary", "five_view_seq", "five_view_seq_w_other")

# Checkpoint for a trained 5-view model
CKPT_PATH_MULTI= constants.DIR_RESULTS + "/five_view/0/epoch=6-step=1392.ckpt"

# Checkpoint for a trained binary-classifier model
CKPT_PATH_BINARY = constants.DIR_RESULTS + \
    "/binary_classifier/0/epoch=12-step=2586.ckpt"

# Checkpoint for binary-classifier model trained with weighted loss
CKPT_PATH_BINARY_WEIGHTED = constants.DIR_RESULTS + \
    "/binary_classifier_weighted/0/epoch=11-step=2387.ckpt"

# Checkpoint for 5-view (sequential) CNN-LSTM model
CKPT_PATH_SEQUENTIAL = constants.DIR_RESULTS + \
    "cnn_lstm_8/0/epoch=31-step=5023.ckpt"

# Table to store/retrieve predictions and labels for test set
TEST_PRED_PATH = constants.DIR_RESULTS + "/test_set_results(%s).csv"

# NOTE: Chosen checkpoint and model type
CKPT_PATH = CKPT_PATH_SEQUENTIAL
MODEL_TYPE = MODEL_TYPES[-1]


################################################################################
#                             Inference - Related                              #
################################################################################
def get_test_set_metadata(df_metadata, hparams, dir=constants.DIR_IMAGES):
    """
    Get metadata table containing (filename, label) for each image in the test
    set.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    hparams : dict
        Hyperparameters used in model training run, used to load exact test set.
    dir : str, optional
        Path to directory containing metadata, by default constants.DIR_IMAGES

    Returns
    -------
    pandas.DataFrame
        Metadata of each image in the test set
    """
    # Set up data
    dm = UltrasoundDataModule(df=df_metadata, dir=dir, **hparams)
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


def predict_on_images(model, filenames, dir=constants.DIR_IMAGES):
    """
    Performs inference on images specified. Returns predictions, probabilities
    and raw model output.

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch model.
    filenames : np.array or array-like
        Filenames (or full paths) to images to infer on.
    dir : str, optional
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
            img_path = filename if dir is None else f"{dir}/{filename}"

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


def predict_on_sequences(model, filenames, dir=constants.DIR_IMAGES):
    """
    Performs inference on a full ultrasound sequence specified. Returns
    predictions, probabilities and raw model output.

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch model.
    filenames : np.array or array-like
        Filenames (or full paths) to images from one unique sequence to infer.
    dir : str, optional
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
            img_path = filename if dir is None else f"{dir}/{filename}"

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
def plot_confusion_matrix(df_pred, filter_confident=False):
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
    """
    # If flagged, get for most confident pred. for each view label per seq.
    if filter_confident:
        df_pred = filter_most_confident(df_pred)

    cm = confusion_matrix(df_pred["label"], df_pred["pred"],
                          labels=constants.CLASSES)
    disp = ConfusionMatrixDisplay(cm, display_labels=constants.CLASSES)
    disp.plot()
    plt.tight_layout()
    plt.show()


def plot_pred_probability_by_views(df_pred):
    """
    Plot average probability of prediction per view label.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
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
    
    # Bar plot
    sns.barplot(data=df_prob_by_view, x="View", y="Probability",
                order=constants.CLASSES)
    plt.show()


def check_misclassifications(df_pred):
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
    """
    # Filter for most confident predictions for each expected view label
    df_filtered = filter_most_confident(df_pred, local=True)

    # Get misclassified instances
    df_misclassified = df_filtered[(df_filtered.label != df_filtered.pred)]

    # 1. Proportion of misclassification from wrong body side
    prop_swapped = df_misclassified.apply(
        lambda row: row.label.split("_")[0] == row.pred.split("_")[0],
        axis=1).mean()

    # 2. Proportion of misclassification from adjacent views
    prop_adjacent = df_misclassified.apply(
        lambda row: row.pred in constants.LABEL_ADJACENCY[row.label],
        axis=1).mean()

    # Format output
    df_results = pd.Series({
        "Wrong Side": prop_swapped,
        "Adjacent Label": prop_adjacent,
        "Other": 1 - (prop_swapped + prop_adjacent)
    })
    df_results.name = "Proportion"
    df_results = df_results.reset_index(name="Proportion")
    df_results = df_results.rename(columns={"index": ""})

    # Print to command line
    print_table(df_results, show_index=False)


def plot_prob_over_sequence(df_pred):
    """
    Plots average probability of prediction over each number in the sequence.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    """
    df = df_pred.groupby(by=["seq_number"])["prob"].mean().reset_index()
    sns.barplot(data=df, x="seq_number", y="prob")
    plt.xlabel("Number in the US Sequence")
    plt.ylabel("Prediction Probability")
    plt.tight_layout()
    plt.show()


def plot_image_count_over_sequence(df_pred):
    """
    Plots number of imgaes for each number in the sequence.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Test set predictions. Each row is a test example with a label,
        prediction, and other patient and sequence-related metadata.
    """
    df_count = df_pred.groupby(by=["seq_number"]).apply(lambda df: len(df))
    df_count.name = "count"
    df_count = df_count.rename(columns={"seq_number": "Number in Sequence",
                                        0: "Number of Images"})
    sns.barplot(data=df_count, x="seq_number", y="count")
    plt.xlabel("Number in the US Sequence")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()


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


def filter_most_confident(df_pred, local=False):
    """
    Given predictions for all images across multiple US sequences, filter the
    prediction with the highest confidence (based on probability).

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
    def _get_local_groups(df):
        """
        Identify local groups of consecutive labels. Return a list of increasing
        integers which identify these groups.

        Note
        ----
        Assumes input dataframe is already sorted by US sequence number.

        Parameters
        ----------
        df : pandas.DataFrame
            Contains test set predictions and metadata for the US sequence for
            one patient.

        Returns
        -------
        list
            Increasing integer values, which represent each local group of
            images with the same label.
        """
        curr_val = 0
        prev_label = None
        local_groups = []

        for label in df.label.tolist():
            if label != prev_label:
                curr_val += 1
                prev_label = label
            
            local_groups.append(curr_val)

        return local_groups

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
            _get_local_groups)
        df_pred["local_group"] = np.concatenate(local_grps_per_seq.values)

        df_seqs = df_pred.groupby(by=["id", "visit", "local_group"])
        df_filtered = df_seqs.apply(lambda df: df[df.out == df.out.max()])

    return df_filtered


################################################################################
#                                  Main Flows                                  #
################################################################################
def main_test_set(model_cls, checkpoint_path=CKPT_PATH_MULTI,
                  save_path=TEST_PRED_PATH, sequential=False,
                  include_unlabeled=False):
    """
    Performs inference on test set, and saves results

    Parameters
    ----------
    model_cls : class reference
        Used to load specific model from checkpoint path
    checkpoint_path : str, optional
        Path to file containing EfficientNetPL model checkpoint, by default
        CKPT_PATH_MULTI
    save_path : str, optional
        Path to file to save test results to, by default TEST_PRED_PATH
    sequential : bool, optional
        If True, feed full image sequences into model, by default False.
    include_unlabeled : bool, optional
        If True, include unlabeled images in test set, by default False.
    """
    # 2. Get metadata, specifically for the test set
    # 2.0 Get image filenames and labels
    if not include_unlabeled:
        df_metadata = utils.load_metadata()
    else:
        df_metadata = utils.load_metadata(extract=True, include_unlabeled=True,
                                          dir=constants.DIR_IMAGES)
        df_metadata = utils.remove_only_unlabeled_seqs(df_metadata)

    # 2.1 Get hyperparameters for run
    model_dir = os.path.dirname(checkpoint_path)
    hparams = get_hyperparameters(model_dir)

    # 2.2 Load test metadata
    df_test_metadata = get_test_set_metadata(df_metadata, hparams)

    # 2.3 Sort, so no mismatch occurs due to groupby sorting
    df_test_metadata = df_test_metadata.sort_values(by=["id", "visit"],
                                                    ignore_index=True)

    # 3. Load existing model and send to device
    model = model_cls.load_from_checkpoint(checkpoint_path)
    model = model.to(DEVICE)

    # 4. Get predictions on test set
    if not sequential:
        filenames = df_test_metadata.filename.tolist()
        preds, probs, outs = predict_on_images(model=model, filenames=filenames,
                                               dir=None)
    else:
        # Perform inference one sequence at a time
        ret = df_test_metadata.groupby(by=["id", "visit"]).apply(
            lambda df: predict_on_sequences(
                model=model, filenames=df.filename.tolist(), dir=None)
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
    # Add model type to save path
    test_save_path = TEST_PRED_PATH % (MODEL_TYPE,)

    # Inference on test set
    if not os.path.exists(test_save_path):
        sequential = ("seq" in MODEL_TYPE)

        # Get model class based on model type
        if sequential:
            model_cls = EfficientNetLSTM
        else:
            model_cls = EfficientNetPL

        # Includes other
        include_unlabeled = ("other" in MODEL_TYPE)

        main_test_set(model_cls, CKPT_PATH, test_save_path,
                      sequential=sequential,
                      include_unlabeled=include_unlabeled)

    # Load test metadata
    df_pred = pd.read_csv(test_save_path)

    if "five_view" in MODEL_TYPE and "other" not in MODEL_TYPE:
        # Print reasons for misclassification of most confident predictions
        check_misclassifications(df_pred)

        # Plot confusion matrix
        plot_confusion_matrix(df_pred, filter_confident=True)
    elif "binary" in MODEL_TYPE:
        # Plot binary avg. prediction probabilites by view
        plot_pred_probability_by_views(df_pred)

    pass