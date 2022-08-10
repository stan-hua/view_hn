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
from src.data_prep.dataloaders import UltrasoundDataModule
from src.data_prep.utils import load_metadata, extract_data_from_filename
from src.models.efficientnet_pl import EfficientNetPL


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure seaborn color palette
sns.set_palette("Paired")


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Flag to use GPU or not
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Checkpoint for a trained 5-view model
CKPT_PATH_MULTI= constants.DIR_RESULTS + "/five_view/0/epoch=6-step=1392.ckpt"

# Checkpoint for a trained binary-classifier model
CKPT_PATH_BINARY = constants.DIR_RESULTS + \
    "/binary_classifier/0/epoch=12-step=2586.ckpt"
CKPT_PATH_BINARY_WEIGHTED = constants.DIR_RESULTS + \
    "/binary_classifier_weighted/0/epoch=11-step=2387.ckpt"

# Table containing predictions and labels for test set
TEST_PRED_PATH = constants.DIR_RESULTS + "/test_set_results(%s).csv"

# Type of models (5 view, binary)
MODEL_TYPES = ("five_view", "binary")

################################################################################
#                                  Functions                                   #
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
    extract_data_from_filename(df_test, col="orig_filename")
    df_test = df_test.drop(columns="orig_filename")

    return df_test


def predict_on_images(model, filenames, dir=constants.DIR_IMAGES,
                      probability=False):
    """
    Performs inference on images specified, and returns predictions.

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch model.
    filenames : np.array or array-like
        Filenames (or full paths) to images to infer on.
    dir : str, optional
        Path to directory containing images, by default constants.DIR_IMAGES
    probability : bool, optional
        If True, saves probability of prediction. Otherwise, saves raw model
        output (activation), by default True.

    Returns
    -------
    tuple of np.array
        (prediction for each image, probability of each prediction)
    """
    # Set to evaluation mode
    model.eval()

    # Predict on each images one-by-one
    with torch.no_grad():
        preds = []
        probs = []

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

            if probability:
                # Get probability
                prob = torch.nn.functional.softmax(out, dim=1)
                prob = prob.detach().cpu().numpy().max()
                probs.append(prob)
            else:
                # Get maximum activation
                prob = float(out.max().detach().cpu())
                probs.append(prob)

            # Convert from encoded label to label name
            pred_label = constants.IDX_TO_CLASS[pred]
            preds.append(pred_label)

    return np.array(preds), np.array(probs)


def plot_confusion_matrix(df_pred):
    """
    Plot confusion matrix based on model predictions.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Each row is a test example with "label" and "pred" defined.
    """
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
        Each row is a test example with "label", "pred" and "prob" defined.
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


################################################################################
#                                  Main Flows                                  #
################################################################################
def main_test_set(checkpoint_path=CKPT_PATH_MULTI, save_path=TEST_PRED_PATH):
    """
    Performs inference on test set, and saves results

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to file containing EfficientNetPL model checkpoint, by default
        CKPT_PATH_MULTI
    save_path : str, optional
        Path to file to save test results to, by default TEST_PRED_PATH
    """
    # 2. Get metadata, specifically for the test set
    # 2.0 Get image filenames and labels
    df_metadata = load_metadata()
    # 2.1 Get hyperparameters for run
    model_dir = os.path.dirname(checkpoint_path)
    hparams = get_hyperparameters(model_dir)
    # 2.2 Load test metadata
    df_test_metadata = get_test_set_metadata(df_metadata, hparams)

    # 3. Load existing model and send to device
    model = EfficientNetPL.load_from_checkpoint(checkpoint_path)
    model = model.to(DEVICE)

    # 4. Get predictions on test set
    filenames = df_test_metadata.filename.tolist()
    preds, probs = predict_on_images(model=model, filenames=filenames, dir=None)

    # 5. Save predictions on test set
    df_test_metadata["pred"] = preds
    df_test_metadata["prob"] = probs
    df_test_metadata.to_csv(save_path, index=False)


if __name__ == '__main__':
    model_type = "five_view"
    assert model_type in MODEL_TYPES

    # Add model type to save path
    test_save_path = TEST_PRED_PATH % (model_type,)

    # Inference on test set
    main_test_set(CKPT_PATH_MULTI if "five_view" in model_type  \
        else CKPT_PATH_BINARY, test_save_path)

    # Load test metadata
    if os.path.exists(test_save_path):
        df_pred = pd.read_csv(test_save_path)

        if "five_view" in model_type:
            # Plot confusion matrix
            plot_confusion_matrix(df_pred)
        else:
            # Plot binary avg. prediction probabilites by view
            plot_pred_probability_by_views(df_pred)
