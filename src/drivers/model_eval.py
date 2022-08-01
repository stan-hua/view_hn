"""
model_eval.py

Description: Used to evaluate a trained model's performance on the testing set.
"""

# Non-standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.io import read_image, ImageReadMode

# Custom libraries
from src.data import constants
from src.data_prep.dataloaders import UltrasoundDataModule
from src.data_prep.utils import load_metadata, extract_data_from_filename
from src.models.efficientnet_pl import EfficientNetPL


################################################################################
#                                  Constants                                   #
################################################################################
# Flag to use GPU or not
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Checkpoint for a trained 5-view model
CKPT_PATH = constants.DIR_RESULTS + "/five_view/0/last.ckpt"

# Checkpoint for a trained binary-classifier model
CKPT_PATH_BINARY = constants.DIR_RESULTS + "/binary_classifier/0/last.ckpt"

# Table containing predictions and labels for test set
TEST_PRED_PATH = constants.DIR_RESULTS + "/test_set_results(five_view).csv"


################################################################################
#                                  Functions                                   #
################################################################################
def get_test_set_metadata(df_metadata, dir=constants.DIR_IMAGES):
    """
    Get metadata table containing (filename, label) for each image in the test
    set.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    dir : str, optional
        Path to directory containing metadata, by default constants.DIR_IMAGES

    Returns
    -------
    pandas.DataFrame
        Metadata of each image in the test set
    """
    # Set up data
    hparams = {
        "img_size": constants.IMG_SIZE,
        "train": True,
        "test": True,
        "train_test_split": 0.75,
        "train_val_split": 0.75
    }
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


def predict_on_images(model, filenames, dir=constants.DIR_IMAGES):
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

    Returns
    -------
    np.array
        Prediction for each image 
    """
    # Set to evaluation mode
    model.eval()

    # Predict on each images one-by-one
    with torch.no_grad():
        preds = []
        for filename in filenames:
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
            pred = pred.detach().cpu().numpy()[0]

            # Convert from encoded label to label name
            pred_label = constants.IDX_TO_CLASS[pred]

            preds.append(pred_label)

    return np.array(preds)


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


################################################################################
#                                  Main Flows                                  #
################################################################################
def main_test_set():
    """
    Performs inference on test set, and saves results
    """
    # 1. Get image filenames and labels
    df_metadata = load_metadata()

    # 2. Get metadata, specifically for the test set
    df_test_metadata = get_test_set_metadata(df_metadata)

    # 3. Load existing model and send to device
    model = EfficientNetPL.load_from_checkpoint(CKPT_PATH)
    model = model.to(DEVICE)

    # 4. Get predictions on test set
    filenames = df_test_metadata.filename.tolist()
    preds = predict_on_images(model=model, filenames=filenames, dir=None)

    # 5. Save predictions on test set
    df_test_metadata["pred"] = preds
    df_test_metadata.to_csv(TEST_PRED_PATH, index=False)


if __name__ == '__main__':
    if os.path.exists(TEST_PRED_PATH):
        df_test_metadata = pd.read_csv(TEST_PRED_PATH)

    # Plot confusion matrix
    plot_confusion_matrix(df_test_metadata)
