"""
grad_cam.py

Description: Creates images for model explainability, where GradCAM is layered.
"""

# Standard libraries
import argparse
import logging
import os
from collections import defaultdict

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

# Custom libraries
from src.data import constants
from src.data_viz import utils as viz_utils
from src.scripts import load_data, load_model, model_eval


# Set theme
viz_utils.set_theme("dark")

################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)

# Mapping of original label to simplified name
LABEL_TO_SHORTHAND = {
    "Sagittal_Left": "left_sag",
    "Sagittal_Right": "right_sag",
    "Transverse_Left": "left_trans",
    "Transverse_Right": "right_trans",
    "Bladder": "bladder",
}

# Labels to show GradCAMs for
SHOW_LABELS = [
    "Sagittal_Left",
    "Sagittal_Right",
    "Transverse_Left",
    "Transverse_Right",
    "Bladder",
]


################################################################################
#                                Main Functions                                #
################################################################################
def explain_model_on_dset(exp_name, dset, label_whitelist=SHOW_LABELS,
                          save_dir=None, **kwargs):
    """
    Given an experiment name and dataset, extract grad-cam heatmaps for real
    examples and save to directory specified

    Parameters
    ----------
    exp_name : str
        Experiment name
    dset : str
        Name of data split or evaluation dataset
    label_whitelist : list, optional
        Whitelist of labels to create images, by default SHOW_LABELS
    save_dir : str, optional
        Directory to save images to, by default creates `exp_name/dset`
        subdirectory under `grad_cam` directory
    """
    # INPUT: If save directory not provided, create default subdirectory
    if not save_dir:
        save_dir = os.path.join(constants.DIR_FIGURES_CAM, exp_name, dset)

    # Get model hyperparameters
    hparams = load_model.get_hyperparameters(exp_name=exp_name)

    # Get label part
    label_part = hparams.get("label_part")

    # Load model
    model = load_model.load_pretrained_from_exp_name(exp_name)
    # Get last convolutional layer for model
    target_layers = [load_model.get_last_conv_layer(model)]
    # Set model to train
    model.eval()
    if hasattr(model, "temporal_backbone"):
        model.temporal_backbone.train()

    # Create GradCAM object
    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # Load predictions
    pred_path = model_eval.create_save_path(exp_name=exp_name, dset=dset)
    df_preds = pd.read_csv(pred_path)

    # Make GradCAMs for correctly AND incorrectly predicted images
    for op in ("==", "!="):
        # Get filenames of images whose split labels were classified right/wrong
        df_temp = df_preds[df_preds.eval(f"label {op} pred")]
        filenames = df_temp.filename.tolist()

        # Create folder prefix based on result of classification
        prefix = "classified_" if op == "==" else "misclassified_"

        # Make GradCAMs separately for each label
        for label in label_whitelist:
            # Get label shorthand
            label_shorthand = LABEL_TO_SHORTHAND[label]
            # Create subfolder name
            subfolder_name = f"{prefix}{label_shorthand}"
            # Create full path to subfolder
            temp_save_dir = os.path.join(save_dir, subfolder_name)

            # Create GradCAMs
            explain_model_for_images_with_label(
                cam=cam,
                dset=dset,
                label=label,
                label_part=label_part,
                filenames=filenames,
                save_dir=temp_save_dir,
                **kwargs)


def explain_model_for_images_with_label(cam, dset, label, label_part,
                                        filenames=None,
                                        save_dir=None,
                                        n=4):
    """
    Create GradCAM explanations for label-specific images from a dataset.

    Parameters
    ----------
    cam : pytorch_grad_cam.GradCAM
        GradCAM object loaded with a model
    dset : str
        Name of evaluation split/dataset
    label : str
        ORIGINAL (unsplit) label to filter for.
    label_part : str
        Type of label split from original, that the model was trained on. One of
        ("side", "plane")
    filenames : list, optional
        List of filenames to filter for, by default None
    save_dir : str, optional
        Directory to save GradCAM images to, by default None
    n : int, optional
        Number of images to plot in GradCAM, by default 4
    """
    # Create filters on:
    #   a) ORIGINAL labels
    filters = {
        "orig_label": set([label]),
    }
    # If provided, filter on b) specific files
    if filenames:
        filters["filename"] = set(filenames)

    # Create image dataloader, filtering for the right labels
    img_dataloader = load_data.get_dset_dataloader_filtered(
        dset=dset,
        filters=filters,
        keep_orig_label=True,
        label_part=label_part,
        full_seq=False,
        full_path=True,
        num_workers=0,
    )

    # Get image tensors (with N images) for each label
    idx_to_imgs = get_n_images_per_label(img_dataloader, n=n)

    # Extract GradCAM-overlayed images
    idx_to_orig_imgs, idx_to_overlayed_imgs = extract_gradcams(cam, idx_to_imgs)
    # Skip, if no images found
    if not idx_to_overlayed_imgs:
        LOGGER.warning(f"No correct/incorrectly classified images found for "
                       "label: {label}! Skipping...")
        return

    # Get all original and overlayed images
    orig_imgs_lst = []
    overlayed_imgs_lst = []
    for label_idx in list(idx_to_overlayed_imgs.keys()):
        # Get original images
        curr_orig_imgs = idx_to_orig_imgs[label_idx]
        # Get overlayed imgaes
        curr_overlayed_imgs = idx_to_overlayed_imgs[label_idx]
        if curr_orig_imgs and curr_overlayed_imgs:
            orig_imgs_lst.append(curr_orig_imgs)
            overlayed_imgs_lst.append(curr_overlayed_imgs)

    # Concatenate
    orig_imgs = np.concatenate(orig_imgs_lst)
    overlayed_imgs = np.concatenate(overlayed_imgs_lst)

    # Create grid plot with original images
    viz_utils.gridplot_images(
        orig_imgs,
        filename=f"orig_gridplot.png",
        save_dir=save_dir,
        title=f"(Original) Full Label: {label} | Label Part: {label_part}"
    )

    # Create grid plot with GradCAM
    viz_utils.gridplot_images(
        overlayed_imgs,
        filename=f"cam_gridplot.png",
        save_dir=save_dir,
        title=f"(GradCAM) Full Label: {label} | Label Part: {label_part}"
        )

    # Close open figures
    plt.close("all")


def get_n_images_per_label(img_dataloader, n=4):
    """
    Get N images per label.

    Parameters
    ----------
    img_dataloader : torch.DataLoader
        Used to load images and metadata
    n : int, optional
        Number of images to get GradCAMs for per label, by default 4

    Returns
    -------
    dict of (int : torch.FloatTensor)
        Mapping of prediction index to image tensor of examples
    """
    # Store number of images for each label
    idx_to_num_imgs = defaultdict(int)

    # Accumulate images for each label
    idx_to_example_imgs = defaultdict(list)

    # Extract embeddings in batches
    for data, metadata in img_dataloader:
        # If shape is (1, seq_len, C, H, W), flatten first dimension
        if len(data.size()) == 5:
            data = data.squeeze(dim=0)

        # Filter for data in list
        labels = metadata["label"]

        # Store image tensors for each label
        for label_idx in labels.unique():
            label_idx = label_idx.item()

            # Early continue, if stored enough example images for that label
            if idx_to_num_imgs[label_idx] >= n:
                continue

            # Filter for label-specific images
            label_mask = (labels == label_idx)
            label_data = data[label_mask]

            # Store and update count
            idx_to_example_imgs[label_idx].append(label_data)
            idx_to_num_imgs[label_idx] += len(label_data)

    # Combine each accumulated lists of image tensors
    for label_idx in list(idx_to_example_imgs.keys()):
        # Concatenate image tensors
        img_tensors = idx_to_example_imgs[label_idx]
        img_tensor = torch.cat(img_tensors)

        # Ensure within limit of number of images
        img_tensor = img_tensor[:n]

        idx_to_example_imgs[label_idx] = img_tensor

    return idx_to_example_imgs


def extract_gradcams(cam, idx_to_imgs):
    """
    Extract class activation maps (CAM) and overlay over images provided.

    Parameters
    ----------
    cam : pytorch_grad_cam.GradCAM
        GradCAM object loaded with a model
    idx_to_imgs : dict of (int, torch.FloatTensor)
        Mapping of prediction index to example images

    Returns
    -------
    tuple of (dict of (int, np.ndarray))
        First tuple: Mapping of prediction index to original images
        Second tuple: Mapping of prediction index to GradCAM overlayed image
    """
    # Accumulate original and overlayed images
    idx_to_orig_img = defaultdict(list)
    idx_to_overlayed_img = defaultdict(list)

    # For each label, create overlayed images
    for label_idx, imgs in tqdm(idx_to_imgs.items()):
        # Specify target (label) of interest
        targets = [ClassifierOutputTarget(label_idx)]

        # POST-PROCESSING: For each image, create GradCAM-overlayed image
        for i in range(len(imgs)):
            curr_img = imgs[i]

            # Extract GradCAM
            cam_mask = cam(
                input_tensor=curr_img.unsqueeze(0),
                targets=targets,
            )

            # Convert to RGB numpy array
            img_arr = curr_img.numpy()
            # Move channel dimension
            if img_arr.shape[0] == 3:
                img_arr = np.moveaxis(img_arr, 0, 2)

            # Store original image
            idx_to_orig_img[label_idx].append(np.uint8(img_arr * 255))

            # Create overlayed image
            overlayed_img = show_cam_on_image(img_arr, cam_mask[0],
                                              use_rgb=True)
            idx_to_overlayed_img[label_idx].append(overlayed_img)

    return idx_to_orig_img, idx_to_overlayed_img


################################################################################
#                               Helper Functions                               #
################################################################################
def init(parser):
    """
    Initialize ArgumentParser arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser object
    """
    arg_help = {
        "exp_name": "Name of experiment",
        "dset": "Name of evaluation splits or datasets",
        "label_whitelist": "Whitelist of labels to get GradCAMs for",
    }

    parser.add_argument("--exp_name", required=True, nargs="+",
                        help=arg_help["exp_name"])
    parser.add_argument("--dset", required=True, nargs="+",
                        help=arg_help["dset"])
    parser.add_argument("--label_whitelist", nargs="+",
                        default=SHOW_LABELS,
                        help=arg_help["label_whitelist"])


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Extract GradCAM for all specified experiments and datasets
    for EXP_NAME in ARGS.exp_name:
        for DSET in ARGS.dset:
            explain_model_on_dset(
                exp_name=EXP_NAME,
                dset=DSET,
                label_whitelist=ARGS.label_whitelist,
            )
