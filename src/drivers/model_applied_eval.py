"""
model_applied_eval.py

Description: Used to evaluate a trained model's performance on downstream
             clinical task (prediction of surgery for obstruction).
"""
# pylint: disable=wrong-import-order

# Standard libraries
import argparse
import logging
import os
import sys
from collections import OrderedDict

# Non-standard libraries
import numpy as np
import pandas as pd
import torch
from sklearn import metrics as skmetrics

# Custom libraries
from src.data import constants
from src.drivers import load_model, model_eval

# Path to `projects` directory
# NOTE: This can be ignored
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    __file__))))
# Path to HN project directory
HN_PROJ_DIR = os.path.join(PROJ_DIR, "temporal_hydronephrosis")

# Modify path to include projects and HN project directory
for _path in (PROJ_DIR, HN_PROJ_DIR):
    if _path not in sys.path:
        sys.path.append(_path)

# Custom libraries
from temporal_hydronephrosis.op import model_training as hn_model_training
from temporal_hydronephrosis.utilities import dataset_prep as hn_data_utils


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Flag to overwrite existing results
OVERWRITE = True

# Flag to use GPU or not
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Name for HN model
HN_MODEL = "baseline"

# Default parameters for data loader
DEFAULT_DATALOADER_PARAMS = {
    "batch_size": 16,
    "shuffle": False,
    "num_workers": 0,
    "pin_memory": False,
}

# Formatter to create path to SickKids/Stanford canonical view dataset
SK_PATH_FORMATTER = constants.DIR_SICKKIDS_NON_SEQ_DATA

# Filename for HN inference
HN_INFERENCE_FNAME = "{dset}_set_hn_results.csv"


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
        "pairing_method": "Method for creating image pairs to feed into HN "
                          "model",
    }
    parser.add_argument("--exp_name", required=True,
                        nargs='+',
                        help=arg_help["exp_name"])
    parser.add_argument("--dset", default=[constants.DEFAULT_EVAL_DSET],
                        nargs='+',
                        help=arg_help["dset"])
    parser.add_argument("--pairing_method", default="canonical",
                        help=arg_help["pairing_method"])


################################################################################
#                               Helper Functions                               #
################################################################################
def create_path_to_hn_inference_dir(exp_name, pairing_method="canonical"):
    """
    Create path to directory, containing saved predictions by HN model.

    Parameters
    ----------
    exp_name : str
        Experiment name
    pairing_method : str, optional
        Method for pairing images (via ground-truth view labels or experiment
        view labels), by default "canonical"

    Returns
    -------
    str
        Path to directory containing saved predictions (if it exists)
    """
    # CASE 0: If pairing method results in view label model not mattering
    if pairing_method in ("canonical", "random"):
        save_dir = os.path.join(constants.DIR_HN_INFERENCE, pairing_method)
    # CASE 1: If pairing method makes use of predicted view labels
    else:
        save_dir = os.path.join(constants.DIR_HN_INFERENCE, exp_name,
                                pairing_method)

    return save_dir


def create_path_to_canonical_views(patient_id, visit, side, dset="sickkids"):
    """
    Create path to canonical view images in SickKids/Stanford data.

    Parameters
    ----------
    patient_id : str
        Patient ID
    visit : str
        Visit ID
    side : str
        Which kidney ("left" or "right")
    dset : str, optional
        Evaluation dataset, by default "sickkids"

    Returns
    -------
    list
        List of paths to Sagittal, then Transverse image. If path doesn't exist,
        return None.
    """
    # 0. Ensure side is lower-case
    side = side.lower()

    # 0. Ensure side and dset are valid values
    assert side in ("left", "right")
    assert dset in ("val", "test", "sickkids", "stanford")

    paths = []

    # CASE 1: SickKids path desired
    if dset in ("val", "test", "sickkids"):
        side_mapping = {"left": "Left", "right": "Right"}
        common_path = f"{constants.DIR_SICKKIDS_NON_SEQ_DATA}/" \
            f"ORIG{patient_id}_{side_mapping[side]}_{visit}"
        for plane in ("sag", "trv"):
            plane_path = f"{common_path}_{plane}-preprocessed.png"
            paths.append(plane_path)
    # CASE 2: Stanford path desired
    elif dset in ("stanford"):
        side_mapping = {"left": "L", "right": "R"}
        common_path = f"{constants.DIR_STANFORD_NON_SEQ_DATA}/" \
            f"batch_dicom_{patient_id}-{visit}/"
        for plane in ("S", "T"):
            plane_path = f"{common_path}/{side_mapping[side]}{plane}_cropped-"\
                "preprocessed.png"
            paths.append(plane_path)

    # Verify that these paths exist
    for path in paths:
        if not os.path.exists(path):
            LOGGER.info(f"Expected SAG/TRANS image pairs does not exist!\n"
                        f"\tDataset: {dset}\n"
                        f"\tPatient, Visit, Side: ({patient_id}, "
                        f"{visit}, {side})\n"
                        f"\tPath: {path}")
            return None

    return paths


def filter_out_single_image_videos(df_metadata):
    """
    Filter out videos (patient, visit, kidney side) with only 1 image.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Each row contains image-level metadata for a patient, visit, kidney side
        and kidney plane.

    Returns
    -------
    pd.DataFrame
        Metadata table, excluding single-image videos
    """
    # Identify samples with only 1 image (i.e., can't be paired)
    df_count = df_metadata.groupby(["id", "visit", "side"]).size()
    df_one_image = df_count[df_count < 2]

    # Filter out metadata of samples with only 1 image, if any
    if not df_one_image.empty:
        LOGGER.warning("`%s` samples have only 1 image!", len(df_one_image))

        # Perform LEFT JOIN to remove single metadata samples
        df_two_or_more_images = df_count[df_count >= 2].to_frame()
        df_metadata = df_two_or_more_images.join(
            df_metadata.set_index(["id", "visit", "side"]),
            how="left"
        )
        # Remove temporary count column
        df_metadata = df_metadata.drop(columns=[0])
        # Reset index
        df_metadata = df_metadata.reset_index()

    return df_metadata


def get_most_confident_plane_predictions(df_video_metadata):
    """
    Given metadata for images from 1 video, get most confident plane
    predictions.

    Parameters
    ----------
    df_video_metadata : pd.DataFrame
        Contains metadata and plane prediction for 1+ images from the same
        video.

    Returns
    -------
    pd.DataFrame
        Metadata table with only 1 image from each unique plane
    """
    # Ensure only 1 unique video present
    idx_cols = ["id", "visit", "side"]
    unique_identifiers = df_video_metadata[idx_cols].drop_duplicates()
    assert len(unique_identifiers) == 1, \
        "Please provide image metadata from only 1 video!"

    # Select images from predicted views with the highest probability
    df_most_confident = df_video_metadata.groupby(by=["pred"]).apply(
        lambda df: df[df.prob == df.prob.max()])

    # Early exit, if there are at least 2+ images
    if len(df_most_confident) >= 2:
        # Ensure only 2 images
        return df_most_confident.iloc[:2]

    # Select images from predicted views with the two highest probabilities
    df_most_confident = df_video_metadata.groupby(by=["pred"]).apply(
        lambda df: df[df.prob.isin(df.prob.nlargest(2))].iloc[:2])

    return df_most_confident


def pair_by_canonical_views(df_metadata, dset):
    """
    Given metadata from view labeling datasets (i.e., with non-canonical views),
    replace metadata for same patient and visit with their canonical views.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Contains patient ID, visit ID and side
    dset : str
        Evaluation dataset

    Returns
    -------
    pd.DataFrame
        Contains metadata (and file path) to canonical views
    """
    # Can only be video datasets with a corresponding canonical view dataset
    assert dset in ("test", "stanford")

    # Identify unique samples by patient ID, visit ID and kidney side
    keep_cols = ["id", "visit", "side", "hn", "surgery"]
    df_unique_metadata = df_metadata[keep_cols].drop_duplicates()

    # Create expected paths
    df_unique_metadata["filename"] = df_unique_metadata.apply(
        lambda row: create_path_to_canonical_views(
            row.id, row.visit, row.side, dset), axis=1)

    # If any missing SAG/TRANS image pairs, log missing
    total_na = df_unique_metadata["filename"].isna().sum()
    if total_na:
        num_samples = len(df_unique_metadata)
        LOGGER.warning(f"Missing SAG/TRANS image pairs for dataset {dset}: "
                       f"({total_na}/{num_samples})")

    # Drop NA
    df_unique_metadata = df_unique_metadata.dropna(subset=["filename"])

    # Make image pairs have separate rows
    df_unique_metadata = df_unique_metadata.explode("filename")

    # Add plane labels
    df_unique_metadata["plane"] = ["Sagittal", "Transverse"] * \
        int(len(df_unique_metadata) / 2)

    return df_unique_metadata


def pair_by_random_choice(df_metadata):
    """
    Given metadata from view labeling datasets (i.e., with non-canonical views),
    filter metadata for 2 random images from the same patient, visit and kidney
    side.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Contains patient ID, visit ID and side

    Returns
    -------
    pd.DataFrame
        Contains metadata (and file path) to randomly paired same-kidney images
    """
    # Remove single-image videos
    df_metadata = filter_out_single_image_videos(df_metadata)

    # Randomly choose 2 images from the same patient-visit and kidney
    df_paired_metadata = df_metadata.groupby(["id", "visit", "side"]).apply(
        lambda df: df.sample(n=2))

    # Remove groupby index
    df_paired_metadata = df_paired_metadata.reset_index(drop=True)

    return df_paired_metadata


# TODO: Finish this
def pair_by_most_confident_plane_pred(df_metadata):
    """
    Pair SAG/TRANS by most confident plane predictions.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Contains patient ID, visit ID and side, as well as "pred" for predicted
        plane.

    Returns
    -------
    pd.DataFrame
        Contains metadata (and file path) to randomly paired same-kidney images
    """
    # Remove single-image videos
    df_metadata = filter_out_single_image_videos(df_metadata)

    # Choose 2 images from the same patient-visit and kidney
    # NOTE: If no sagittal AND transverse image predicted, two from the same
    #       side will work.
    df_paired_metadata = df_metadata.groupby(["id", "visit", "side"]).apply(
        get_most_confident_plane_predictions)

    # Remove groupby index
    df_paired_metadata = df_paired_metadata.reset_index(drop=True)

    return df_paired_metadata


def filter_for_hn_pairs(df_metadata, dset, pairing_method="canonical"):
    """
    Filter metadata file for pairs of images from the same sample (id, visit).

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Contains columns for patient ID (`id`), visit ID (`visit`),
        specifying left/right kidney (`side`), anatomical plane (`plane`),
        need for surgery (`hn`), and the full path to the image file
        (`filename`)
    dset : str
        Evaluation dataset
    pairing_method : str, optional
        Method for selecting pairs to pass to the HN model, by default
        "canonical" (i.e., select canonical pairs given test set IDs)

    Returns
    -------
    pd.DataFrame
        Guaranteed to have a pair of images for the same patient ID and visit
        ID. However, it's not guaranteed to be from the same side/plane.
    """
    # Remove bladder images
    mask = df_metadata.side.str.lower().isin(["left", "right"])
    df_metadata = df_metadata[mask]

    # Remove images without HN/surgery label
    df_metadata = df_metadata.dropna(subset=["hn", "surgery"])

    # CASE 1: Pairing images by canonical view
    if pairing_method == "canonical":
        return pair_by_canonical_views(df_metadata, dset)
    # CASE 2: Pair random images from the same side
    elif pairing_method == "random":
        return pair_by_random_choice(df_metadata)
    # CASE 3: Pairing by most confident plane prediction
    elif pairing_method == "most_confident_pred":
        return pair_by_most_confident_plane_pred(df_metadata)

    raise NotImplementedError


def create_hn_dataloader(df_metadata, **overwrite_dataloader_params):
    """
    Create dataloader compatible with the HN model by:
        (1) Pairing same kidney images with the same patient ID, visit ID and
            with Sagittal and Transverse plane,

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Contains columns for patient ID (`id`), visit ID (`visit`),
        specifying left/right kidney (`side`), anatomical plane (`plane`),
        need for surgery (`hn`), and the full path to the image file
        (`filename`)
    **overwrite_dataloader_params : dict
        Keyword arguments to pass into torch.nn.DataLoader

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader compatible with HN model
    """
    # 1. Perform check on data
    # 1.1 Identify cases that don't have a pair (Sagittal/Transverse)
    missing_pair = df_metadata.groupby(by=["id", "visit", "side"]).apply(
        lambda df: len(df) < 2)
    if missing_pair.any():
        missing_pair = missing_pair[missing_pair]
        raise RuntimeError("Following cases are missing a Sagittal/Transverse "
                           f"pair!\n{missing_pair}")

    # 1.2 Identify cases that have too many images
    excess_imgs = df_metadata.groupby(by=["id", "visit", "side"]).apply(
        lambda df: len(df) > 2)
    if missing_pair.any():
        excess_imgs = excess_imgs[excess_imgs]
        raise RuntimeError("Following cases have more than a Sagittal/"
                           f"Transverse pair!\n{excess_imgs}")

    # Update default dataloader parameters
    dataloader_params = DEFAULT_DATALOADER_PARAMS.copy()
    if overwrite_dataloader_params:
        dataloader_params.update(overwrite_dataloader_params)

    # Create dataset
    dataset = HNDataset(df_metadata, device=DEVICE)

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)

    return dataloader


def extract_surgery_predictions(df_paired_metadata):
    """
    Create predictions on paired SAG/TRANS ultrasound images, given by metadata.

    Precondition
    ------------
    `df_metadata` contains pairs identifiable by ("id", "visit", "side")

    Parameters
    ----------
    df_paired_metadata : pd.DataFrame
        Contains columns for patient ID (`id`), visit ID (`visit`),
        specifying left/right kidney (`side`), anatomical plane (`plane`),
        need for surgery (`hn`), and the full path to the image file
        (`filename`)

    Returns
    -------
    pd.DataFrame
        Metadata with "surgery_pred" for predicted need for surgery
    """
    # 0. Create copy of metadata
    df_paired_metadata = df_paired_metadata.copy()

    # 1. Create a DataLoader
    dataloader = create_hn_dataloader(df_paired_metadata)

    # 2. Load HN model
    hn_model = hn_model_training.instantiate_model(HN_MODEL, pretrained=True)
    hn_model = hn_model.to(DEVICE)

    # 3. Perform HN inference
    outs, _, pair_ids = hn_model.extract_preds(dataloader)

    # 4. Convert log probability to predicted label (1/0)
    probs = np.exp(outs)
    preds = (probs > 0.5).astype(int)

    # 5. Re-combine with metadata
    df_surgery_preds = pd.DataFrame({
        "pair_id": pair_ids,
        "surgery_pred": preds,
        "surgery_prob": probs,
    })
    df_surgery_preds = df_surgery_preds.set_index("pair_id")
    df_paired_metadata = df_paired_metadata.set_index("pair_id")
    df_paired_metadata = df_paired_metadata.join(df_surgery_preds, how="left")
    df_paired_metadata = df_paired_metadata.reset_index()

    return df_paired_metadata


def load_surgery_predictions(exp_name, dset=constants.DEFAULT_EVAL_DSET,
                        pairing_method="canonical"):
    """
    Load HN surgery model predictions.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.
    pairing_method : str, optional
        Method for selecting pairs to pass to the HN model, by default
        "canonical" (i.e., select canonical pairs given test set IDs)

    Returns
    -------
    pd.DataFrame
        Contains HN predictions for surgery
    """
    # Create path to saved HN predictions
    save_dir = create_path_to_hn_inference_dir(
        exp_name,
        pairing_method=pairing_method)
    save_path = os.path.join(
        save_dir, HN_INFERENCE_FNAME.format(dset=dset))
    # Early exit, if results don't exist yet
    if not os.path.exists(save_path):
        raise RuntimeError(
            "HN predictions don't exist for...\n"
            f"`exp_name`: {exp_name}\n"
            f"`dset`: {dset}\n"
            f"`pairing_method`: {pairing_method}\n"
        )

    # Load prediction
    df_surgery_preds = pd.read_csv(save_path)

    return df_surgery_preds


################################################################################
#                     Analysis - Related Helper Functions                      #
################################################################################
def calculate_metrics(df_pred, ci=False,
                      label_col="label", pred_col="pred",
                      **ci_kwargs):
    """
    Calculate important metrics given prediction and labels

    Parameters
    ----------
    df_pred : pd.DataFrame
        Model predictions and labels
    ci : bool, optional
        If True, add bootstrapped confidence interval, by default False.
    label_col : str, optional
        Name of label column, by default "label"
    pred_col : str, optional
        Name of label column, by default "pred"
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
    unique_labels = sorted(df_pred[label_col].unique())
    for label in unique_labels:
        df_pred_filtered = df_pred[df_pred[label_col] == label]
        metrics[f"Label Accuracy ({label})"] = 0
        if not df_pred_filtered.empty:
            metrics[f"Label Accuracy ({label})"] = \
                model_eval.calculate_accuracy(
                    df_pred_filtered,
                    label_col=label_col,
                    pred_col=pred_col,
                )

    # 2. Overall accuracy
    metrics["Overall Accuracy"] = model_eval.calculate_accuracy(
        df_pred,
        label_col=label_col,
        pred_col=pred_col,
    )
    # Bootstrap confidence interval
    if ci:
        point, (lower, upper) = model_eval.bootstrap_metric(
            df_pred=df_pred,
            metric_func=skmetrics.accuracy_score,
            label_col=label_col,
            pred_col=pred_col,
            **ci_kwargs)
        metrics["Overall Accuracy"] = f"{point} ({lower}, {upper})"

    # 3. F1 Score by class
    # NOTE: Overall F1 Score isn't calculated because it's equal to
    #       Overall Accuracy in multi-label problems.
    f1_scores = skmetrics.f1_score(df_pred[label_col], df_pred[pred_col],
                                   labels=unique_labels,
                                   average=None)
    for i, f1_score in enumerate(f1_scores):
        metrics[f"Label F1-Score ({unique_labels[i]})"] = round(f1_score, 4)

    return pd.Series(metrics)


def eval_calculate_all_metrics(df_pred):
    """
    Calculates all eval. metrics in a proper table format

    Parameters
    ----------
    df_pred : pandas.DataFrame
        HN model predictions. Each row contains a label,
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
    df_metrics_all = calculate_metrics(
        df_pred, ci=True,
        label_col="surgery", pred_col="surgery_pred")
    df_metrics_all.name = "All"
    accum_metric_tables.append(df_metrics_all)

    # 2. Combine
    df_metrics = pd.concat(accum_metric_tables, axis=1)

    return df_metrics


################################################################################
#                                Helper Classes                                #
################################################################################
class HNDataset(torch.utils.data.Dataset):
    """
    HNDataset class.

    Note
    ----
    Used to load image data for HN prediction, given a metadata table.
    """

    def __init__(self, df_metadata, device=DEVICE):
        """
        Instantiate HNDataset object.

        Parameters
        ----------
        df_metadata : pd.DataFrame
            Contains columns for pair ID (`paid_id`) that identifies paired
            SAG/TRANS images, patient ID (`id`), visit ID (`visit`),
            specifying left/right kidney (`side`), anatomical plane (`plane`),
            need for surgery (`hn`), and the full path to the image file
            (`filename`)
        device : torch.device, optional
            Name of device, by default DEVICE.
        """
        self.df_metadata = df_metadata
        self.device = device

        # Reindex by pair ID
        self.df_metadata = self.df_metadata.set_index("pair_id")

        # Store unique pair IDs
        self.pair_ids = self.df_metadata.index.unique().sort_values().to_numpy()


    def __len__(self):
        """
        Get number of items in dataset (i.e., number of SAG/TRANS image pairs).

        Returns
        -------
        int
            Number of SAG/TRANS image pairs
        """
        return len(self.pair_ids)


    def __getitem__(self, idx):
        """
        Return dataset element at index.

        Parameters
        ----------
        idx : int
            Index into dataset

        Returns
        -------
        tuple of (dict, int, int)
            (1) Paired image data in form: {"img": Tensor of shape (2, H, W)},
                where 2 represents SAG/TRANS images stacked,
            (2) Label for surgery (yes/no), and
            (3) Unique pair ID
        """
        pair_id = self.pair_ids[idx]

        # 0. Get metadata rows, corresponding to Sagittal and Transverse images
        metadata = self.df_metadata.loc[pair_id]

        # 1. Perform checks on paired images
        # 1.1. Ensure only 2 images
        assert len(metadata) == 2, \
            "Found non-paired image metadata!"
        # 1.2. Ensure images are from 1 side
        assert len(set(metadata["side"])) == 1, \
            "Paired images are not from the same side!"
        # 1.3. Ensure only 1 label
        # NOTE: 1.2 and 1.3 is necessary to evaluate surgery prediction
        assert len(set(metadata["surgery"])) == 1

        # 2. Load images
        data_dict = {}
        # If there are SAG/TRANS, order by Sagittal then Transverse
        if metadata["plane"].nunique() == 2:
            metadata = metadata.sort_values(by=["plane"])
        paths = metadata["filename"].tolist()
        imgs = []
        for path in paths:
            # Perform custom data preprocessing
            img = hn_data_utils.special_ST_preprocessing(path)
            imgs.append(img)
        data_dict["img"] = torch.FloatTensor(np.stack(imgs)).to(self.device)

        # 3. Get HN label
        label = metadata["surgery"].unique()[0]

        return data_dict, label, pair_id


################################################################################
#                                  Main Flows                                  #
################################################################################
def infer_hn_dset(exp_name, dset=constants.DEFAULT_EVAL_DSET,
                  pairing_method="canonical"):
    """
    Perform inference with HN model, given view label inferred on dset split.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.
    pairing_method : str, optional
        Method for selecting pairs to pass to the HN model, by default
        "canonical" (i.e., select canonical pairs given test set IDs)
    """
    # Create save path for HN prediction
    save_dir = create_path_to_hn_inference_dir(
        exp_name,
        pairing_method=pairing_method)
    save_path = os.path.join(save_dir, HN_INFERENCE_FNAME.format(dset=dset))
    # Early exit, if results already exist and specified not to overwrite
    if os.path.exists(save_path) and not OVERWRITE:
        return

    # 0. Ensure `exp_name` points to existing model
    load_model.get_exp_dir(exp_name)

    # 1. Load view label predictions for dataset
    df_views_pred = model_eval.load_view_predictions(
        exp_name, dset,
        mask_bladder=False)

    # 2. Choose pairs of images to feed into HN model
    df_views_pair_pred = filter_for_hn_pairs(
        df_views_pred, dset,
        pairing_method=pairing_method)

    # 3. Create new column to identify paired same-side images
    pair_ids = df_views_pair_pred.groupby(by=["id", "visit", "side"]).ngroup()
    df_views_pair_pred["pair_id"] = pair_ids

    # 4. Perform HN inference
    df_surgery_preds = extract_surgery_predictions(df_views_pair_pred)

    # 5. Ensure saving directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 6. Save predictions
    df_surgery_preds.to_csv(save_path, index=False)


def analyze_dset_surgery_preds(exp_name, dset=constants.DEFAULT_EVAL_DSET,
                          pairing_method="canonical"):
    """
    Analyze predictions from HN model, given pairing method.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Specific split of dataset. One of (train, val, test), by default
        constants.DEFAULT_EVAL_DSET.
    pairing_method : str, optional
        Method for selecting pairs to pass to the HN model, by default
        "canonical" (i.e., select canonical pairs given test set IDs)
    """
    # 0. Create path to HN inference directory
    save_dir = create_path_to_hn_inference_dir(
        exp_name,
        pairing_method=pairing_method)

    # 1. Load HN predictions
    df_surgery_preds = load_surgery_predictions(exp_name, dset, pairing_method)

    # 2. Compute metrics on HN predictions
    df_metrics = eval_calculate_all_metrics(df_surgery_preds)

    # 3. Save to HN inference directory
    fname = f"{dset}_hn_metrics.csv"
    df_metrics.to_csv(os.path.join(save_dir, fname))


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
            # 1. Perform inference
            infer_hn_dset(exp_name=EXP_NAME, dset=DSET,
                          pairing_method=ARGS.pairing_method)

            # 2. Evaluate predictions and embeddings
            analyze_dset_surgery_preds(exp_name=EXP_NAME, dset=DSET,
                                  pairing_method=ARGS.pairing_method)
