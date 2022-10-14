"""
utils.py

Description: Contains helper functions for a variety of data/label preprocessing
             functions.
"""

# Standard libraries
import glob
import os

# Non-standard libraries
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from skimage.exposure import equalize_hist

# Custom libraries
from src.data import constants


################################################################################
#                               Metadata Related                               #
################################################################################
def load_sickkids_metadata(path=constants.SK_METADATA_FILE, extract=False,
                  relative_side=False,
                  include_unlabeled=False,
                  img_dir=None,
                  include_test_set=False,
                  test_path=constants.SK_TEST_METADATA_FILE):
    """
    Load SickKids metadata table with filenames and view labels.

    Note
    ----
    If <include_unlabeled> specified, <img_dir> must be provided.

    If <relative_side> is True, the following examples happens:
        - [Saggital_Left, Transverse_Right, Bladder] ->
                [Saggital_First, Transverse_Second, Bladder]
        - [Saggital_Right, Transverse_Left, Bladder] ->
                [Saggital_First, Transverse_Second, Bladder]

    Parameters
    ----------
    path : str, optional
        Path to CSV metadata file, by default constants.SK_METADATA_FILE
    extract : bool, optional
        If True, extracts patient ID, US visit, and sequence number from the
        filename, by default False.
    relative_side : bool, optional
        If True, converts side (Left/Right) to order in which side appeared
        (First/Second/None). Requires <extract> to be True, by default False.
    include_unlabeled : bool, optional
        If True, include all unlabeled images in <img_dir>, by default False.
    img_dir : str, optional
        Directory containing unlabeled (and labeled) images.
    include_test_set : bool, optional
        If True and path to test metadata file specified, include test set
        labels in loaded metadata, by default False.
    test_path : bool, optional
        If <include_test_set>, this path points to the metadata file for the
        internal test data, by default constants.SK_TEST_METADATA.

    Returns
    -------
    pandas.DataFrame
        May contain metadata (filename, view label, patient id, visit, sequence
        number).
    """
    df_metadata = pd.read_csv(path)

    # If specified, include internal test set labels
    if include_test_set:
        df_test_metadata = pd.read_csv(test_path)
        df_metadata = pd.concat([df_metadata, df_test_metadata],
                                ignore_index=True)

    # Rename columns
    df_metadata = df_metadata.rename(columns={"IMG_FILE": "filename",
                                              "revised_labels": "label"})

    # If specified, include unlabeled images in directory provided
    if include_unlabeled:
        assert img_dir is not None, "Please provide `img_dir` as an argument!"

        # Get all image paths
        all_img_paths = glob.glob(os.path.join(img_dir, "*"))
        df_others = pd.DataFrame({"filename": all_img_paths})
        df_others.filename = df_others.filename.map(os.path.basename)
        
        # Remove found paths to already labeled images
        labeled_img_paths = set(df_metadata.filename.tolist())
        df_others = df_others[~df_others.filename.isin(labeled_img_paths)]

        # Exclude Stanford data
        df_others = df_others[~df_others.filename.str.startswith("SU2")]

        # NOTE: Unlabeled images have label "Other"
        df_others["label"] = "Other"
        
        # Merge labeled and unlabeled data
        df_metadata = pd.concat([df_metadata, df_others], ignore_index=True)

    if extract:
        extract_data_from_filename(df_metadata)

        # Convert side (in label) to order of relative appearance (First/Second)
        if relative_side:
            df_metadata = df_metadata.sort_values(
                by=["id", "visit", "seq_number"])
            relative_labels = df_metadata.groupby(by=["id", "visit"])\
                .apply(lambda df: pd.Series(make_side_label_relative(
                    df.label.tolist()))).to_numpy()
            df_metadata["label"] = relative_labels

    return df_metadata


def load_stanford_metadata(path=constants.STANFORD_METADATA_FILE, extract=False,
                           include_unlabeled=False, relative_side=False,
                           img_dir=None):
    """
    Load Stanford metadata table with filenames and view labels.

    Note
    ----
    If <include_unlabeled> specified, <img_dir> must be provided.

    If <relative_side> is True, the following examples happens:
        - [Saggital_Left, Transverse_Right, Bladder] ->
                [Saggital_First, Transverse_Second, Bladder]
        - [Saggital_Right, Transverse_Left, Bladder] ->
                [Saggital_First, Transverse_Second, Bladder]

    Parameters
    ----------
    path : str, optional
        Path to CSV metadata file, by default constants.STANFORD_METADATA_FILE
    extract : bool, optional
        If True, extracts patient ID, US visit, and sequence number from the
        filename, by default False.
    relative_side : bool, optional
        If True, converts side (Left/Right) to order in which side appeared
        (First/Second/None). Requires <extract> to be True, by default False.
    include_unlabeled : bool, optional
        If True, include all unlabeled images in <img_dir>, by default False.
    img_dir : str, optional
        Directory containing unlabeled (and labeled) images.

    Returns
    -------
    pandas.DataFrame
        May contain metadata (filename, view label, patient id, visit, sequence
        number).
    """
    df_metadata = pd.read_csv(path)

    # If specified, include unlabeled images in directory provided
    if include_unlabeled:
        assert img_dir is not None, "Please provide `img_dir` as an argument!"

        # Get all image paths
        all_img_paths = glob.glob(os.path.join(img_dir, "*"))
        df_others = pd.DataFrame({"filename": all_img_paths})
        df_others.filename = df_others.filename.map(os.path.basename)
        
        # Remove found paths to already labeled images
        labeled_img_paths = set(df_metadata.filename.tolist())
        df_others = df_others[~df_others.filename.isin(labeled_img_paths)]

        # Only include Stanford data
        df_others = df_others[df_others.filename.str.startswith("SU2")]

        # NOTE: Unlabeled images have label "Other"
        df_others["label"] = "Other"
        
        # Merge labeled and unlabeled data
        df_metadata = pd.concat([df_metadata, df_others], ignore_index=True)

    if extract:
        extract_data_from_filename(df_metadata)

        # Convert side (in label) to order of relative appearance (First/Second)
        if relative_side:
            df_metadata = df_metadata.sort_values(
                by=["id", "visit", "seq_number"])
            relative_labels = df_metadata.groupby(by=["id", "visit"])\
                .apply(lambda df: pd.Series(make_side_label_relative(
                    df.label.tolist()))).to_numpy()
            df_metadata["label"] = relative_labels

    return df_metadata


def extract_data_from_filename(df_metadata, col="filename"):
    """
    Extract metadata from each image's filename. Assign columns in-place.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    col : str, optional
        Name of column containing filename, by default "filename"
    """
    df_metadata["basename"] = df_metadata[col].map(os.path.basename)
    df_metadata["id"] = df_metadata["basename"].map(
            lambda x: x.split("_")[0])
    df_metadata["visit"] = df_metadata["basename"].map(
        lambda x: x.split("_")[1])
    df_metadata["seq_number"] = df_metadata["basename"].map(
        lambda x: int(x.split("_")[2].replace(".jpg", "")))
    df_metadata.drop(columns=["basename"], inplace=True)


def get_from_paths(paths, item="id"):
    """
    Extract metadata from a list of paths.

    Parameters
    ----------
    paths : list
        List of image paths
    item : str
        Desired item to extract. One of "id" (patient ID), "visit" or
        "seq_number".

    Returns
    -------
    np.array
        List of extracted metadata strings
    """
    assert item in ("id", "visit", "seq_number")

    patient_ids = []
    for path in paths:
        patient_ids.append(get_from_path(path, item))

    return np.array(patient_ids)


def get_from_path(path, item="id"):
    """
    Extract metadata from an image path.

    Parameters
    ----------
    path : str
        Image path
    item : str
        Desired item to extract. One of "id" (patient ID), "visit" or
        "seq_number".

    Returns
    -------
    str
        Extracted metadata strings
    """
    assert item in ("id", "visit", "seq_number")

    item_to_idx = {
        "id": 0,
        "visit": 1,
        "seq_number": 2
    }
    idx = item_to_idx[item]

    extracted = os.path.basename(path).split("_")[idx]

    # If seq number, convert to integer
    if item == "seq_number":
        extracted = int(extracted.split(".")[0])

    return extracted


def remove_only_unlabeled_seqs(df_metadata):
    """
    Removes rows from fully unlabeled ultrasound sequences.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image. May contain
        unlabeled images (label as "Other")

    Returns
    -------
    pandas.DataFrame
        Metadata dataframe with completely unlabeled sequences removed
    """
    # If including unlabeled, exclude sequences that are all "Others"
    # NOTE: This leads to the same patient-visit splits as in training
    mask = df_metadata.groupby(by=["id", "visit"]).apply(
        lambda df: not all(df.label == "Other"))
    mask = mask[mask]
    mask.name = "bool_mask"

    # Join to filter 
    df_metadata = df_metadata.set_index(["id", "visit"])
    df_metadata = df_metadata.join(mask, how="inner")

    # Remove added column
    df_metadata = df_metadata.reset_index()
    df_metadata = df_metadata.drop(columns=["bool_mask"])

    return df_metadata


def extract_from_label(label, extract="plane"):
    """
    Extract data from label string.
    Parameters
    ----------
    label : str
        Label of the form <plane>_<side> or "Bladder"
    extract : str, optional
        One of "plane" or "side", by default "plane"
    Returns
    -------
    str
        Extracted item from label
    """
    label_parts = label.split("_")
    if extract == "side":
        return label_parts[1] if len(label_parts) > 1 else "None"
    return label_parts[0]


def make_side_label_relative(labels):
    """
    First side (Left/Right) becomes First, while the other becomes Second. Use
    this to convert all labels from side to relative side.

    Parameters
    ----------
    labels : array-like
        List of labels of the form <plane>_<side> or Bladder

    Returns
    -------
    list
        List of transformed label to <plane>_[First/Second] or Bladder
    """
    first_side = None
    # Mapping from label with side to relative (First/Second)
    side_to_relative = {}

    new_labels = []
    for label in labels:
        label_parts = label.split("_")

        # If Bladder
        if len(label_parts) == 1:
            new_labels.append(label)
            continue

        # If first side, perform set up
        if first_side is None:
            first_side = label_parts[1]
            second_side = "Left" if first_side != "Left" else "Right"

            for plane in ("Transverse", "Saggital"):
                side_to_relative[f"{plane}_{first_side}"] = f"{plane}_First"
                side_to_relative[f"{plane}_{second_side}"] = f"{plane}_Second"

        # Convert label with side to relative side
        new_labels.append(side_to_relative[label])

    return new_labels


################################################################################
#                                Data Splitting                                #
################################################################################
def split_by_ids(patient_ids, train_split=0.8, seed=constants.SEED):
    """
    Splits list of patient IDs into training and val/test set.

    Note
    ----
    Split may not be exactly equal to desired train_split due to imbalance of
    patients.

    Parameters
    ----------
    patient_ids : np.array or array-like
        List of patient IDs (IDs can repeat).
    train_split : float, optional
        Proportion of total data to leave for training, by default 0.8
    seed : int, optional
        If provided, sets random seed to value, by default constants.SEED

    Returns
    -------
    tuple of (np.array, np.array)
        Contains (train_indices, val_indices), which are arrays of indices into
        patient_ids to specify which are used for training or validation/test.
    """
    # Get expected # of items in training set
    n = len(patient_ids)
    n_train = int(n * train_split)

    # Add soft lower/upper bounds (5%) to expected number. 
    # NOTE: Assume it won't become negative
    n_train_min = int(n_train - (n * 0.05))
    n_train_max = int(n_train + (n * 0.05))

    # Create mapping of patient ID to number of occurrences
    id_to_len = {}
    for _id in patient_ids:
        if _id not in id_to_len:
            id_to_len[_id] = 0
        id_to_len[_id] += 1

    # Shuffle unique patient IDs
    unique_ids = list(id_to_len.keys())
    shuffled_unique_ids = shuffle(unique_ids, random_state=seed)

    # Randomly choose patients to add to training set until full
    train_ids = set()
    n_train_curr = 0
    for _id in shuffled_unique_ids:
        # Add patient if number of training samples doesn't exceed upper bound
        if n_train_curr + id_to_len[_id] <= n_train_max:
            train_ids.add(_id)
            n_train_curr += id_to_len[_id]

        # Stop when there is roughly enough in the training set 
        if n_train_curr >= n_train_min:
            break

    # Create indices
    train_idx = []
    val_idx = []
    for idx, _id in enumerate(patient_ids):
        if _id in train_ids:
            train_idx.append(idx)
        else:
            val_idx.append(idx)

    # Convert to arrays
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    return train_idx, val_idx


def cross_validation_by_patient(patient_ids, num_folds=5):
    """
    Create train/val indices for Cross-Validation with exclusive patient ids
    betwen training and validation sets.

    Parameters
    ----------
    patient_ids : np.array or array-like
        List of patient IDs (IDs can repeat).
    num_folds : int
        Number of folds for cross-validation

    Returns
    -------
    list of <num_folds> tuples of (np.array, np.array)
        Each tuple in the list corresponds to a fold's (train_ids, val_ids)
    """
    folds = []

    training_folds = []
    remaining_ids = patient_ids

    # Get patient IDs for training set in each folds
    while num_folds > 1:
        proportion = 1 / num_folds
        train_idx, rest_idx = split_by_ids(remaining_ids, proportion)

        training_folds.append(np.unique(remaining_ids[train_idx]))
        remaining_ids = remaining_ids[rest_idx]
        
        num_folds -= 1

    # The last remaining IDs are the patient IDs of the last fold
    training_folds.append(np.unique(remaining_ids))

    # Create folds
    fold_idx = list(range(len(training_folds)))
    for i in fold_idx:
        # Get training set indices
        uniq_train_ids = set(training_folds[i])
        train_idx = np.where([_id in uniq_train_ids for _id in patient_ids])[0]

        # Get validation set indices
        val_indices = fold_idx.copy()
        val_indices.remove(i)
        val_patient_ids = np.concatenate(
            np.array(training_folds, dtype=object)[val_indices])
        uniq_val_ids = set(val_patient_ids)
        val_idx = np.where([_id in uniq_val_ids for _id in patient_ids])[0]

        folds.append((train_idx, val_idx))

    return folds


################################################################################
#                             Image Preprocessing                              #
################################################################################
def preprocess_image_dir(image_dir, save_dir, file_regex="*.*"):
    """
    Perform image preprocessing on images in <image_dir> directory with filename
    possibly matching regex. Save preprocessed images to save_dir.

    Parameters
    ----------
    image_dir : str
        Path to directory of unprocessed images
    save_dir : str
        Path to directory to save processed images
    file_regex : str, optional
        Regex for image filename. Only preprocesses these images, by default
        "*.*".
    """
    for img_path in glob.glob(os.path.join(image_dir, file_regex)):
        # Read and preprocess image
        img = cv2.imread(img_path)
        processed_img = preprocess_image(img, (100, 100), (256, 256))

        # Save image to specified directory
        filename = os.path.basename(img_path)
        new_path = os.path.join(save_dir, filename)
        cv2.imwrite(new_path, processed_img)


def preprocess_image(img, crop_dims=(150, 150), resize_dims=(256, 256),
                     ignore_crop=False):
    """
    Perform preprocessing on image array:
        (1) Center crop 150 x 150 (by default)
        (2) Resize to 256 x 256 (by default)
        (3) Histogram normalize image

    Parameters
    ----------
    img : np.array
        Input image to crop
    crop_dims : tuple, optional
        Dimensions (height, width) of image crop, by default (150, 150).
    resize_dims : tuple, optional
        Dimensions (height, width) of final image, by default (256, 256).
    ignore_crop : bool, optional
        If True, do not crop image during preprocessing, by default False.

    Returns
    -------
    np.array
        Preprocessed image.
    """
    if not ignore_crop:
        height, width = img.shape[0], img.shape[1]

        # Sanitize input to be less than or equal to max width/height
        crop_height = min(crop_dims[0], height)
        crop_width = min(crop_dims[1], img.shape[1])

        # Get midpoints and necessary distance from midpoints
        mid_x, mid_y = int(width/2), int(height/2)
        half_crop_height, half_crop_width = int(crop_width/2), int(crop_height/2)

        # Crop image
        crop_img = img[mid_y-half_crop_width:mid_y+half_crop_width,
                    mid_x-half_crop_height:mid_x+half_crop_height]

        # Resize cropped image
        resized_img = cv2.resize(crop_img, resize_dims)
    else:
        resized_img = img

    # Histogram normalize images
    equalized_img = equalize_hist(resized_img)

    # Scale back to 255
    processed_img = 255 * equalized_img

    return processed_img
