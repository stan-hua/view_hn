"""
Script to create metadata file for UIowa and CHOP for view labeling
"""

# Standard libraries
import glob
import os
import re

# Non-standard libraries
import cv2
import pandas as pd

# Custom libraries
from src.data import constants
from src.data_prep import utils


################################################################################
#                                  Constants                                   #
################################################################################
# Originating source data directory
SRC_DIR_UIOWA_DATA = os.path.join(os.path.dirname(constants.DIR_DATA), "UIowa")


################################################################################
#                                  Functions                                   #
################################################################################
def create_chop_metadata(dir_data, save_path=None):
    """
    Based on folder structure and image names, create metadata table with
    relative image paths.

    Parameters
    ----------
    dir_data : str
        Path to CHOP folder
    save_path : str, optional
        If provided, save metadata table to file path

    Returns
    -------
    pandas.DataFrame
        Metadata table containing `filename`, `label`, `id`, `visit` and
        `seq_number`
    """
    # Label mapping
    label_map = {"TRAN_L": "Transverse_Left", "SAG_L": "Sagittal_Left",
                 "TRAN_R": "Transverse_Right", "SAG_R": "Sagittal_Right"}

    # Get filenames of all viable images
    fname_regex = "(.*)_(SAG|TRAN)_(L|R)_cropped-preprocessed\.png"
    img_fnames = [path for path in os.listdir(dir_data)
                  if re.match(fname_regex, path)]

    # Since images are single-visit, all visits and sequence numbers will be 0
    visit = "0"
    sequence_number = 0

    # Accumulate metadata for each image
    rows = []
    for img_fname in img_fnames:
        # Split filename into parts
        fname_parts = img_fname.split("_")

        # Get patient ID
        patient_id = fname_parts[0]

        # Get view label
        label = "_".join(fname_parts[1:3])
        label = label_map[label]

        # Create metadata row
        img_metadata = pd.DataFrame({
            "filename": [img_fname],
            "label": [label],
            "id": [patient_id],
            "visit": [visit],
            "seq_number": [sequence_number],
        })

        # Store metadata
        rows.append(img_metadata)

    # Combine metadata rows
    df_metadata = pd.concat(rows, ignore_index=True)

    # Save metadata to location
    if save_path:
        df_metadata.to_csv(save_path, index=False)

    return df_metadata


def create_uiowa_metadata_from_src(dir_data, save_path=None):
    """
    Based on folder structure and image names, create metadata table with
    relative image paths.

    Parameters
    ----------
    dir_data : str
        Path to UIowa folder
    save_path : str, optional
        If provided, save metadata table to file path

    Returns
    -------
    pandas.DataFrame
        Metadata table containing `filename`, `label`, `id`, `visit` and
        `seq_number`
    """
    # List of possible filenames for UIowa images
    possible_fnames = set(["LT_cropped.png", "LS_cropped.png",
                           "RT_cropped.png", "RS_cropped.png"])

    # Label mapping
    label_map = {"LT": "Transverse_Left", "LS": "Sagittal_Left",
                 "RT": "Transverse_Right", "RS": "Sagittal_Right"}

    # Get folders for each patient
    patient_folders = glob.glob(os.path.join(dir_data, "C*"))
    patient_folders.extend(glob.glob(os.path.join(dir_data, "O*")))

    # Since images are single-visit, all visits and sequence numbers will be 0
    visit = "0"
    sequence_number = 0

    # Accumulate metadata for each image
    rows = []
    for folder in patient_folders:
        # Get patient ID from sub-folder name
        patient_id = os.path.basename(folder)

        # Get filenames of valid images
        existing_img_fnames = [fname for fname in os.listdir(folder)
                              if fname in possible_fnames]

        # Store metadata for each image
        for img_fname in existing_img_fnames:
            # Get view label from filename
            label = img_fname.split("_")[0]
            label = label_map[label]

            # Create relative path to image
            filename = patient_id + "/" + img_fname

            # Create metadata row
            img_metadata = pd.DataFrame({
                "filename": [filename],
                "label": [label],
                "id": [patient_id],
                "visit": [visit],
                "seq_number": [sequence_number],
            })

            # Store metadata
            rows.append(img_metadata)

    # Combine metadata rows
    df_metadata = pd.concat(rows, ignore_index=True)

    # Save metadata to location
    if save_path:
        df_metadata.to_csv(save_path, index=False)

    return df_metadata


def preprocess_uiowa_data(src_dir=SRC_DIR_UIOWA_DATA,
                          dst_dir=constants.DIR_UIOWA_DATA):
    """
    Preprocesses UIowa Data and places it into the ViewLabeling data directory.

    Parameters
    ----------
    src_dir : str, optional
        Source directory containined nested data, by default
        SRC_DIR_UIOWA_DATA
    dst_dir : str, optional
        Destination directory to store preprocessed images, where patient ID
        is prepended to the original filename, by default
        constants.DIR_UIOWA_DATA
    """
    # Create metadata for src directory
    df_metadata = create_uiowa_metadata_from_src(src_dir)

    # Create new filename
    df_metadata["old_path"] = df_metadata["filename"]
    df_metadata["filename"] = df_metadata["filename"].str.replace("/", "_")

    # Make full path to source images
    df_metadata["old_path"] = df_metadata["old_path"].map(
        lambda x: os.path.join(src_dir, x))

    for old_path, new_fname in df_metadata[["old_path", "filename"]].itertuples(
            index=False, name=None):
        # Preprocess image
        img = cv2.imread(old_path)
        processed_img = utils.preprocess_image(
            img,
            resize_dims=(256, 256),
            ignore_crop=True)

        # Save image to specified directory
        new_path = os.path.join(dst_dir, new_fname)
        cv2.imwrite(new_path, processed_img)

    # Drop old filename
    df_metadata = df_metadata.drop(columns=["old_path"])

    # Save metadata
    df_metadata.to_csv(constants.UIOWA_METADATA_FILE, index=False)


def create_stanford_non_seq_metadata(dir_data, save_path=None):
    """
    Based on folder structure and image names, create metadata table with
    relative image paths.

    Parameters
    ----------
    dir_data : str
        Path to Stanford Non-Sequence folder
    save_path : str, optional
        If provided, save metadata table to file path

    Returns
    -------
    pandas.DataFrame
        Metadata table containing `filename`, `label`, `id`, `visit` and
        `seq_number`
    """
    # Label mapping
    label_map = {"LT": "Transverse_Left", "LS": "Sagittal_Left",
                 "RT": "Transverse_Right", "RS": "Sagittal_Right"}

    # Get filenames of all viable images
    fname_regex = "batch_dicom_([^_]+)-([^_]+)(LS|LT|RS|RT)" + \
                  "_cropped-preprocessed\.png"
    img_fnames = [path for path in os.listdir(dir_data)
                  if re.match(fname_regex, path)]

    # Since no videos, all sequence numbers will be 0
    sequence_number = 0

    # Accumulate metadata for each image
    rows = []
    for img_fname in img_fnames:
        # Search regex matches again
        match = re.match(fname_regex, img_fname)
        assert match, "Filtered filenames MUST match regex at this point!"

        # Get patient ID
        patient_id = match.group(1)

        # Get patient visit
        visit = match.group(2)

        # Get view label
        label = label_map[match.group(3)]

        # Create metadata row
        img_metadata = pd.DataFrame({
            "filename": [img_fname],
            "label": [label],
            "id": [patient_id],
            "visit": [visit],
            "seq_number": [sequence_number],
        })

        # Store metadata
        rows.append(img_metadata)

    # Combine metadata rows
    df_metadata = pd.concat(rows, ignore_index=True)

    # Save metadata to location
    if save_path:
        df_metadata.to_csv(save_path, index=False)

    return df_metadata


def create_sickkids_silent_trial_metadata(dir_data, save_path=None):
    """
    Based on folder structure and image names, create metadata table with
    relative image paths.

    Parameters
    ----------
    dir_data : str
        Path to SickKids Silent Trial folder
    save_path : str, optional
        If provided, save metadata table to file path

    Returns
    -------
    pandas.DataFrame
        Metadata table containing `filename`, `label`, `id`, `visit` and
        `seq_number`
    """
    # Regex that matches filenames for SickKids Silent Trial images
    fname_regex = ".*(Sag|SAG|Trv|TRV)(\s?)(\d+)(\w)-preprocessed\.png"

    # Label mappings
    plane_label_map = {
        "Sag": "Sagittal",
        "SAG": "Sagittal",
        "Trv": "Transverse",
        "TRV": "Transverse",
    }
    side_label_map = {
        "L": "Left",
        "R": "Right",
    }

    # Get folders for each patient
    patient_folders = glob.glob(os.path.join(dir_data, "Study ID *"))

    # Since data is not image sequences, all sequence numbers will be 0
    sequence_number = 0

    # Accumulate metadata for each image
    rows = []
    for folder in patient_folders:
        # Get patient ID from sub-folder name
        patient_id = os.path.basename(folder).split(" ")[-1]

        # Get filenames of all viable images
        img_fnames = [path for path in os.listdir(folder)
                    if re.match(fname_regex, path)]

        # Store metadata for each image
        for img_fname in img_fnames:
            # Perform regex search again
            # NOTE: All filenames must be filtered at this point
            match = re.match(fname_regex, img_fname)

            # Get view label from filename
            plane = plane_label_map[match.group(1)]
            side = side_label_map[match.group(4)]
            label = "_".join([plane, side])

            # Get visit number
            visit = match.group(3)

            # Create relative path to image
            filename = os.path.basename(folder) + "/" + img_fname

            # Create metadata row
            img_metadata = pd.DataFrame({
                "filename": [filename],
                "label": [label],
                "id": [patient_id],
                "visit": [visit],
                "seq_number": [sequence_number],
            })

            # Store metadata
            rows.append(img_metadata)

    # Combine metadata rows
    df_metadata = pd.concat(rows, ignore_index=True)

    # Save metadata to location
    if save_path:
        df_metadata.to_csv(save_path, index=False)

    return df_metadata


if __name__ == '__main__':
    # Create CHOP metadata
    df_metadata = create_chop_metadata(
        constants.DIR_CHOP_DATA,
        constants.UIOWA_METADATA_FILE,
    )

    # Preprocess UIowa data & Create metadata
    preprocess_uiowa_data(src_dir=SRC_DIR_UIOWA_DATA)

    # Create Stanford (Non-Seq) Metadata
    create_stanford_non_seq_metadata(
        constants.DIR_STANFORD_NON_SEQ_DATA,
        constants.SU_NON_SEQ_METADATA_FILE,
    )

    # Create Stanford (Non-Seq) Metadata
    create_sickkids_silent_trial_metadata(
        constants.DIR_SICKKIDS_SILENT_TRIAL_DATA,
        constants.SK_ST_METADATA_FILE,
    )
