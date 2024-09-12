"""
Script to create metadata file for UIowa and CHOP for view labeling
"""

# Standard libraries
import glob
import json
import os
import re

# Non-standard libraries
import cv2
import fire
import pandas as pd
from tqdm import tqdm

# Custom libraries
from src.data import constants
from src.data_prep import utils


################################################################################
#                                  Constants                                   #
################################################################################
# Originating source data directory
SRC_DIR_UIOWA_DATA = os.path.join(os.path.dirname(constants.DIR_DATA), "UIowa")

# Video/image dataset names
VIDEO_DSETS = (
    "sickkids", "stanford",
    "sickkids_beamform", "stanford_beamform",
)
IMAGE_DSETS = (
    "sickkids_image", "sickkids_silent_trial", "stanford_image",
    "uiowa", "chop", "ubc_adult"
)

# Label columns
LABEL_COLS = ["plane", "side", "label"]


################################################################################
#                             Image Preprocessing                              #
################################################################################
def preprocess_uiowa_data(src_dir=SRC_DIR_UIOWA_DATA,
                          dst_dir=constants.DSET_TO_IMG_SUBDIR_FULL["uiowa"]):
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
        constants.DSET_TO_IMG_SUBDIR_FULL["uiowa"]
    """
    # If destination directory already exists, don't overwrite
    if os.path.exists(dst_dir) and os.listdir(dst_dir) \
            and os.path.exists(constants.DSET_TO_METADATA["raw"]["uiowa"]):
        print("UIowa image data is already pre-processed! Skipping...")
        return

    # Create metadata for src directory
    df_metadata = create_uiowa_metadata_from_src(src_dir)

    # Create new filename
    df_metadata["old_path"] = df_metadata["filename"]
    df_metadata["filename"] = df_metadata["filename"].str.replace("/", "_")

    # Make full path to source images
    df_metadata["old_path"] = df_metadata["old_path"].map(
        lambda x: os.path.join(src_dir, x))

    for old_path, new_fname in tqdm(df_metadata[["old_path", "filename"]].itertuples(
            index=False, name=None)):
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
    df_metadata.to_csv(constants.DSET_TO_METADATA["raw"]["uiowa"], index=False)


################################################################################
#                      Creating "Raw" Metadata Functions                       #
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


def create_stanford_image_metadata(dir_data, save_path=None):
    """
    Based on folder structure and image names, create metadata table with
    relative image paths.

    Note
    ----
    These contain images acquired specifically for hydronephrosis prediction.

    Parameters
    ----------
    dir_data : str
        Path to Stanford image folder
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


def create_sickkids_image_metadata(dir_data, save_path=None):
    """
    Based on folder structure and image names, create metadata table with
    relative image paths.

    Note
    ----
    These contain images acquired specifically for hydronephrosis prediction.

    Parameters
    ----------
    dir_data : str
        Path to SickKids image folder
    save_path : str, optional
        If provided, save metadata table to file path

    Returns
    -------
    pandas.DataFrame
        Metadata table containing `filename`, `label`, `id`, `visit` and
        `seq_number`
    """
    print("Creating SickKids (Image) Metadata...")

    # Label mapping
    label_map = {"trv": "Transverse", "sag": "Sagittal"}

    # Get filenames of all viable images
    fname_regex = "ORIG(\d*)_(Left|Right)_(\d*)_(sag|trv)-preprocessed.png"
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
        patient_id = int(match.group(1))

        # Get patient visit
        visit = match.group(3)

        # Get side
        side = match.group(2)

        # Get plane
        plane = label_map[match.group(4)]

        # Create view label
        label = "_".join([plane, side])

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


def create_ubc_adult_image_metadata():
    """
    Create (raw) UBC adult kidney dataset metadata

    Note
    ----
    Assumes label files are in `.../ViewLabeling/Datasheets/raw/ubc_adult`

    """
    # Label mapping
    # NOTE: Side isn't provided, so it needs to be filled in with unknown for now
    label_map = {"Transverse": "Transverse", "Longitudinal": "Sagittal"}

    # Load first and second labels
    fnames = ["reviewed_labels_1.csv", "reviewed_labels_2.csv"]
    dir_metadata_raw = os.path.join(constants.DIR_METADATA_RAW, "ubc_adult")
    accum_metadata = []
    for fname in fnames:
        path = os.path.join(dir_metadata_raw, fname)
        assert os.path.exists(path), \
            f"[UBC Adult Kidney Dataset] Original label file doesn't exist at: `{path}`"
        # Load view labels
        df_curr = pd.read_csv(path)
        view_labels = pd.json_normalize(df_curr["file_attributes"].apply(json.loads))
        df_curr = df_curr[["filename"]].merge(view_labels, left_index=True, right_index=True)
        accum_metadata.append(df_curr)

    # Concatenate first and second label files
    df_metadata = pd.concat(accum_metadata, ignore_index=True)

    # Aggregate annotation across raters
    df_metadata = df_metadata.groupby("filename", as_index=False).apply(
        lambda df_raters: pd.Series({
            "filename": df_raters["filename"].iloc[0],
            "quality": df_raters["Quality"].mode().iloc[0],
            "view": df_raters["View"].mode().iloc[0],
            "num_views": df_raters["View"].nunique(),
        }),
        include_groups=True,
    )

    # Drop all images with Poor quality
    mask = (df_metadata["quality"].isin(["Poor", "Unsatisfactory"]))
    df_metadata = df_metadata[~mask]
    print(f"[UBC Adult] Dropping `{mask.sum()}` poor-quality images...")

    # Drop all images with 2+ view labels
    mask = df_metadata["num_views"] > 1
    df_metadata = df_metadata[~mask]
    print(f"[UBC Adult] Dropping `{mask.sum()}` images with 2+ view labels...")

    # Get plane from view label
    df_metadata["plane"] = df_metadata["view"].map(label_map.get)
    # Drop all images that aren't Sagittal/Transverse views
    mask = df_metadata["plane"].isna()
    df_metadata = df_metadata[~mask]
    print(f"[UBC Adult] Dropping `{mask.sum()}` images with non - SAG/TRANS view images...")

    # Create inferred ID, visit and sequence number from filename
    df_metadata["id"] = df_metadata["filename"].map(lambda x: int(x.split("_")[0]))
    df_metadata["visit"] = df_metadata["filename"].map(lambda x: int(x.split("_")[1].split("-")[1]))
    df_metadata["seq_number"] = df_metadata["filename"].map(lambda x: int(x.split("_")[1].split("-")[2]))

    # Save metadata
    cols = ["filename", "id", "visit", "seq_number", "quality", "plane"]
    df_metadata = df_metadata[cols]
    df_metadata = df_metadata.sort_values(by=["id", "visit", "seq_number"])
    df_metadata.to_csv(constants.DSET_TO_METADATA["raw"]["ubc_adult"],
                       index=False)


################################################################################
#                            Cleaning Raw Metadata                             #
################################################################################
def clean_sickkids_video_metadata():
    """
    Preprocesses processed SickKids (video) to:
        a) extract view label and its sub-parts plane and side,
        b) add unlabeled data
    """
    print("Cleaning SickKids Video metadata...")
    # Load the original metadata file and the originally test metadata file
    df_metadata = pd.read_csv(constants.DSET_TO_METADATA["raw"]["sickkids"])

    # Rename columns
    df_metadata = df_metadata.rename(columns={
        "image_file": "filename",
        "IMG_FILE": "filename",
        "revised_labels": "label"
    })

    # Fix mislabel saggital --> sagittal
    fix_label_map = {"Saggital_Left": "Sagittal_Left",
                     "Saggital_Right": "Sagittal_Right"}
    df_metadata["label"] = df_metadata["label"].map(lambda x: fix_label_map.get(x, x))

    # Extract side/plane from label
    for curr_label_part in ("plane", "side"):
        df_metadata[curr_label_part] = df_metadata["label"].map(
            lambda x: utils.extract_from_label(x, extract=curr_label_part))

    ############################################################################
    #                          Get unlabeled data                              #
    ############################################################################
    # Get all image paths
    img_dir = constants.DSET_TO_IMG_SUBDIR_FULL["sickkids"]
    all_img_paths = glob.glob(os.path.join(img_dir, "*.*"))
    df_others = pd.DataFrame({"filename": all_img_paths})
    # Only keep filename
    df_others["filename"] = df_others["filename"].map(os.path.basename)

    # Remove found paths to already labeled images
    labeled_img_paths = set(df_metadata["filename"].tolist())
    df_others = df_others[~df_others["filename"].isin(labeled_img_paths)]

    # Exclude Stanford data
    df_others = df_others[~df_others["filename"].str.startswith("SU2")]

    # Exclude all segmentations masks
    df_others = df_others[~df_others["filename"].str.contains("seg.")]

    # NOTE: Unlabeled images have label
    df_others["label"] = None

    # Merge labeled and unlabeled data
    df_metadata = pd.concat([df_metadata, df_others], ignore_index=True)

    ############################################################################
    #     Extract other important metadata (patient/visit ID, HN, surgery)     #
    ############################################################################
    # Extract patient/visit ID and sequence number
    df_metadata = utils.extract_data_from_filename(df_metadata)

    # Get all available surgery labels
    df_metadata = utils.extract_hn_labels(
        df_metadata, sickkids=True, stanford=False)

    # Get machines for each image
    df_metadata["machine"] = utils.get_machine_for_filenames(
        df_metadata.filename.tolist(), sickkids=True)

    # Assign data split
    df_metadata["split"] = df_metadata.apply(
        lambda row: utils.assign_split_row(row, "sickkids"), axis=1)

    # Assign hospital/dataset
    df_metadata["dset"] = "sickkids"

    # Reorder columns
    cols = ["dset", "split"] + df_metadata.columns.tolist()[:-2]
    df_metadata = df_metadata[cols]

    # Add directory name
    df_metadata["dir_name"] = constants.DSET_TO_IMG_SUBDIR_FULL["sickkids"]

    # Print and store basic data statistics
    print_basic_data_stats(df_metadata)

    # Save metadata file
    df_metadata.to_csv(constants.DSET_TO_METADATA["clean"]["sickkids"], index=False)
    print("Cleaning SickKids Video metadata...DONE")


def clean_stanford_video_metadata():
    """
    Preprocess processed Stanford (video) metadata
    """
    print("Cleaning Stanford Video metadata...")
    df_metadata = pd.read_csv(constants.DSET_TO_METADATA["raw"]["stanford"])

    # Fix mislabel saggital --> sagittal
    fix_label_map = {"Saggital_Left": "Sagittal_Left",
                     "Saggital_Right": "Sagittal_Right"}
    df_metadata["label"] = df_metadata["label"].map(lambda x: fix_label_map.get(x, x))

    # Extract side/plane from label
    for curr_label_part in ("plane", "side"):
        df_metadata[curr_label_part] = df_metadata["label"].map(
            lambda x: utils.extract_from_label(x, extract=curr_label_part))

    ############################################################################
    #                          Get unlabeled data                              #
    ############################################################################
    # Get all image paths
    img_dir = constants.DSET_TO_IMG_SUBDIR_FULL["stanford"]
    all_img_paths = glob.glob(os.path.join(img_dir, "*.*"))
    df_others = pd.DataFrame({"filename": all_img_paths})
    # Only keep filename
    df_others["filename"] = df_others["filename"].map(os.path.basename)

    # Remove found paths to already labeled images
    labeled_img_paths = set(df_metadata["filename"].tolist())
    df_others = df_others[~df_others["filename"].isin(labeled_img_paths)]

    # Include only Stanford data
    df_others = df_others[df_others["filename"].str.startswith("SU2")]

    # NOTE: Unlabeled images have label "Other"
    df_others["label"] = None

    # Merge labeled and unlabeled data
    df_metadata = pd.concat([df_metadata, df_others], ignore_index=True)

    # Drop duplicates
    # NOTE: Metadata table contains duplicate rows
    df_metadata = df_metadata.drop_duplicates()

    ############################################################################
    #     Extract other important metadata (patient/visit ID, HN, surgery)     #
    ############################################################################
    # Extract patient/visit ID and sequence number
    df_metadata = utils.extract_data_from_filename(df_metadata)

    # Get all available surgery labels
    df_metadata = utils.extract_hn_labels(
        df_metadata, sickkids=False, stanford=True)

    # Assign data split
    df_metadata = utils.assign_split_table(df_metadata, train_split=0.2)

    # Assign hospital/dataset
    df_metadata["dset"] = "stanford"

    # Reorder columns
    cols = ["dset", "split"] + df_metadata.columns.tolist()[:-2]
    df_metadata = df_metadata[cols]

    # Add directory name
    df_metadata["dir_name"] = constants.DSET_TO_IMG_SUBDIR_FULL["stanford"]

    # Print and store basic data statistics
    print_basic_data_stats(df_metadata)

    # Save metadata file
    df_metadata.to_csv(constants.DSET_TO_METADATA["clean"]["stanford"], index=False)
    print("Cleaning Stanford Video metadata...DONE")


def clean_image_dsets_metadata():
    """
    Preprocess metadata for image datasets:
        1. SickKids Silent Trial
        2. Stanford (image)
        3. UIowa
        4. CHOP

    Note
    ----
    Not a lot to be done because they were created through this script as well.
    """
    # Load metadata
    for dset in IMAGE_DSETS:
        print(f"Cleaning `{dset}` metadata...")
        df_metadata = pd.read_csv(constants.DSET_TO_METADATA["raw"][dset])

        # Extract side/plane from label
        for curr_label_part in ("plane", "side"):
            df_metadata[curr_label_part] = df_metadata["label"].map(
                lambda x: utils.extract_from_label(x, extract=curr_label_part))

        # Drop duplicates
        # NOTE: Metadata table contains duplicate rows
        df_metadata = df_metadata.drop_duplicates()

        # Assign data split, if not already
        df_metadata = utils.assign_split_table(df_metadata, train_split=0.2)

        # Assign hospital/dataset
        df_metadata["dset"] = dset

        # Reorder columns
        cols = ["dset", "split"] + [col for col in df_metadata.columns.tolist() if col not in ["dset", "split"]]
        df_metadata = df_metadata[cols]

        # Add directory name
        df_metadata["dir_name"] = constants.DSET_TO_IMG_SUBDIR_FULL[dset]

        # Print and store basic data statistics
        print_basic_data_stats(df_metadata)

        # Save metadata file
        df_metadata.to_csv(constants.DSET_TO_METADATA["clean"][dset], index=False)
        print(f"Cleaning `{dset}` metadata...DONE")


def clean_sickkids_video_beamform_metadata():
    """
    Preprocesses cropped beamform SickKids (video)
    """
    print("Cleaning SickKids Video (Beamform) metadata...")

    # Load clean metadata file for SickKids Video
    df_metadata = pd.read_csv(constants.DSET_TO_METADATA["clean"]["sickkids"])

    ############################################################################
    #                        Map to Beamform Images                            #
    ############################################################################
    # Change dset
    df_metadata["dset"] = "sickkids_beamform"

    # Overwrite filename and directory name
    df_metadata["filename"] = df_metadata.apply(
        lambda row: "_".join(map(str, [row["id"], row["visit"], row["seq_number"]])) + ".jpg",
        axis=1
    )
    df_metadata["dir_name"] = constants.DSET_TO_IMG_SUBDIR_FULL["sickkids_beamform"]

    # Check which images don't exist
    exists_mask = df_metadata.apply(
        lambda row: os.path.exists(os.path.join(row["dir_name"], row["filename"])),
        axis=1
    )
    print(f"{(~exists_mask).sum()} images don't have original beamform format! Removing...")

    # Filter for images that do exist
    df_metadata = df_metadata[exists_mask]

    ############################################################################
    #                           Post-Processing                                #
    ############################################################################
    # Print and store basic data statistics
    print_basic_data_stats(df_metadata)

    # Save metadata file
    df_metadata.to_csv(constants.DSET_TO_METADATA["clean"]["sickkids_beamform"], index=False)
    print("Cleaning SickKids Video (Beamform) metadata...DONE")


def clean_stanford_video_beamform_metadata():
    """
    Preprocesses cropped beamform Stanford (video)
    """
    print("Cleaning Stanford Video (Beamform) metadata...")

    # Load clean metadata file for Stanford Video
    df_metadata = pd.read_csv(constants.DSET_TO_METADATA["clean"]["stanford"])

    ############################################################################
    #                        Map to Beamform Images                            #
    ############################################################################
    # Change dset
    df_metadata["dset"] = "stanford_beamform"

    # Overwrite filename and directory name
    df_metadata["filename"] = df_metadata.apply(
        lambda row: "_".join(map(str, [row["id"], row["visit"], row["seq_number"]])) + ".jpg",
        axis=1
    )
    df_metadata["dir_name"] = constants.DSET_TO_IMG_SUBDIR_FULL["stanford_beamform"]

    # Check which images don't exist
    exists_mask = df_metadata.apply(
        lambda row: os.path.exists(os.path.join(row["dir_name"], row["filename"])),
        axis=1
    )
    print(f"{(~exists_mask).sum()} images don't have original beamform format! Removing...")

    # Filter for images that do exist
    df_metadata = df_metadata[exists_mask]

    ############################################################################
    #                           Post-Processing                                #
    ############################################################################
    # Print and store basic data statistics
    print_basic_data_stats(df_metadata)

    # Save metadata file
    df_metadata.to_csv(constants.DSET_TO_METADATA["clean"]["stanford_beamform"], index=False)
    print("Cleaning Stanford Video (Beamform) metadata...DONE")


################################################################################
#                               Helper Functions                               #
################################################################################
def print_basic_data_stats(df_metadata):
    """
    Print basic stats about the data

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Each row is metadata for an US image
    """
    df_labeled = df_metadata[~df_metadata["label"].isna()]
    split_to_df = {
        split: df_labeled[df_labeled["split"] == split]
        for split in ("train", "val", "test")
    }
    df_unlabeled = df_metadata[df_metadata["label"].isna()]

    # Compute stats
    stats = {
        "all": {
            "num_patients": df_metadata["id"].nunique(),
            "num_videos": df_metadata.groupby(by=["id", "visit"]).ngroups,
            "num_images": len(df_metadata)
        },
        "labeled": {
            "num_patients": df_labeled["id"].nunique(),
            "num_videos": df_labeled.groupby(by=["id", "visit"]).ngroups,
            "num_images": len(df_labeled),
            "label_distribution": (df_labeled["label"].value_counts() / len(df_labeled)).round(3).to_dict(),
        },
        **{
            f"labeled_{split}": {
                "num_patients": split_to_df[split]["id"].nunique(),
                "num_videos": split_to_df[split].groupby(by=["id", "visit"]).ngroups,
                "num_images": len(split_to_df[split]),
                "label_distribution": (split_to_df[split]["label"].value_counts() / len(split_to_df[split])).round(3).to_dict(),
            }
            for split in ("train", "val", "test")
            if not split_to_df[split].empty
        },
        "unlabeled": {
            "num_patients": df_unlabeled["id"].nunique(),
            "num_videos": df_unlabeled.groupby(by=["id", "visit"]).ngroups,
            "num_images": len(df_unlabeled)
        }
    }

    # Print stats
    print(json.dumps(stats, indent=4))

    # Get dataset
    assert df_metadata["dset"].nunique() == 1, "Should only be one dataset!"
    dset = df_metadata["dset"].unique()[0]

    # Create folder to store it in (under EDA)
    dir_path = os.path.join(constants.DIR_FIGURES_EDA, "metadata_stats")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, f"{dset}_stats.json")
    with open(file_path, "w") as handler:
        json.dump(stats, handler, indent=4)


################################################################################
#                                  Main Flows                                  #
################################################################################
def main_prepare_image_datasets():
    """
    Prepare image datasets, specifically:
        1. Create CHOP metadata
        2. Prepare UIowa data and create metadata
        3. Create Stanford (image) metadata
        4. Create SickKids Silent Trial metadata
    """
    # Create CHOP metadata
    create_chop_metadata(
        constants.DSET_TO_IMG_SUBDIR_FULL["chop"],
        constants.DSET_TO_METADATA["raw"]["chop"],
    )

    # Preprocess UIowa data & Create metadata
    preprocess_uiowa_data(src_dir=SRC_DIR_UIOWA_DATA)

    # Create Stanford (Image) Metadata
    create_stanford_image_metadata(
        constants.DSET_TO_IMG_SUBDIR_FULL["stanford_image"],
        constants.DSET_TO_METADATA["raw"]["stanford_image"],
    )

    # Create SickKids Silent Trial Metadata
    create_sickkids_silent_trial_metadata(
        constants.DSET_TO_IMG_SUBDIR_FULL["sickkids_silent_trial"],
        constants.DSET_TO_METADATA["raw"]["sickkids_silent_trial"],
    )

    # Create SickKids (Image) Metadata
    create_sickkids_image_metadata(
        constants.DSET_TO_IMG_SUBDIR_FULL["sickkids_image"],
        constants.DSET_TO_METADATA["raw"]["sickkids_image"],
    )


def main_clean_metadata():
    """
    Create clean metadata files for video and image datasets.
    """
    # Video datasets
    clean_sickkids_video_metadata()
    clean_stanford_video_metadata()

    # Image datasets
    clean_image_dsets_metadata()

    # Correct SickKids/Stanford view labels
    ref_path = constants.DSET_TO_METADATA["raw"]["sickkids_corrections"]
    main_correct_labels(ref_path, label_col="plane")

    # Video (beamform) datasets
    # NOTE: This needs to happen after label correction, since it takes after
    #       the cleaned video metadata
    clean_sickkids_video_beamform_metadata()
    clean_stanford_video_beamform_metadata()


def main_correct_labels(ref_path, label_col="plane"):
    """
    Correct labels for each dataset based on a reference file

    Parameters
    ----------
    ref_path : str
        Path to XLSX file containing modified labels and
        (dset, id, visit, seq_number) for every image that needs correction
    label_col : str
        Name of column whose label is corrected
    """
    assert label_col in LABEL_COLS, f"`label_col` must be one of {LABEL_COLS}"
    # Load file with label corrections
    df_correct = pd.read_excel(ref_path)
    # Convert to dictionary, mapping to new label
    index_cols = ["dset", "id", "visit", "seq_number"]
    df_correct[index_cols] = df_correct[index_cols].astype(str)
    idx_to_label = df_correct.set_index(index_cols)["new_label"].to_dict()

    # For each metadata, check if any of the labels overlap
    for dset in list(constants.VIDEO_DSETS) + list(constants.IMAGE_DSETS):
        # Exclude beamform video datasets
        if "_beamform" in dset:
            continue

        clean_dset_path = constants.DSET_TO_METADATA["clean"][dset]
        df_metadata = pd.read_csv(clean_dset_path)
        df_metadata[index_cols] = df_metadata[index_cols].astype(str)
        # Check if any rows will be modififed
        corrected_mask = df_metadata.apply(
            lambda row: tuple(row[col] for col in index_cols) in idx_to_label,
            axis=1,
        )
        # Skip, if no labels corrected
        if not corrected_mask.sum():
            continue

        # Modify labels
        print(f"Dataset: `{dset}` | Modifying {corrected_mask.sum()} labels!")
        df_metadata.loc[corrected_mask, label_col] = df_metadata[corrected_mask].apply(
            lambda row: idx_to_label[tuple(row[col] for col in index_cols)],
            axis=1,
        )

        # For rows whose new label is Bladder, set other label columns
        # NOTE: In case old side/plane label contains kidney information
        bladder_mask = df_metadata[label_col].isin(["Bladder", None])
        corrected_bladder_mask = corrected_mask & bladder_mask
        if corrected_bladder_mask.sum():
            other_label_cols = [col for col in LABEL_COLS if col != label_col]
            for col in other_label_cols:
                new_val = None if col == "side" else "Bladder"
                df_metadata.loc[corrected_bladder_mask, col] = new_val

        # Save changes
        df_metadata.to_csv(clean_dset_path, index=False)


def main_update_img_dirs():
    """
    If home directory or image data directories have changed, update metadata
    to point to new paths based on `src.data.constants` file.
    """
    print("[main_update_img_dirs] Assumes your $HOME has been changed and you're migrating data!")
    print("[main_update_img_dirs] Changing $HOME path hard-coded in metadata files!")

    # For each metadata, check if any of the labels overlap
    for dset in list(constants.VIDEO_DSETS) + list(constants.IMAGE_DSETS):
        # Get cleaned metadata
        clean_dset_path = constants.DSET_TO_METADATA["clean"][dset]
        df_metadata = pd.read_csv(clean_dset_path)

        # Remove old image directory from filename, if applicable
        if "dir_name" in df_metadata.columns.tolist():
            df_metadata["filename"] = df_metadata.apply(
                lambda row: row["filename"].replace(row["dir_name"], ""),
                axis=1
            )

        # Update with new image directory
        img_dir = constants.DSET_TO_IMG_SUBDIR_FULL[dset]
        df_metadata["dir_name"] = img_dir

        # Save changes
        df_metadata.to_csv(clean_dset_path, index=False)


if __name__ == "__main__":
    # Add command-line interface
    fire.Fire({
        "prep_img_dsets": main_prepare_image_datasets,
        "clean_metadata": main_clean_metadata,
        "correct_labels": main_correct_labels,
        "update_img_dirs": main_update_img_dirs,
    })
