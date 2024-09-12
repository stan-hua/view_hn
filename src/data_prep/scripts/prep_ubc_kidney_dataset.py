"""
prep_ubc_kidney_dataset.py

Description: Script to prepare image and metadata for UBC kidney dataset
"""

# Standard libraries
import glob
import json
import os

# Non-standard libraries
import cv2
import fire
import pandas as pd
from tqdm import tqdm

# Custom libraries
from src.data import constants
from src.data_prep import utils


# Set up `progress_apply` for pandas
tqdm.pandas()


################################################################################
#                                  Functions                                   #
################################################################################
def create_ubc_adult_image_metadata():
    """
    Create (raw) UBC adult kidney dataset metadata

    Note
    ----
    Assumes raw images are in `.../ViewLabeling/UBC_AdultKidneyDataset/raw/`
    Assumes label files are in `.../ViewLabeling/Datasheets/raw/ubc_adult`

    """
    print("[UBC Adult] Processing metadata...")
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

    # Create placeholder label with unknown side
    # NOTE: Requires manually annotating side of each file
    df_metadata["side"] = df_metadata["view"].map(label_map.get)
    # Drop all images that aren't Sagittal/Transverse views
    mask = df_metadata["plane"].isna()
    df_metadata = df_metadata[~mask]
    print(f"[UBC Adult] Dropping `{mask.sum()}` images with non - SAG/TRANS view images...")

    # Create inferred ID, visit and sequence number from filename
    df_metadata["id"] = df_metadata["filename"].map(
        lambda x: "_".join([
            x.split("_")[0],
            x.split("_")[1].split("-")[1],
            x.split("_")[1].split("-")[2],
    ]))
    df_metadata["visit"] = 0
    df_metadata["seq_number"] = 0

    # NOTE: Below assumes that images were preprocessed into clean directory
    # Set directory name as new directory
    df_metadata["dir_name"] = constants.DSET_TO_IMG_SUBDIR_FULL["ubc_adult"]

    # Rename filename
    df_metadata["filename"] = df_metadata.apply(
        lambda row: row["id"] + ".png",
        axis=1
    )

    # Add placeholder side and create dummy side
    df_metadata["side"] = "Unknown"
    df_metadata["label"] = df_metadata.apply(
        lambda row: "_".join([row["plane"], row["side"]]),
        axis=1
    )

    # Assign 75-25 train/test split
    df_metadata = utils.assign_split_table(df_metadata, train_split=0.75)

    # Save metadata
    cols = ["split", "filename", "id", "visit", "seq_number", "quality", "label", "plane", "side", "dir_name"]
    df_metadata = df_metadata[cols]
    df_metadata = df_metadata.sort_values(by=["id", "visit", "seq_number"])
    df_metadata.to_csv(constants.DSET_TO_METADATA["raw"]["ubc_adult"],
                       index=False)
    print("[UBC Adult] Processing metadata...DONE")


def preprocess_ubc_adult_images():
    """
    Preprocess UBC adult kidney dataset:
        (1) Center crop with 1:1 aspect ratio and resize to 256x256,
        (2) Histogram equalize image
    """
    print("[UBC Adult] Processing images...")
    raw_img_dir = constants.DSET_TO_IMG_SUBDIR_FULL["ubc_adult_raw"]
    clean_img_dir = constants.DSET_TO_IMG_SUBDIR_FULL["ubc_adult"]

    for img_path in glob.glob(os.path.join(raw_img_dir, "*.png")):
        # Read and preprocess image as RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # NOTE: Crop scale is chosen to match 150x150 previous crop
        processed_img = utils.preprocess_image(
            img, crop_scale=0.65, resize_dims=(256, 256))

        # Create new filename from inferred ID, visit and sequence number
        filename = os.path.basename(img_path)
        filename = "_".join([
            filename.split("_")[0],
            filename.split("_")[1].split("-")[1],
            filename.split("_")[1].split("-")[2],
        ]) + ".png"

        # Store image
        new_path = os.path.join(clean_img_dir, filename)
        cv2.imwrite(new_path, processed_img)
    print("[UBC Adult] Processing images...DONE")


if __name__ == "__main__":
    # Add command-line options
    fire.Fire({
        "prep_metadata": create_ubc_adult_image_metadata,
        "prep_images": preprocess_ubc_adult_images,
    })
