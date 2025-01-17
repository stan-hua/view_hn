"""
segment_prep_script.py

Description: Used to extract foreground and background from image segmentations.
"""

# Standard libraries
import glob
import os
from collections import defaultdict

# Non-standard libraries
import cv2
import pandas as pd

# Custom libraries
from config import constants
from src.data_prep import utils


################################################################################
#                                  Constants                                   #
################################################################################
SEGMENT_TO_COLOR_BOUNDS = {
    "bladder": ((0, 180, 0), (160, 255, 160)),
    "kidney": ((0, 0, 209), (190, 190, 255)),
    "hn": ((180, 0, 0), (255, 160, 160))
}


################################################################################
#                               Helper Functions                               #
################################################################################
def get_source_to_segmented_filenames(paths):
    """
    Given all image file paths, find segmented images (given by filename suffix)
    and create a mapping of {source filename : [segmented image filenames]}.

    Parameters
    ----------
    paths : list
        List of image file paths

    Returns
    -------
    dict
        Maps source filename to list of filenames of corresponding segmented
        images.
    """
    # Extract filenames from possible paths
    filenames = [os.path.basename(path) for path in paths]

    # Get mapping of original filename to corresponding segmentations
    src_to_segments = defaultdict(list)
    for filename in filenames:
        parts = filename.split("_")
        if len(parts) == 3:
            continue

        # NOTE: Ignore liver for now
        if "liver" in filename:
            continue

        src_filename = "_".join(parts[:3]) + ".jpg"
        src_to_segments[src_filename].append(filename)

    return src_to_segments


def load_metadata_with_segmentation(segment_files):
    """
    Attempt to get labels for segmented image files.

    Parameters
    ----------
    segment_files : list
        Filenames of segmented images

    Returns
    -------
    pandas.DataFrame
        Contains image metadata (label, patient ID, visit number, seq. number)
    """
    # Extract filenames from possibly paths
    df_filenames = pd.DataFrame({"filename": segment_files})
    df_filenames.filename = df_filenames.filename.map(os.path.basename)

    # Load metadata
    df_metadata = utils.load_metadata("sickkids")

    # Temporarily set index for table join
    df_filenames = df_filenames.set_index("filename")
    df_metadata = df_metadata.set_index("filename")
    df_segments_with_labels = df_filenames.join(df_metadata, how="inner")

    return df_segments_with_labels.reset_index()


def get_segmented_type(filename):
    """
    Return item segmented in image from filename.

    Parameters
    ----------
    filename : str
        Filename of image. If segmented, must contain at least one of
        ("bseg", "kseg", "cseg")

    Returns
    -------
    str
        Item segmented (bladder, kidney, hn), or None if not found
    """
    if "bseg" in filename:
        return "bladder"
    elif "kseg" in filename:
        return "kidney"
    elif "cseg" in filename:
        return "hn"
    return None


def get_fg_and_bg_of_segmentation(src_img, segment_img,
                                  lower_bound, upper_bound):
    """
    Get foreground and background of image segmentation.

    Parameters
    ----------
    src_img : numpy.array
        Source ultrasound image (BGR format)
    segment_img : numpy.array
        Source ultrasound image with segmentation (BGR format)
    lower_bound : tuple of (int, int, int)
        Contains lower bound on BGR pixel values
    upper_bound : tuple of (int, int, int)
        Contains upper bound on BGR pixel values

    Returns
    -------
    tuple of (numpy.array, numpy.array)
        Contains (foreground image, background image)
    """
    # Extract foreground
    foreground_mask = cv2.inRange(segment_img, lower_bound, upper_bound)
    foreground = cv2.bitwise_and(src_img, src_img, mask=foreground_mask)

    # Extract background
    background_mask = cv2.bitwise_not(foreground_mask)
    background = cv2.bitwise_and(src_img, src_img, mask=background_mask)

    return foreground, background


################################################################################
#                                    Script                                    #
################################################################################
if __name__ == '__main__':
    paths = glob.glob(os.path.join(constants.DIR_SEGMENT, "*"))
    src_to_segments = get_source_to_segmented_filenames(paths)

    # For kidney/bladder segmentation
    # Identify foreground
    # Identify background
    for src_file, segment_files in src_to_segments.items():
        # Load original image
        src_img = cv2.imread(f"{constants.DIR_SEGMENT}/{src_file}",
                             cv2.IMREAD_GRAYSCALE)

        # Perform preprocessing on source image
        src_img_proc = utils.preprocess_image(src_img, ignore_crop=True)

        # Go through segmentations
        for segment_file in segment_files:
            # Load segmented image
            segment_img = cv2.imread(f"{constants.DIR_SEGMENT}/{segment_file}")

            # Get upper/lower bound BGR values in segmentation
            item_segmented = get_segmented_type(segment_file)
            lower_bound, upper_bound = SEGMENT_TO_COLOR_BOUNDS[item_segmented]

            # Save mask to file
            mask_save_path = os.path.join(constants.DIR_SEGMENT_PROC,
                                        "mask", src_file)
            if not os.path.exists(fg_save_path):
                mask = cv2.inRange(segment_img, lower_bound, upper_bound)
                cv2.imwrite(mask_save_path, mask)

            # Get foreground/background of PROCESSED source image
            foreground, background = get_fg_and_bg_of_segmentation(
                src_img_proc, segment_img,
                lower_bound, upper_bound
            )

            # If hydro, save only background
            if item_segmented == "hn":
                save_path = os.path.join(constants.DIR_SEGMENT_PROC,
                                         "background_hydro", src_file)
                cv2.imwrite(save_path, background)
            else:
                # Save foreground
                fg_save_path = os.path.join(constants.DIR_SEGMENT_PROC,
                                            "foreground", src_file)
                cv2.imwrite(fg_save_path, foreground)

                # Save background
                bg_save_path = os.path.join(constants.DIR_SEGMENT_PROC,
                                            "background", src_file)
                cv2.imwrite(bg_save_path, background)
