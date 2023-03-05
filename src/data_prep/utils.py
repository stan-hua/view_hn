"""
utils.py

Description: Contains helper functions for a variety of data/label preprocessing
             functions.
"""

# Standard libraries
import glob
import logging
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
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)


################################################################################
#                             Metadata Extraction                              #
################################################################################
def load_metadata(hospital, prepend_img_dir=False, **kwargs):
    """
    Loads metadata for a specific hospital

    Parameters
    ----------
    hospital : str
        One of ("sickkids", "stanford", "uiowa", "chop")
    prepend_img_dir : bool, optional
        If True, prepends default image directory for hospital to "filename"
        column, by default False
    **kwargs : dict, optional
        Keyword arguments to pass into specific metadata loading function.

    Returns
    -------
    pd.DataFrame
        May contain metadata (filename, view label, patient id, visit, sequence
        number).
    """
    assert hospital in ("sickkids", "stanford", "uiowa", "chop")

    # Mapping of hospital to metadata loading function
    hospital_to_func = {
        "sickkids": load_sickkids_metadata,
        "stanford": load_stanford_metadata,
        "uiowa": load_uiowa_metadata,
        "chop": load_chop_metadata,
    }

    # Retrieve metadata
    df_metadata = hospital_to_func[hospital](**kwargs)

    # If specified, attempt to prepend default image directory to filename
    if prepend_img_dir:
        img_dir = constants.HOSPITAL_TO_IMG_DIR[hospital]
        df_metadata["filename"] = df_metadata["filename"].map(
            lambda x: os.path.join(img_dir, x)
        )

    # Attach hospital name to data
    df_metadata["hospital"] = hospital

    return df_metadata


# TODO: Add second test path
def load_sickkids_metadata(path=constants.SK_METADATA_FILE,
                           label_part=None,
                           extract=False,
                           include_hn=False,
                           relative_side=False,
                           include_unlabeled=False,
                           img_dir=constants.DIR_IMAGES,
                           include_test_set=False,
                           test_path=constants.SK_TEST_METADATA_FILE):
    """
    Load SickKids metadata table with filenames and view labels.

    Note
    ----
    If <relative_side> is True, the following examples happens:
        - [Sagittal_Left, Transverse_Right, Bladder] ->
                [Sagittal_First, Transverse_Second, Bladder]
        - [Sagittal_Right, Transverse_Left, Bladder] ->
                [Sagittal_First, Transverse_Second, Bladder]

    Parameters
    ----------
    path : str, optional
        Path to CSV metadata file, by default constants.SK_METADATA_FILE
    label_part : str, optional
        If specified, either `side` or `plane` is extracted from each label
        and used as the given label, by default None.
    extract : bool, optional
        If True, extracts patient ID, US visit, and sequence number from the
        filename, by default False.
    include_hn : bool, optional
        If True, include all available hydronephrosis and surgery labels, by
        default False.
    relative_side : bool, optional
        If True, converts side (Left/Right) to order in which side appeared
        (First/Second/None). Requires <extract> to be True, by default False.
    include_unlabeled : bool, optional
        If True, include all unlabeled images in <img_dir>, by default False.
    img_dir : str, optional
        Directory containing unlabeled (and labeled) images, by default
        constants.DIR_IMAGES
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

    # Fix mislabel saggital --> sagittal
    fix_label_map = {"Saggital_Left": "Sagittal_Left",
                     "Saggital_Right": "Sagittal_Right"}
    df_metadata.label = df_metadata.label.map(lambda x: fix_label_map.get(x, x))

    # Change label to side/plane, if specified
    if label_part:
        df_metadata["label"] = df_metadata["label"].map(
            lambda x: extract_from_label(x, extract=label_part))

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
        df_metadata = extract_data_from_filename(df_metadata)

        # Include all available surgery labels
        if include_hn:
            df_metadata = extract_hn_labels(
                df_metadata, sickkids=True, stanford=False)

        # Convert side (in label) to order of relative appearance (First/Second)
        if relative_side:
            df_metadata = df_metadata.sort_values(
                by=["id", "visit", "seq_number"])
            relative_labels = df_metadata.groupby(by=["id", "visit"])\
                .apply(lambda df: pd.Series(make_side_label_relative(
                    df.label.tolist()))).to_numpy()
            df_metadata["label"] = relative_labels

    return df_metadata


def load_stanford_metadata(path=constants.SU_METADATA_FILE,
                           label_part=None,
                           extract=False,
                           include_hn=False,
                           relative_side=False,
                           include_unlabeled=False,
                           img_dir=constants.DIR_IMAGES):
    """
    Load Stanford metadata table with filenames and view labels.

    Note
    ----
    If <relative_side> is True, the following examples happens:
        - [Sagittal_Left, Transverse_Right, Bladder] ->
                [Sagittal_First, Transverse_Second, Bladder]
        - [Sagittal_Right, Transverse_Left, Bladder] ->
                [Sagittal_First, Transverse_Second, Bladder]

    Parameters
    ----------
    path : str, optional
        Path to CSV metadata file, by default constants.STANFORD_METADATA_FILE
    label_part : str, optional
        If specified, either `side` or `plane` is extracted from each label
        and used as the given label, by default None.
    extract : bool, optional
        If True, extracts patient ID, US visit, and sequence number from the
        filename, by default False.
    include_hn : bool, optional
        If True, include all available hydronephrosis and surgery labels, by
        default False.
    relative_side : bool, optional
        If True, converts side (Left/Right) to order in which side appeared
        (First/Second/None). Requires <extract> to be True, by default False.
    include_unlabeled : bool, optional
        If True, include all unlabeled images in <img_dir>, by default False.
    img_dir : str, optional
        Directory containing unlabeled (and labeled) images, by default
        constants.DIR_IMAGES

    Returns
    -------
    pandas.DataFrame
        May contain metadata (filename, view label, patient id, visit, sequence
        number).
    """
    df_metadata = pd.read_csv(path)

    # Fix mislabel saggital --> sagittal
    fix_label_map = {"Saggital_Left": "Sagittal_Left",
                     "Saggital_Right": "Sagittal_Right"}
    df_metadata.label = df_metadata.label.map(lambda x: fix_label_map.get(x, x))

    # Change label to side/plane, if specified
    if label_part:
        df_metadata["label"] = df_metadata["label"].map(
            lambda x: extract_from_label(x, extract=label_part))

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
        df_metadata = extract_data_from_filename(df_metadata)

        # Include all available surgery labels
        if include_hn:
            df_metadata = extract_hn_labels(
                df_metadata, sickkids=False, stanford=True)

        # Convert side (in label) to order of relative appearance (First/Second)
        if relative_side:
            df_metadata = df_metadata.sort_values(
                by=["id", "visit", "seq_number"])
            relative_labels = df_metadata.groupby(by=["id", "visit"])\
                .apply(lambda df: pd.Series(make_side_label_relative(
                    df.label.tolist()))).to_numpy()
            df_metadata["label"] = relative_labels

    # Drop duplicates
    # NOTE: Metadata table contains duplicate rows
    df_metadata = df_metadata.drop_duplicates()

    return df_metadata


def load_uiowa_metadata(path=constants.UIOWA_METADATA_FILE,
                        label_part=None,
                        extract=False,
                        relative_side=False,
                        include_hn=False,
                        include_unlabeled=False,
                        **kwargs,
                        ):
    """
    Load UIOwa metadata table with filenames and view labels.

    Note
    ----
    If <relative_side> is True, the following examples happens:
        - [Sagittal_Left, Transverse_Right, Bladder] ->
                [Sagittal_First, Transverse_Second, Bladder]
        - [Sagittal_Right, Transverse_Left, Bladder] ->
                [Sagittal_First, Transverse_Second, Bladder]

    Parameters
    ----------
    path : str, optional
        Path to CSV metadata file, by default constants.UIOWA_METADATA_FILE
    label_part : str, optional
        If specified, either `side` or `plane` is extracted from each label
        and used as the given label, by default None.
    extract : bool, optional
        If True, extracts patient ID, US visit, and sequence number from the
        filename, by default False.
    relative_side : bool, optional
        If True, converts side (Left/Right) to order in which side appeared
        (First/Second/None). Requires <extract> to be True, by default False.

    Returns
    -------
    pandas.DataFrame
        May contain metadata (filename, view label, patient id, visit, sequence
        number).
    """
    # Raise error, if asked to include unlabeled images
    if include_unlabeled:
        raise NotImplementedError("Getting unlabeled images from UIowa is "
                                  "currently not supported!")
    # Raise error, if asked to provide surgery labels
    if include_hn:
        raise NotImplementedError("UIowa has no HN labels!")

    # Load metadata
    df_metadata = pd.read_csv(path)

    # Change label to side/plane, if specified
    if label_part:
        df_metadata["label"] = df_metadata["label"].map(
            lambda x: extract_from_label(x, extract=label_part))

    # Since other extracted metadata was already saved to the metadata file,
    # simply remove extra metadata if not specified
    if not extract:
        df_metadata = df_metadata.drop(columns=["id", "visit", "seq_number"])
    # Convert side (in label) to order of relative appearance (First/Second)
    elif relative_side:
        df_metadata = df_metadata.sort_values(
            by=["id", "visit", "seq_number"])
        relative_labels = df_metadata.groupby(by=["id", "visit"])\
            .apply(lambda df: pd.Series(make_side_label_relative(
                df.label.tolist()))).to_numpy()
        df_metadata["label"] = relative_labels

    return df_metadata


def load_chop_metadata(path=constants.CHOP_METADATA_FILE,
                       label_part=None,
                       extract=False,
                       relative_side=False,
                       include_hn=False,
                       include_unlabeled=False,
                       **kwargs
                       ):
    """
    Load CHOP metadata table with filenames and view labels.

    Note
    ----
    If <relative_side> is True, the following examples happens:
        - [Sagittal_Left, Transverse_Right, Bladder] ->
                [Sagittal_First, Transverse_Second, Bladder]
        - [Sagittal_Right, Transverse_Left, Bladder] ->
                [Sagittal_First, Transverse_Second, Bladder]

    Parameters
    ----------
    path : str, optional
        Path to CSV metadata file, by default constants.CHOP_METADATA_FILE
    label_part : str, optional
        If specified, either `side` or `plane` is extracted from each label
        and used as the given label, by default None.
    extract : bool, optional
        If True, extracts patient ID, US visit, and sequence number from the
        filename, by default False.
    relative_side : bool, optional
        If True, converts side (Left/Right) to order in which side appeared
        (First/Second/None). Requires <extract> to be True, by default False.

    Returns
    -------
    pandas.DataFrame
        May contain metadata (filename, view label, patient id, visit, sequence
        number).
    """
    # Raise error, if asked to include unlabeled images
    if include_unlabeled:
        raise NotImplementedError("Getting unlabeled images from UIowa is "
                                  "currently not supported!")
    # Raise error, if asked to provide surgery labels
    if include_hn:
        raise NotImplementedError("CHOP has no HN labels!")

    # Load metadata
    df_metadata = pd.read_csv(path)

    # Change label to side/plane, if specified
    if label_part:
        df_metadata["label"] = df_metadata["label"].map(
            lambda x: extract_from_label(x, extract=label_part))

    # Since other extracted metadata was already saved to the metadata file,
    # simply remove extra metadata if not specified
    if not extract:
        df_metadata = df_metadata.drop(columns=["id", "visit", "seq_number"])
    # Convert side (in label) to order of relative appearance (First/Second)
    elif relative_side:
        df_metadata = df_metadata.sort_values(
            by=["id", "visit", "seq_number"])
        relative_labels = df_metadata.groupby(by=["id", "visit"])\
            .apply(lambda df: pd.Series(make_side_label_relative(
                df.label.tolist()))).to_numpy()
        df_metadata["label"] = relative_labels

    return df_metadata


def extract_data_from_filename(df_metadata, col="filename"):
    """
    Extract metadata from each image's filename and combines with HN labels.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    col : str, optional
        Name of column containing filename, by default "filename".

    Returns
    -------
    Metadata table with additional data extracted
    """
    # Create copy to avoid in-place assignment
    df_metadata = df_metadata.copy()

    # Extract data from filenames
    df_metadata["basename"] = df_metadata[col].map(os.path.basename)
    df_metadata["id"] = df_metadata["basename"].map(
            lambda x: x.split("_")[0])
    df_metadata["visit"] = df_metadata["basename"].map(
        lambda x: x.split("_")[1])
    df_metadata["seq_number"] = df_metadata["basename"].map(
        lambda x: int(x.split("_")[2].replace(".jpg", "")))
    df_metadata.drop(columns=["basename"], inplace=True)

    return df_metadata


def extract_data_from_filename_and_join(df_metadata, hospital="sickkids",
                                        **kwargs):
    """
    Extract extra metadata

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image, but is missing
        extra desired metadata
    hospital : str or list, optional
        Name of hospital/s in `df_metadata`, by default "sickkids".
    **kwargs : dict, optional
        Keyword arguments to pass into `load_metadata`.

    Returns
    -------
    Metadata table with additional data extracted
    """
    # INPUT: If only 1 hospital provided, ensure its a list
    hospitals = [hospital] if isinstance(hospital, str) else hospital

    # Create copy to avoid in-place assignment
    df_metadata = df_metadata.copy()

    # Load metadata from ALL hospitals specified
    hospitals_metadata = []
    for hospital in hospitals:
        curr_metadata = load_metadata(hospital=hospital, extract=True, **kwargs)
        hospitals_metadata.append(curr_metadata)
    df_metadata_all = pd.concat(hospitals_metadata)

    # If no overlapping filenames, try loading all metadata WITH image directory
    if not contains_overlapping_vals(df_metadata, df_metadata_all, "filename"):
        df_metadata_all = load_metadata(
            hospital=hospital,
            extract=True,
            prepend_img_dir=True)

    # Raise RuntimeError, if filenames still don't overlap
    if not contains_overlapping_vals(df_metadata, df_metadata_all, "filename"):
        raise RuntimeError("Filenames in provided metadata table and source "
                           "metadata table do NOT overlap!")

    # Check columns that can be used for identification
    idx_cols = ["filename", "id", "visit", "seq_number"]
    idx_cols = [col for col in idx_cols if col in df_metadata.columns.tolist()]

    # Perform LEFT JOIN on identifying columns
    df_metadata = left_join_filtered_to_source(
        df_metadata, df_metadata_all,
        index_cols=idx_cols)

    return df_metadata


def extract_hn_labels(df_metadata, sickkids=True, stanford=True):
    """
    Extract labels for hydronephrosis (HN) and surgery, if available, and insert
    them into the supplied metadata table in-place.

    Note
    ----
    Unlabeled will appear as null values.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    sickkids : bool, optional
        If True, loads in HN labels for SickKids data, by default True.
    stanford : bool, optional
        If True, loads in HN labels for Stanford data, by default True.

    Returns
    -------
    pandas.DataFrame
        Metadata table with HN and surgery labels filled in.
    """
    # Load SickKids HN labels
    df_hn = pd.DataFrame()
    if sickkids:
        df_hn = pd.concat([df_hn, pd.read_csv(constants.SK_HN_METADATA_FILE)])
    if stanford:
        df_hn = pd.concat([df_hn, pd.read_csv(constants.SU_HN_METADATA_FILE)])

    # Ensure ID is string
    df_hn["id"] = df_hn["id"].astype(str)
    df_metadata["id"] = df_metadata["id"].astype(str)

    # Join to df_metadata
    df_metadata = pd.merge(df_metadata, df_hn, how="left", on=["id", "side"])

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
    assert extract in ("plane", "side")
    label_parts = label.split("_")
    if extract == "side":
        return label_parts[1] if len(label_parts) > 1 else "None"
    return label_parts[0]


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


def get_labels_for_filenames(filenames, sickkids=True, stanford=True,
                             label_part=None,
                             **kwargs):
    """
    Attempt to get labels for all filenames given, using metadata file

    Parameters
    ----------
    filenames : list or array-like or pandas.Series
        List of filenames
    sickkids : bool, optional
        If True, include SickKids image labels, by default True.
    stanford : bool, optional
        If True, include Stanford image labels, by default True.
    label_part : str, optional
        If specified, either `side` or `plane` is extracted from each label
        and used as the given label, by default None.
    **kwargs : dict, optional
        Keyword arguments to pass into hospital-specific metadata-loading
        functions.

    Returns
    -------
    numpy.array
        List of view labels. For filenames not found, label will None.
    """
    df_labels = pd.DataFrame()

    # Get SickKids metadata
    if sickkids:
        df_labels = pd.concat([df_labels, load_sickkids_metadata(**kwargs)],
                              ignore_index=True)

    # Get Stanford metadata
    if stanford:
        df_labels = pd.concat([df_labels, load_stanford_metadata()],
                              ignore_index=True)

    # If specified, extract specific label part
    if label_part:
        df_labels["label"] = df_labels["label"].map(
            lambda x: extract_from_label(x, extract=label_part))

    # Get mapping of filename to labels
    filename_to_label = dict(zip(df_labels["filename"], df_labels["label"]))

    # Extract only filename
    filenames = [os.path.basename(filename) for filename in filenames]

    # Get labels
    view_labels = np.array([*map(filename_to_label.get, filenames)])

    return view_labels


def get_machine_for_filenames(filenames, sickkids=True):
    """
    Attempt to get machine labels for all filenames given, using metadata file

    Parameters
    ----------
    filenames : list or array-like or pandas.Series
        List of filenames
    sickkids : bool, optional
        If True, include SickKids image labels, by default True.

    Returns
    -------
    numpy.array
        List of machine labels. For filenames not found, label will None.
    """
    filename_to_machine = {}
    if sickkids:
        # Get mapping of filename to machine
        df_machines = pd.concat([
            pd.read_csv(constants.SK_MACHINE_METADATA_FILE),
            pd.read_csv(constants.SK_MACHINE_TEST_METADATA_FILE)
        ])
        df_machines = df_machines.set_index("IMG_FILE")
        filename_to_machine.update(df_machines["machine"].to_dict())

    # Get machine label for each filename
    machine_labels = np.array([*map(filename_to_machine.get, filenames)])

    return machine_labels


def get_label_boundaries(df_metadata, min_block_size=3):
    """
    Provide mask for samples at the boundaries of contiguous label blocks for
    each unique ultrasound sequence.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    min_block_size : int, optional
        Minimum number of images of the same label to be a valid contiguous
        label block, to be searched for the images at the boundaries

    Returns
    -------
    pd.Series
        Boolean mask, where True signifies image is at the boundary of two
        contiguous label blocks
    """
    def _find_boundaries(df_seq):
        """
        Given metadata for a unique ultrasound sequence, identify boundaries
        between contiguous label blocks.

        Parameters
        ----------
        df_seq : pandas.DataFrame
            Contains metadata for all ultrasound images for one unique sequence

        Returns
        -------
        pd.Series
            Boolean mask at a per-sequence level
        """
        # Get indices to sort/unsort by sequence number
        sort_idx, unsort_idx = argsort_unsort(df_seq["seq_number"])
        df_seq = df_seq.iloc[sort_idx]

        # Identify boundaries of label blocks
        all_labels = df_seq["label"]
        last_label = None
        curr_block_size = 0
        boundary_mask = []
        for label in all_labels:
            # Case 1: If same label as last
            if label == last_label:
                curr_block_size += 1
                continue

            # Case 2: If new label and last block size was >= the minimum
            last_block_mask = [False]*curr_block_size
            if curr_block_size >= min_block_size:
                last_block_mask[0] = True
                last_block_mask[-1] = True
            # Update accumulators
            boundary_mask.extend(last_block_mask)
            last_label = label
            curr_block_size = 1

        # NOTE: Boundary mask for last block wasn't updated
        if len(boundary_mask) != len(all_labels):
            last_block_mask = [False]*curr_block_size
            if curr_block_size >= min_block_size:
                last_block_mask[0] = True
                last_block_mask[-1] = True
            boundary_mask.extend(last_block_mask)

        # Unsort mask
        boundary_mask = np.array(boundary_mask)
        return boundary_mask[unsort_idx]

    mask = df_metadata.groupby(by=["id", "visit"]).apply(_find_boundaries)

    # Flatten mask
    mask = np.concatenate(mask.values)

    return mask


def identify_not_in_label_group(df_metadata, min_block_size=3):
    """
    Provide mask for samples not part of a contiguous label block for
    each unique ultrasound sequence.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains metadata for an ultrasound image.
    min_block_size : int, optional
        Minimum number of images of the same label to be a valid contiguous
        label block, to be searched for the images at the boundaries

    Returns
    -------
    pd.Series
        Boolean mask, where True signifies image is in a contiguous label block
    """
    def _find_blocks(df_seq):
        """
        Given metadata for a unique ultrasound sequence, identify boundaries
        between contiguous label blocks.

        Parameters
        ----------
        df_seq : pandas.DataFrame
            Contains metadata for all ultrasound images for one unique sequence

        Returns
        -------
        pd.Series
            Boolean mask at a per-sequence level
        """
        # Get indices to sort/unsort by sequence number
        sort_idx, unsort_idx = argsort_unsort(df_seq["seq_number"])
        df_seq = df_seq.iloc[sort_idx]

        # Identify boundaries of label blocks
        all_labels = df_seq["label"]
        last_label = None
        curr_block_size = 0
        boundary_mask = []
        for label in all_labels:
            # Case 1: If same label as last
            if label == last_label:
                curr_block_size += 1
                continue

            # Case 2: If new label and last block size was >= the minimum
            last_block_mask = [True]*curr_block_size
            if curr_block_size >= min_block_size:
                last_block_mask = [False for _ in range(len(last_block_mask))]
            # Update accumulators
            boundary_mask.extend(last_block_mask)
            last_label = label
            curr_block_size = 1

        # NOTE: Boundary mask for last block wasn't updated
        if len(boundary_mask) != len(all_labels):
            last_block_mask = [False]*curr_block_size
            if curr_block_size >= min_block_size:
                last_block_mask[0] = True
                last_block_mask[-1] = True
            boundary_mask.extend(last_block_mask)

        # Unsort mask
        boundary_mask = np.array(boundary_mask)
        return boundary_mask[unsort_idx]

    mask = df_metadata.groupby(by=["id", "visit"]).apply(_find_blocks)

    # Flatten mask
    mask = np.concatenate(mask.values)

    return mask


################################################################################
#                           Metadata Post-Processing                           #
################################################################################
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


def restrict_seq_len(df_metadata, n=18):
    """
    Filter metadata for ultrasound image sequences, to restrict each sequence to
    have exactly `n` images.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Each row contains an `id` and `visit`, which specify a unique US
        sequence.
    n : int, optional
        Absolute sequence length to restrict to, by default 18.

    Returns
    -------
    pandas.DataFrame
        Filtered metadata table
    """
    def downsample_df(df):
        """
        Downsample dataframe with consecutive rows.

        Parameters
        ----------
        df : pandas.DataFrame
            Metadata table with n+ rows

        Returns
        -------
        pandas.DataFrame
            Downsampled metadata dataframe
        """
        if len(df) == n:
            return df
        start_idx = np.random.randint(0, high=len(df)-n)
        return df.iloc[start_idx: start_idx+n]

    # Create copy to avoid in-place operations
    df_metadata = df_metadata.copy()

    # Create identifiers for ultrasound sequences
    # NOTE: Unique sequences are identified by patient ID and hospital visit
    seq_ids = pd.Series(
        df_metadata[["id", "visit"]].itertuples(index=False, name=None))

    # Get number of images per US sequence
    seq_counts = seq_ids.value_counts()
    initial_num_seq = len(seq_counts)

    # Filter for minimum length sequences
    seq_counts = seq_counts[seq_counts >= n]
    filtered_num_seq = len(seq_counts)
    kept_idx = set(seq_counts.index.tolist())
    idx = seq_ids.map(lambda x: x in kept_idx)
    df_metadata = df_metadata[idx]

    # Log number of sequences that decreased
    LOGGER.info(f"Filter for Sequences w/ {n}+ Images: "
                f"{initial_num_seq} -> {filtered_num_seq}")

    # Downsample sequences with > `n` images
    df_metadata = df_metadata.groupby(by=["id", "visit"]).apply(
        downsample_df)

    return df_metadata


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
    # If specified, crop image
    if not ignore_crop:
        height, width = img.shape[0], img.shape[1]

        # Sanitize input to be less than or equal to max width/height
        crop_height = min(crop_dims[0], height)
        crop_width = min(crop_dims[1], img.shape[1])

        # Get midpoints and necessary distance from midpoints
        mid_x, mid_y = int(width/2), int(height/2)
        half_crop_height, half_crop_width = int(crop_width/2), int(crop_height/2)

        # Crop image
        img = img[mid_y-half_crop_width:mid_y+half_crop_width,
                  mid_x-half_crop_height:mid_x+half_crop_height]

    # Resize image
    resized_img = cv2.resize(img, resize_dims)

    # Histogram normalize images
    equalized_img = equalize_hist(resized_img)

    # Scale back to 255
    processed_img = 255 * equalized_img

    return processed_img


################################################################################
#                        Miscellaneous Helper Functions                        #
################################################################################
def argsort_unsort(arr):
    """
    Given an array to sort, return indices to sort and unsort the array.

    Note
    ----
    Given arr = [C, A, B, D] and its index array be [0, 1, 2, 3].
        Sort indices: [2, 0, 1, 3] result in [A, B, C, D] & [1, 2, 0, 3],
        respectively.

    To unsort, sort index array initially sorted by `arr`
        Initial index: [1, 2, 0, 3]
        Sorted indices: [2, 0, 1, 3] result in arr = [C, A, B, D]

    Parameters
    ----------
    arr : pd.Series or np.array
        Array of items to sort

    Returns
    -------
    tuple of (np.array, np.array)
        First array contains indices to sort the array.
        Second array contains indices to unsort the sorted array.
    """
    sort_idx = np.argsort(arr)
    unsort_idx = np.argsort(np.arange(len(arr))[sort_idx])

    return sort_idx, unsort_idx


def left_join_filtered_to_source(df_filtered, df_full, index_cols=None):
    """
    Given the same table (with filtered columns) and its full table, perform a
    LEFT JOIN to get all columns data from the full table.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Table with columns removed
    df_full : pd.DataFrame
        Source table with all columns present
    index_cols : list, optional
        If provided, reindex both tables on this index before joining.

    Returns
    -------
    pd.DataFrame
        Filtered table with all columns data
    """
    # If specified, set index
    if index_cols:
        df_filtered = df_filtered.set_index(index_cols)
        df_full = df_full.set_index(index_cols)

    # Perform LEFT JOIN
    df_filtered = df_filtered.join(df_full, how="left", rsuffix="dup__")

    # Remove duplicate columnss
    drop_cols = [col for col in df_filtered.columns if col.startswith("dup__")]
    df_filtered = df_filtered.drop(columns=drop_cols)

    # Reset index, if earlier specified
    if index_cols:
        df_filtered = df_filtered.reset_index()

    return df_filtered


def contains_overlapping_vals(df_a, df_b, column):
    """
    Checks if values overlap in column `column` between dataframes.

    Parameters
    ----------
    df_a : pd.DataFrame
        Arbitrary table A with column `column`
    df_b : pd.DataFrame
        Arbitrary table B with column `column`
    column : str
        Name of column to check for overlapping values

    Returns
    -------
    bool
        Returns True if values overlap between dataframes in column `column`,
        and False, otherwise.
    """
    # Get unique values
    uniq_vals_a = set(df_a[column])
    uniq_vals_b = set(df_b[column])

    return len(uniq_vals_a.intersection(uniq_vals_b)) != 0
