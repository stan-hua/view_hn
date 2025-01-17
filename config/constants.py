"""
constants.py

Description: Stores global constants accessible anywhere in the repository.
"""

# Standard libraries
import os
from os.path import dirname, join
from collections import defaultdict


################################################################################
#                                  Debugging                                   #
################################################################################
DEBUG = True

SEED = 42

# Specify GPU ("cuda") or CPU ("cpu")
DEVICE = "cuda"

################################################################################
#                                  Data Paths                                  #
################################################################################
# Project directories to load data (images, metadata)
DIR_HOME = os.environ["HOME"]
# DIR_DATA = DIR_HOME + "SickKids/Lauren Erdman - HN_Stanley/ViewLabeling/"     # SickKids OneDrive path
DIR_PROJECT = dirname(dirname(__file__))
DIR_DATA = os.environ.get("DIR_DATA", join(DIR_PROJECT, "data"))
assert os.path.exists(DIR_DATA), "Data directory does not exist!"

# Metadata directories
DIR_METADATA = join(DIR_DATA, "Datasheets")
DIR_METADATA_RAW = join(DIR_METADATA, "raw")
DIR_METADATA_CLEAN = join(DIR_METADATA, "clean")

# Dataset to image sub-directory mapping
DSET_TO_IMG_SUBDIR = {
    "sickkids": join("Images", "all_lab_img"),
    "stanford": join("Images", "all_lab_img"),
    "sickkids_beamform": "SK_Img_noContrast_20220831",
    "stanford_beamform": "stanford_raw",
    "sickkids_image": "Preprocessed",
    "sickkids_silent_trial": "SilentTrial/HN Outputs",
    "stanford_image": "Stanford",
    "uiowa": "UIowa",
    "chop": "CHOP",
    # "ubc_adult": "UBC_AdultKidneyDataset/clean",
    # "ubc_adult_raw": "UBC_AdultKidneyDataset/raw",
    # "sickkids_subset_1": "Images/all_lab_img",
    # "sickkids_subset_2": "Images/all_lab_img",
    # "sickkids_subset_3": "Images/all_lab_img",
    # "sickkids_subset_4": "Images/all_lab_img",
    # "sickkids_subset_5": "Images/all_lab_img",
}

# Complete image sub-directory paths
DSET_TO_IMG_SUBDIR_FULL = {
    dset: join(DIR_DATA, subdir)
    for dset, subdir in DSET_TO_IMG_SUBDIR.items()
}

# Segmentation directories
DIR_SEGMENT = join(DIR_DATA, "Segmented_originals")
DIR_SEGMENT_PROC = join(DIR_DATA, "sk_segmented_images")
DIR_SEGMENT_SRC = join(DIR_SEGMENT_PROC, "src")
DIR_SEGMENT_MASK = join(DIR_SEGMENT_PROC, "mask")

# Directories for saving data
DIR_SAVE = join(DIR_PROJECT, "save_data")
DIR_WEIGHTS = join(DIR_SAVE, "weights")
DIR_EMBEDS = join(DIR_SAVE, "embeddings")
DIR_FIGURES = join(DIR_SAVE, "figures")
DIR_RESULTS = join(DIR_SAVE, "results")
DIR_INFERENCE = join(DIR_SAVE, "inference")
DIR_HN_INFERENCE = join(DIR_SAVE, "hn_inference")

# Figures directories
DIR_FIGURES_EDA = join(DIR_FIGURES, "eda")
DIR_FIGURES_UMAP = join(DIR_FIGURES, "umap")
DIR_FIGURES_CAM = join(DIR_FIGURES, "grad_cam")
DIR_FIGURES_PRED = join(DIR_FIGURES, "predictions")

# Configuration directories
DIR_CONFIG = join(DIR_PROJECT, "config")
DIR_CONFIG_SPECS = join(DIR_CONFIG, "configspecs")

################################################################################
#                           Processed Metadata Files                           #
################################################################################
DSET_TO_METADATA = {
    "raw": {
        # SickKids (Video)
        # Old metadata files below are merged into 1 with "Other labels"
        # "sickkids": "fulltrain-view_label_df_20200423-noOther.csv",
        # "sickkids_test": "test-view_label_df_20200423-noOther.csv",
        "sickkids": "renalUSlabels_withOther_20240824.csv",
        "sickkids_machine": "fulltrain_datasheet_sda_20201022.csv",
        "sickkids_test_machine": "test_datasheet_sda_20201022.csv",
        "sickkids_hn": "sickkids_hn_labels.csv",
        "sickkids_corrections": "sickkids_stanford_corrected_view_labels.xlsx",
        # SickKids Silent Trial (Image)
        "sickkids_silent_trial": "sickkids_silent_trial_metadata.csv",
        "sickkids_silent_trial_hn": "sickkids_silent_trial_hn_metadata.csv",
        # SickKids (Image)
        "sickkids_image": "sickkids_image_labels.csv",
        # Stanford (Video)
        "stanford": "stanford_image_labels.csv",
        "stanford_hn": "stanford_hn_labels.csv",
        # Stanford (Image)
        "stanford_image": "stanford_image_metadata.csv",
        # UIowa
        "uiowa": "uiowa_metadata.csv",
        "uiowa_hn": "uiowa_hn_metadata.csv",
        # CHOP
        "chop": "chop_metadata.csv",
        "chop_hn": "chop_hn_metadata.csv",
        # UBC Open Adult Kidney dataset
        "ubc_adult": "ubc_adult_image_metadata.csv",
    },
    "clean": {
        # Video datasets
        "sickkids": "sickkids_video_metadata.csv",
        "stanford": "stanford_video_metadata.csv",

        # Video (beamform) datasets
        "sickkids_beamform": "sickkids_video_beamform_metadata.csv",
        "stanford_beamform": "stanford_video_beamform_metadata.csv",

        # Image datasets
        "sickkids_image": "sickkids_image_metadata.csv",
        "stanford_image": "stanford_image_metadata.csv",
        "sickkids_silent_trial": "sickkids_silent_trial_metadata.csv",
        "uiowa": "uiowa_image_metadata.csv",
        "chop": "chop_image_metadata.csv",
        "ubc_adult": "ubc_adult_image_metadata.csv",

        # Subsets of SickKids Video for Adult vs. Child experiment
        "sickkids_subset_1": "adult_vs_child/sickkids_subset_1_metadata.csv",
        "sickkids_subset_2": "adult_vs_child/sickkids_subset_2_metadata.csv",
        "sickkids_subset_3": "adult_vs_child/sickkids_subset_3_metadata.csv",
        "sickkids_subset_4": "adult_vs_child/sickkids_subset_4_metadata.csv",
        "sickkids_subset_5": "adult_vs_child/sickkids_subset_5_metadata.csv",
    }
}
# Prepend raw/clean directory
for key in ("raw", "clean"):
    prepend_dir = DIR_METADATA_RAW if key == "raw" else DIR_METADATA_CLEAN
    DSET_TO_METADATA[key] = {
        dset: join(prepend_dir, subdir)
        for dset, subdir in DSET_TO_METADATA[key].items()
    }


################################################################################
#                                Data Related                                  #
################################################################################
# Possible hospitals
HOSPITALS = ("sickkids", "stanford", "uiowa", "chop")

# Video/image dataset names
VIDEO_DSETS = (
    "sickkids", "stanford",
    "sickkids_beamform", "stanford_beamform",
)
IMAGE_DSETS = (
    "sickkids_image", "stanford_image", "sickkids_silent_trial",
    "uiowa", "chop", "ubc_adult"
)

# Hospitals without bladder
DSETS_MISSING_BLADDER = (
    "stanford_image", "sickkids_silent_trial",
    "uiowa", "chop", "ubc_adult",
)

# Expected image size by model
IMG_SIZE = (256, 256)

# Classes
CLASSES = defaultdict(lambda : ("Sagittal_Left", "Transverse_Left", "Bladder",
                                "Transverse_Right", "Sagittal_Right", "Other"))
CLASSES["relative"] = ("Sagittal_First", "Transverse_First", "Bladder",
                       "Transverse_Second", "Sagittal_Second")

# 2. CLASS_TO_IDX
# 2.1 Split by side (left, right, middle)
CLASS_TO_SIDE_IDX = {"Left": 0, "Right": 1, "Bladder": 2, "Other": 3,}
SIDE_IDX_TO_CLASS = {idx: label for label, idx in CLASS_TO_SIDE_IDX.items()}

# 2.2 Split by plane (sagittal, transverse, bladder)
CLASS_TO_PLANE_IDX = {"Sagittal": 0, "Transverse": 1, "Bladder": 2, "Other": 3}
PLANE_IDX_TO_CLASS = {idx: label for label, idx in CLASS_TO_PLANE_IDX.items()}


# Mapping of label part to variables defined above
LABEL_PARTS = ["side", "plane"]
LABEL_PART_TO_CLASSES = {
    "side": {
        "classes": ("Left", "Right", "Bladder", "Other"),
        "class_to_idx": CLASS_TO_SIDE_IDX,
        "idx_to_class": SIDE_IDX_TO_CLASS,
    },
    "plane":{
        "classes": ("Sagittal", "Transverse", "Bladder", "Other"),
        "class_to_idx": CLASS_TO_PLANE_IDX,
        "idx_to_class": PLANE_IDX_TO_CLASS,
    },
    None: {
        "classes": CLASSES[""],
        "class_to_idx": {label: i for i, label in enumerate(CLASSES[""])},
        "idx_to_class": {i: label for i, label in enumerate(CLASSES[""])},
    }
}
# HACK: Create string version for all views (which includes side and plane)
LABEL_PART_TO_CLASSES["side_and_plane"] = LABEL_PART_TO_CLASSES[None]


# Adjacency list of view labels (allows option of specifying for relative)
LABEL_ADJACENCY = defaultdict(lambda : {
        "Sagittal_Left": ["Transverse_Left"],
        "Transverse_Left": ["Sagittal_Left", "Bladder"],
        "Bladder": ["Transverse_Left", "Transverse_Right"],
        "Transverse_Right": ["Bladder", "Sagittal_Right"],
        "Sagittal_Right": ["Transverse_Right"]
    })
LABEL_ADJACENCY["relative"] = {
        "Sagittal_First": ["Transverse_First"],
        "Transverse_First": ["Sagittal_First", "Bladder"],
        "Bladder": ["Transverse_First", "Transverse_Second"],
        "Transverse_Second": ["Bladder", "Sagittal_Second"],
        "Sagittal_Second": ["Transverse_Second"]
    }


################################################################################
#                                Data Splitting                                #
################################################################################
# Dataset to train/val/set IDs
DSET_TO_SPLIT_IDS = {
    "sickkids": {
        "train": [
            "1006", "1007", "1014", "1015", "1016", "1017", "1018", "1023",
            "1024", "1025", "1026", "1027", "1028", "1029", "1030", "1031",
            "1033", "1034", "1036", "1037", "1040", "1042", "1043", "1046",
            "1049", "1051", "1052", "1053", "1054", "1057", "1058", "1060",
            "1061", "1062", "1063", "1064", "1067", "1068", "1071", "1072",
            "1073", "1074", "1079", "1080", "1082", "1083", "1084", "1085",
            "1086", "1090", "1091", "1094", "1095", "1096", "1097", "1101",
            "1102", "1106", "1108", "1109", "1111", "1112", "1116"
        ],
        "val": [
            "1003", "1010", "1011", "1021", "1022", "1035", "1044", "1045",
            "1048", "1056", "1065", "1087", "1088", "1098", "1099", "1103",
            "1114"
        ],
        "test": [
            "1001","1002","1004","1005","1008","1009","1012","1019","1020",
            "1032","1038","1039","1041","1047","1050","1055","1059","1066",
            "1069","1070","1075","1076","1077","1078","1081","1089","1092",
            "1093","1100","1104","1105","1107","1110","1113","1115"
        ]
    },
}

################################################################################
#                                   Testing                                    #
################################################################################
# Default dataset split to perform inference on
DEFAULT_EVAL_SPLIT = "val"
