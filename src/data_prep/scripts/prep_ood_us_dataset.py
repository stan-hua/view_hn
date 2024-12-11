"""
prep_ood_us_dataset.py

Description: Script to prepare image and metadata for OOD-US dataset

Note
----
The following datasets are used to create the "POCUS Out-Of-View Dataset"
1. [POCUS Atlas](https://www.thepocusatlas.com/) (1.7K videos)
    - CC BY-NC 4.0 - needs attribution and cannot be used for commercial purposes
    - NOTE: OOD between adult vs. pediatric data
2. [USEnhance dataset - Thyroid / Carotid Artery / Breast / Liver / Kidney](https://ultrasoundenhance2023.grand-challenge.org/ultrasoundenhance2023/) (_ images, _ patients)
    - Unknown license, made publicly available
    - NOTE: OOD between high vs. low quality ultrasound machines
3. Simulated Ultrasound Noise Dataset
    - NOTE: Simulate by generating artificial speckle noise and applying different cone/rectangle masks in the center
    - NOTE: Can insert background information from images

- TODO: Consider simulating multiple images in one pane
- TODO: Consider using MixUp between outliers
- TODO: Consider impact of removed background details


The following datasets are used to create the "Ultrasound Out-Of-View Dataset"
1. Brain Ultrasound
    1. [ReMIND dataset](https://www.cancerimagingarchive.net/collection/remind/) (342=3x114 3D iUS videos, 114 patients)
        - CC BY 4.0 - needs attribution but can be used commercially
        - NOTE: OOD against unseen age/sex/race/histopathology
        - 3 sequences per patient: a) before dural opening, b) after dural opening, and c) before intraoperative MRI
            - Before Dural Opening (less clear because of dura)
            - After Dural Opening (more clear but brain shift can occur)
            - Before Intraoperative MRI (part of tumor has been resected by this point)
2. Neck Ultrasound (1K images)
    # TODO: 1. [Thyroid TN3K dataset](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) (3493 images, 2421 patients)
        - MIT License
    # TODO: 2.[Carotid Artery](https://data.mendeley.com/datasets/d4xt63mgjm/1) (1100 images, 11 patients)
        - CC BY-NC 4.0 - needs attribution and cannot be used for commercial purposes
3. Breast Ultrasound (1K images)
    # TODO: 1. [Breast-Lesions-USG dataset](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) (498 images, 500 patients)
        - CC BY 4.0 - needs attribution but can be used commercially
        - NOTE: Can be used to see if OOD detection can bias between unseen datasets
    # TODO: 2. [BUS-BRA dataset](https://zenodo.org/records/8231412) (1875 images, 1064 patients)
        - NOTE: Need to cite original paper (DOI: 10.1002/mp.16812)
    # TODO: 3. [BUS-UCLM dataset](https://data.mendeley.com/datasets/7fvgj4jsp7/2) (683 images, 38 patients)
        - CC BY-NC 4.0 - needs attribution and cannot be used for commercial purposes
4. Heart Ultrasound
    # TODO: 1. [CardiacUDA dataset](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset) (992 videos, 100 patients)
        - Apache 2.0 - needs copy of license and changelist when re-distributing
    # TODO: 2. [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html) (1000 videos, 500 patients)
        - TODO: Get data license
        - NOTE: Need to cite original paper (DOI: 10.1109/TMI.2019.2900516)
5. Lung Ultrasound (1K images)
    1. Adult [COVID POCUS](https://github.com/jannisborn/covid19_ultrasound) dataset (200 videos; images)
        - Scraped data, unknown attribution
        - NOTE: Can be used to see if OOD detection is affected by frame rate
6. Abdominal Ultrasound (1K images)
    1. [Pediatric Appendix](https://www.kaggle.com/datasets/joebeachcapital/regensburg-pediatric-appendicitis) (2.1K images; appendix)
        - CC BY-NC 4.0 - needs attribution and cannot be used for commercial purposes
    2. Adult [USNotAI](https://github.com/LeeKeyu/abdominal_ultrasound_classification) (360 images; bladder, kidney, bowel, gallbladder, liver, spleen)
        - Unknown license, made publicly available
    # TODO: 3. [B-mode-and-CEUS-Liver](https://www.cancerimagingarchive.net/collection/b-mode-and-ceus-liver/) (~600 images, 120 patients)
        - NOTE: Get first half of photos without dual ultrasound
    # TODO: 4. [Liver Ultrasound Tracking](https://clust.ethz.ch/data.html) (63 images, 60 patients)
7. Extremities (Arms / Knee / Legs) Ultrasound (1K images)
    1. Adult (35-70 years) [Knee Ultrasound](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SKP9IB) (>15K images; knee)
        - CC0 1.0 Universal: Public Domain
        - NOTE: Can be used to see if OOD detection can has differences between seen/unseen knee views
    # TODO: 2. Adult [Biceps / Lower Leg Ultrasound](https://data.mendeley.com/datasets/3jykz7wz8d/1) (3917 images, 1283 patients)
        - CC BY-NC 4.0 - needs attribution and cannot be used for commercial purposes
8. Pelvic Ultrasound (1K images)
    1. [Adult ovarian ultrasound dataset](https://figshare.com/articles/dataset/_zip/25058690?file=44222642) (1.2K images; pelvis)
        - CC BY 4.0 - needs attribution but can be used commercially
    # TODO: 2. [Prostate Ultrasound](https://zenodo.org/records/10475293) (75 videos, 75 patients)
        - NOTE: Need to cite paper (DOI: 10.1016/j.compmedimag.2024.102326)
9. Fetal Ultrasound (1K images)
    1. [Fetal Planes dataset](https://zenodo.org/records/3904280) (12K images; abdomen, brain, femur and thorax)
        - CC BY 4.0 - needs attribution but can be used commercially
        - NOTE: Can be used to see if OOD detection can has differences between seen/unseen plane views
    # TODO: 2. [PSFHS dataset](https://ps-fh-aop-2023.grand-challenge.org/) (4000 images, 305 patients)
        - NOTE: Can be used to see if OOD detection can has differences between seen/unseen plane views


Excluded Because of License:
1. [Nerve dataset](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data) (8K images)
    - Cannot be used outside of the competition

Excluded Because of Data Size:
1. [Clarius Clinical Gallery](https://clarius.com/about/clinical-gallery/)
2. Adult [Thyroid dataset](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) (480 images)
    - "Open Access" - unknown license

Excluded Because of Data Quality:
1. Adult [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) (1.57K images, 600 patients)
    - CC0 1.0 Universal: Public Domain
    - Letter on BUSI dataset issues (https://www.sciencedirect.com/science/article/pii/S2352340923003669?via%3Dihub#sec0009)
"""

# Standard libraries
import json
import os
import random
import shutil
import urllib.request
import re
import requests
import zipfile
from collections import deque
from glob import glob
from urllib.error import HTTPError

# Non-standard libraries
import cv2
import kagglehub
import numpy as np
import pandas as pd
import shortuuid
from bs4 import BeautifulSoup
from fire import Fire
from PIL import Image
from tqdm import tqdm


################################################################################
#                                  Constants                                   #
################################################################################
# Random seed
SEED = 42

# Set home directory
# NOTE: Assume that ~/.cache/kagglehub/datasets contains downloaded datasets
HOME_DIR = os.environ["HOME"]
KAGGLE_DIR = os.path.join(HOME_DIR, ".cache", "kagglehub")
KAGGLE_DATASETS_DIR = os.path.join(KAGGLE_DIR, "datasets")
KAGGLE_COMPETITIONS_DIR = os.path.join(KAGGLE_DIR, "competitions")

# Set data directories
DATA_DIR = os.environ.get("OOD_DATA_DIR", "ood_data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "clean")

# Valid organs
CLEAN_DSETS = [
    # Partially Unseen OOD
    "neck", "breast", "lung", "abdomen", "knee", "pelvis", "fetal",
    # Completely Unseen OOD
    "pocus_atlas",
]

# Mapping of dataset name to directories
DATASET_TO_DIR = {
    "raw": {
        "brain": os.path.join(RAW_DATA_DIR, "ReMIND"),
        "neck_thyroid_tn3k": os.path.join(RAW_DATA_DIR, "tn3k"),
        # "neck_thyroid_ddti": os.path.join(RAW_DATA_DIR, "ddti-thyroid-ultrasound-images"),
        # "neck_nerve": os.path.join(RAW_DATA_DIR, "ultrasound-nerve-segmentation"),
        "breast": os.path.join(RAW_DATA_DIR, "breast-ultrasound-images-dataset"),
        "heart": os.path.join(RAW_DATA_DIR, "cardiacudc-dataset"),
        "lung": os.path.join(RAW_DATA_DIR, "covid19_lung_ultrasound"),
        "abdominal_appendix": os.path.join(RAW_DATA_DIR, "regensburg-pediatric-appendicitis"),
        "abdominal_organs": os.path.join(RAW_DATA_DIR, "abdominal_ultrasound_classification"),
        "knee": os.path.join(RAW_DATA_DIR, "knee_ultrasound"),
        "pelvis_ovarian": os.path.join(RAW_DATA_DIR, "pelvis_ovarian"),
        "fetal_planes": os.path.join(RAW_DATA_DIR, "fetal_planes"),

        # Held-out test sets
        "pocus_atlas": os.path.join(RAW_DATA_DIR, "pocus_atlas"),
        "clarius": os.path.join(RAW_DATA_DIR, "clarius"),
    },
    "clean": {
        "brain": os.path.join(CLEAN_DATA_DIR, "brain"),
        "neck_thyroid_tn3k": os.path.join(CLEAN_DATA_DIR, "neck"),
        # "neck_thyroid_ddti": os.path.join(CLEAN_DATA_DIR, "neck"),
        # "neck_nerve": os.path.join(CLEAN_DATA_DIR, "neck"),
        "breast": os.path.join(CLEAN_DATA_DIR, "breast"),
        "heart": os.path.join(CLEAN_DATA_DIR, "heart"),
        "lung": os.path.join(CLEAN_DATA_DIR, "lung"),
        "abdominal_appendix": os.path.join(CLEAN_DATA_DIR, "abdomen"),
        "abdominal_organs": os.path.join(CLEAN_DATA_DIR, "abdomen"),
        "knee": os.path.join(CLEAN_DATA_DIR, "knee"),
        "pelvis_ovarian": os.path.join(CLEAN_DATA_DIR, "pelvis"),
        "fetal_planes": os.path.join(CLEAN_DATA_DIR, "fetal"),

        # Held-out test sets
        "pocus_atlas": os.path.join(CLEAN_DATA_DIR, "pocus_atlas"),
        "clarius": os.path.join(CLEAN_DATA_DIR, "clarius"),

        # Background
        "background": os.path.join(CLEAN_DATA_DIR, "background"),
    }
}


################################################################################
#                               Downloading Data                               #
################################################################################
def download_datasets(*organs):
    """
    Downloads specified ultrasound datasets using predefined download functions.

    Parameters
    ----------
    *organs : Any
        List of organ datasets to download. If None, all available datasets are downloaded.
    """
    print(f"[OOD Ultrasound] Starting download for the following organs: {organs}")

    # Mapping of organ to download method
    # TODO: Add brain
    download_map = {
        "neck_thyroid_tn3k": {
            "func": download_neck_thyroid_tn3k_ultrasound,
        },
        # NOTE: DDTI replaced with TN3K
        # "neck_thyroid_ddti": {
        #     "func": download_kaggle,
        #     "kwargs": {
        #         "kaggle_path": "dasmehdixtr/ddti-thyroid-ultrasound-images",
        #         "data_type": "dataset",
        #         "save_dir": RAW_DATA_DIR,
        #     }
        # },
        # NOTE: No longer used due to license
        # "neck_nerve": {
        #     "func": download_kaggle,
        #     "kwargs": {
        #         "kaggle_path": "ultrasound-nerve-segmentation",
        #         "data_type": "competition",
        #         "save_dir": RAW_DATA_DIR,
        #     }
        # },
        "breast": {
            "func": download_kaggle,
            "kwargs": {
                "kaggle_path": "aryashah2k/breast-ultrasound-images-dataset",
                "data_type": "dataset",
                "save_dir": RAW_DATA_DIR,
            }
        },
        # TODO: Fix  this
        "heart": {
            "func": None,
        },
        "lung": {
            "func": download_covid19_ultrasound,
        },
        "abdominal_appendix": {
            "func": download_kaggle,
            "kwargs": {
                "kaggle_path": "joebeachcapital/regensburg-pediatric-appendicitis",
                "data_type": "dataset",
                "save_dir": RAW_DATA_DIR,
            }
        },
        "abdominal_organs": {
            "func": download_github,
            "kwargs": {
                "repo_owner": "LeeKeyu",
                "repo_name": "abdominal_ultrasound_classification",
                "dir_paths": ["dataset/img/train", "dataset/img/test"],
                "save_dir": RAW_DATA_DIR,
            }
        },
        "pelvis_ovarian": {
            "func": download_from_url,
            "kwargs": {
                "url": "https://figshare.com/ndownloader/files/44222642",
                "save_name": "dataset.zip",
                "save_dir": DATASET_TO_DIR["raw"]["pelvis_ovarian"],
                "unzip": True,
            }
        },
        "fetal_planes": {
            "func": download_from_url,
            "kwargs": {
                "url": "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1",
                "save_name": "dataset.zip",
                "save_dir": DATASET_TO_DIR["raw"]["fetal_planes"],
                "unzip": True,
            }
        },

        # Held-out test sets
        "pocus_atlas": {
            "func": download_pocus_atlas,
        },
        # "clarius": {
        #     "func": download_clarius,
        # },
    }

    # If no organs specified, then download all
    if not organs:
        organs = download_map.keys()
        manual_organs = list(set(DATASET_TO_DIR["raw"].keys()).difference(set(organs)))
        if manual_organs:
            print(
                "WARNING: Some datasets require manual downloading! See README.md..."
                f"\nOrgan List: `{manual_organs}`"
            )

    # Download each dataset
    for organ in organs:
        print(f"Downloading dataset: {organ}")
        download_func = download_map[organ]["func"]
        download_func(
            *download_map[organ].get("args", []),
            **download_map[organ].get("kwargs", {}),
        )


def download_brain_ultrasound(save_dir=None):
    """
    Manually download 3D adult brain ultrasound data from TCIA

    Parameter
    ---------
    save_dir : str
        Path to directory to save images
    """
    save_dir = save_dir if save_dir else DATASET_TO_DIR["raw"]["brain"]

    raise NotImplementedError(
        "TCIA does NOT support automated downloads for the ReMIND (brain ultrasound) dataset!\n"
        "Please download the dataset at: https://www.cancerimagingarchive.net/collection/remind/\n"
        f"Using the NBIA, download only the ultrasound DICOM files and metadata to `{save_dir}`"
    )


def download_neck_thyroid_tn3k_ultrasound(save_dir=None):
    """
    Manually download neck thyroid ultrasound data from Google Drive

    Parameter
    ---------
    save_dir : str
        Path to directory to save images
    """
    save_dir = save_dir if save_dir else DATASET_TO_DIR["raw"]["neck_thyroid_tn3k"]

    raise NotImplementedError(
        "Neck Thyroid TN3K dataset does NOT support automated downloads!\n"
        "Please download the dataset at: https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view?usp=sharing\n"
        f"And extract `Thyroid Dataset/tn3k` folder to `{save_dir}`\n\n"
        "For more details, refer to the GitHub page: https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation/"
    )


def download_covid19_ultrasound(save_dir=None):
    """
    Download data from the `jannisborn/covid19_ultrasound` GitHub repository
    """
    save_dir = save_dir if save_dir else DATASET_TO_DIR["raw"]["lung"]
    video_dir = os.path.join(save_dir, "videos")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    print("[Lung] Beginning download of Lung Video Ultrasound data...")

    # GitHub base URLs
    base_url = "https://raw.githubusercontent.com/jannisborn/covid19_ultrasound/master/data/"
    metadata_url = f"{base_url}/dataset_metadata.csv"

    # Download metadata file
    orig_metadata_path = os.path.join(save_dir, "metadata.csv")
    urllib.request.urlretrieve(metadata_url, orig_metadata_path)
    print("[Lung] Downloaded metadata")

    # Load metadata file to determine which images to download
    df_metadata = pd.read_csv(orig_metadata_path, encoding="cp1258")
    print(f"[Lung] Number of videos in metadata file: {len(df_metadata)}")

    # Drop the unused videos, and videos with artifacts
    df_metadata = df_metadata[df_metadata["Current location"].str.startswith("pocus_videos")]
    print(f"[Lung] Filtering out unused videos and videos with artifacts...")
    print(f"[Lung] Number of videos in metadata file: {len(df_metadata)}")

    # Drop all videos with space in filename (likely don't exist)
    df_metadata = df_metadata[~df_metadata["Filename"].str.contains(" ")]
    print(f"[Lung] Filtering out more unused videos")
    print(f"[Lung] Number of videos in metadata file: {len(df_metadata)}")

    # Randomly shuffle
    df_metadata = df_metadata.sample(frac=1, random_state=SEED)

    # Create base GitHub path (missing file extension)
    df_metadata["base_github_path"] = df_metadata.apply(
        lambda row: os.path.join(base_url, row["Current location"], row["Filename"]),
        axis=1,
    )

    # Attempt to download video files on GitHub using all possible extensions
    extensions = ["mov", "gif", "mp4", "avi", "mpeg", "mpg"]
    print("[Lung] Beginning video downloads...")
    df_metadata["local_save_path"] = df_metadata["base_github_path"].map(
        lambda path: try_video_download_all_exts(path, extensions, video_dir)
    )
    # Drop all videos that failed to download
    df_metadata = df_metadata[~df_metadata["local_save_path"].isna()]
    print(f"[Lung] Final number of downloaded videos from metadata file: {len(df_metadata)}")

    # Remove data directory from path
    df_metadata["local_save_path"] = df_metadata["local_save_path"].map(remove_home_dir)

    # Reorganize columns
    cols = ["local_save_path"]
    cols += [col for col in df_metadata.columns if col not in cols]
    df_metadata = df_metadata[cols]

    # Update raw metadata with the saved videos
    df_metadata.to_csv(orig_metadata_path, index=False)


def download_knee_ultrasound(save_dir=None):
    """
    Manually download adult knee ultrasound data from Harvard dataverse

    Parameter
    ---------
    save_dir : str
        Path to directory to save images
    """
    save_dir = save_dir if save_dir else DATASET_TO_DIR["raw"]["knee"]

    raise NotImplementedError(
        "Harvard Dataverse does NOT support automated downloads for the JoCo Knee Ultrasound dataset!\n"
        "Please download the dataset at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SKP9IB#\n"
        f"And extract dataverse_files/data folder to `{save_dir}`"
    )


def download_pocus_atlas(save_dir=None):
    """
    Download GIFs from the POCUS Atlas (https://www.thepocusatlas.com/)

    The website is scraped and GIFs are downloaded from all sections, with their
    associated text stored in a metadata CSV file.

    Parameters
    ----------
    save_dir : str, optional
        Path to directory to save images, by default DATASET_TO_DIR["raw"]["pocus_atlas"]
    """
    # Create save directory for POCUS atlas
    save_dir = save_dir if save_dir else DATASET_TO_DIR["raw"]["pocus_atlas"]

    # List of website paths to visit
    base_url = "https://www.thepocusatlas.com/"
    focus_to_sub_url = {
        # Adult Data
        "aorta": "aorta",
        "biliary": "hepatobiliary",
        "echocardiography": "echocardiography",
        "gastrointestinal": "gastrointestinal",
        "musculoskeletal": "softtissue-msk",
        "ob_gyn": "obgyn",
        "ocular": "ocular",
        "lung": "lung",
        "renal": "renal",
        "soft_tissue": "softtissuemsk",
        "trauma": "trauma",
        "vascular": "dvt",

        # Pediatric Data
        "pediatric": "pediatrics-1",
        "peds_biliary": "pedsbiliary",
        "peds_gastrointestinal": "pedsgastrointestinal",
        "peds_musculoskeletal": "pedsmsk",
        "peds_lung": "pedslung",
    }

    # For each area, visit the website and download the GIFs
    # NOTE: Store case text associated with each GIF for later reference
    # NOTE: Also create a new ID for each image
    accum_data = {
        "local_save_path": [],
        "focus": [],
        "description": [],
        "video_id": [],
        "src_url": [],
    }
    print("[POCUS Atlas] Beginning GIF downloads...")
    for focus, sub_path in focus_to_sub_url.items():
        print(f"[POCUS Atlas] Downloading GIFs for {focus}...")
        # Create a new subdirectory for each focus
        focus_dir = os.path.join(save_dir, focus)
        os.makedirs(focus_dir, exist_ok=True)

        # Create new URL
        curr_url = f"{base_url}{sub_path}"

        # Load website
        response = requests.get(curr_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get all GIFs on website
        gifs = soup.find_all('img', {'data-src': lambda x: x and x.endswith('.gif')})

        # Download each GIF and store
        for gif_idx, gif in tqdm(enumerate(gifs)):
            # Download the GIF
            curr_save_path = os.path.join(focus_dir, f"{focus}-{gif_idx}.gif")
            gif_response = requests.get(gif["data-src"])
            with open(curr_save_path, 'wb') as f:
                f.write(gif_response.content)

            # Store metadata associated with GIF
            accum_data["local_save_path"].append(curr_save_path)
            accum_data["focus"].append(focus)
            accum_data["description"].append(gif["alt"])
            accum_data["video_id"].append(f"{focus}-{gif_idx}")
            accum_data["src_url"].append(gif["data-src"])
        print(f"[POCUS Atlas] Downloading GIFs for {focus}...DONE")
    print(f"[POCUS Atlas] Final number of downloaded GIFs: {len(accum_data['local_save_path'])}")

    # Save metadata
    orig_metadata_path = os.path.join(save_dir, "metadata.csv")
    df_metadata = pd.DataFrame(accum_data)

    # Remove data directory from path
    df_metadata["local_save_path"] = df_metadata["local_save_path"].map(remove_home_dir)
    df_metadata.to_csv(orig_metadata_path, index=False)


# NOTE: The following function is provided not used
def download_clarius(save_dir=None):
    """
    Download GIFs from Clarius Clinical Gallery (https://clarius.com/about/clinical-gallery/)

    The website is scraped and GIFs are downloaded from all sections, with their
    associated text stored in a metadata CSV file.

    Parameters
    ----------
    save_dir : str, optional
        Path to directory to save images, by default DATASET_TO_DIR["raw"]["clarius"]
    """
    # Create save directory for Clarius
    save_dir = save_dir if save_dir else DATASET_TO_DIR["raw"]["clarius"]

    # List of website paths to visit
    base_url = "https://clarius.com/about/clinical-gallery/?filter_ultrasound_category="
    focus_to_filters = {
        "abdomen": "abdomen",
        "bladder": "bladder",
        "breast": "breast",
        "heart": "cardiac,carotid",
        "hand_wrist": "hand-wrist",
        "hip": "hip",
        "knee": "knee",
        "lung": "lung",
        "musculoskeletal": "msk",
        "nerve": "nerve",
        "ob_gyn": "ob",
        "ocular": "ocular",
        "prostate": "prostate",
        "soft_tissue": "superficial",
        "vascular": "vascular,vascular-access",

        # Animals
        "animal": "small-animal,vet",
    }

    # For each area, visit the website and download the GIFs
    # NOTE: Store case text associated with each GIF for later reference
    # NOTE: Also create a new ID for each image
    accum_data = {
        "local_save_path": [],
        "focus": [],
        "description": [],
        "video_id": [],
    }
    print("[Clarius] Beginning image downloads...")
    for focus, filter_str in focus_to_filters.items():
        print(f"[Clarius] Downloading images for {focus}...")
        # Create a new subdirectory for each focus
        focus_dir = os.path.join(save_dir, focus)
        os.makedirs(focus_dir, exist_ok=True)

        # Create new URL
        curr_url = f"{base_url}{filter_str}"

        # Load website
        response = requests.get(curr_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get all GIFs on website
        exts = [".gif", ".jpg", ".jpeg", ".png"]
        gifs = soup.find_all('img', {'data-src': lambda x: x and any(x.endswith(ext) for ext in exts)})

        # Download each GIF and store
        img_idx = 0
        for img_html in tqdm(gifs):
            # Skip, if image doesn't have a description
            if not img_html["alt"] or not isinstance(img_html["alt"], str):
                continue

            # Download the GIF
            ext = os.path.basename(img_html["data-src"]).split(".")[-1]
            curr_save_path = os.path.join(focus_dir, f"{focus}-{img_idx}.{ext}")
            gif_response = requests.get(img_html["data-src"])
            with open(curr_save_path, 'wb') as f:
                f.write(gif_response.content)

            # Store metadata associated with GIF
            accum_data["local_save_path"].append(curr_save_path)
            accum_data["focus"].append(focus)
            accum_data["description"].append(img_html["alt"])
            accum_data["video_id"].append(f"{focus}-{img_idx}")

            # Increment image index
            img_idx += 1
        print(f"[Clarius] Downloading images for {focus}...DONE")
    print(f"[Clarius] Final number of downloaded images: {len(accum_data['local_save_path'])}")

    # Save metadata
    orig_metadata_path = os.path.join(save_dir, "metadata.csv")
    df_metadata = pd.DataFrame(accum_data)

    # Remove data directory from path
    df_metadata["local_save_path"] = df_metadata["local_save_path"].map(remove_home_dir)
    df_metadata.to_csv(orig_metadata_path, index=False)


################################################################################
#                    Processing Functions (Image Datasets)                     #
################################################################################
def process_datasets(*organs, overwrite=False, aggregate=False):
    """
    Processes downloaded ultrasound datasets using predefined processing functions.

    Parameters
    ----------
    *organs : Any
        List of organ datasets to download. If None, all available datasets are downloaded.
    aggregate : bool, optional
        If True, aggregate all clean datasets into a single file and split data
        into training/calib/test sets.
    """
    # Mapping of organ to processing method
    process_map = {
        "neck": process_neck_datasets,
        "breast": process_breast_dataset,
        "lung": process_lung_dataset,
        "abdomen": process_abdomen_dataset,
        "pelvis": process_pelvis_ovarian_dataset,
        "knee": process_knee_dataset,
        "fetal": process_fetal_planes_dataset,

        # Held-out test sets
        "pocus_atlas": process_pocus_atlas_dataset,
    }

    # If no organs specified, then download all
    if not organs:
        organs = list(process_map.keys())

    # Process each dataset
    print(f"[OOD Ultrasound] Processing data for the following organs: {list(organs)}")
    for organ in organs:
        print(f"Processing dataset: {organ}")
        process_map[organ](overwrite=overwrite)

    # If specified, aggregate all clean datasets
    if aggregate:
        aggregate_processed_datasets()


def process_brain_dataset(data_dir=None, save_dir=None, seed=SEED, overwrite=False):
    """
    Process the Ultrasound Brain dataset from TCIA.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If False, skip processing if the dataset already exists, by default False
    """
    dataset_name = "Brain - ReMIND"
    dataset_key = "brain"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    video_subdir = os.path.join(save_dir, "video")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(video_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # If overwriting, delete existing images
    if overwrite:
        cleanup_img_dir(video_subdir)

    # Load metadata
    df_metadata = pd.read_excel(
        os.path.join(data_dir, "ReMIND-Dataset-Clinical-Data-September-2023.xlsx"),
        sheet_name=0,
    )

    # Rename columns
    df_metadata = df_metadata.rename(columns={
        'Case Number': "patient_id",
        'Age': "age",
        'Sex': "sex",
        'Race': "race",
        'Laterality': "side",
        'WHO Grade': "who_grade",
        'Histopathology': "histopathology",
    })

    # Prepare metadata
    # Set aside patients for test set
    accum_test = []
    # 1. Patients with rare histopathology diagnosis
    common_histo = ["Astrocytoma", "Glioblastoma", "Oligodendroglioma"]
    common_histo_mask = df_metadata["histopathology"].isin(common_histo)
    accum_test.append(df_metadata[~common_histo_mask])
    df_metadata = df_metadata[common_histo_mask]
    # 2. The few Asian patients
    asian_mask = df_metadata["race"] == "Asian"
    accum_test.append(df_metadata[asian_mask])
    df_metadata = df_metadata[~asian_mask]

    # Sample 20 patients for training / calibration set, rest is for testing
    # NOTE: Stratifying by 5 age bins and by sex
    df_metadata["age_bin"] = pd.cut(df_metadata["age"], bins=5)
    df_train_calib = df_metadata.groupby(["age_bin", "sex"]).sample(n=2, random_state=seed)
    accum_test.append(df_metadata[~df_metadata.index.isin(df_train_calib.index)])

    # Create test set from accumulated patients
    df_test = pd.concat(accum_test, ignore_index=True)

    # Add splits then recombine
    df_train_calib["split"] = None
    df_test["split"] = "test"
    df_metadata = pd.concat([df_train_calib, df_test], ignore_index=True)
    # Add dataset
    df_metadata["dset"] = f"{dataset_key}-ReMIND"
    # Add organ
    df_metadata["view"] = "brain"

    # Get DICOM paths
    dicom_paths = glob(os.path.join(data_dir, "ReMIND-*", "*", "*", "*.*"))

    # Extract patient ID and stage
    df_dicoms = pd.DataFrame({"old_video_path": dicom_paths})
    local_path = df_dicoms["old_video_path"].map(lambda x: x.split(data_dir + os.path.sep)[-1])
    # NOTE: Example path: 'ReMIND-001/12-25-1982-NA-Intraop-90478/1.000000-USpredura-64615/1-1.dcm'
    df_dicoms["patient_id"] = local_path.map(lambda x: x.split(os.path.sep)[0].split("-")[1]).astype(int)
    # Map stage to integer (pre-dural=1, post-dural=2, pre-iMRI=3)
    map_stage = {"USpredura": 1, "USpostdura": 2, "USpreimri": 3}
    df_dicoms["stage"] = local_path.map(lambda x: map_stage[x.split(os.path.sep)[-2].split("-")[1]])
    # Create new identifier for each video using patient ID and stage when US was taken
    df_dicoms["video_id"] = dataset_key + "-" + df_dicoms["patient_id"].astype(str) + "-" + df_dicoms["stage"].astype(str)

    # Join DICOM files to metadata
    df_metadata_dicom = merge_tables(df_dicoms, df_metadata, on="patient_id")

    # Convert each DICOM video into image frames.
    # NOTE: Sample 15 frames per video
    accum_old_metadata = {"num_frames": []}
    accum_new_metadata = {"id": [], "video_id": [], "path": []}
    for idx in tqdm(range(len(df_metadata_dicom))):
        row = df_metadata_dicom.iloc[idx]
        local_path = row["old_video_path"]
        video_idx = row["video_id"]
        # Create subdirectory for this video's frames
        curr_video_subdir = os.path.join(video_subdir, video_idx)
        os.makedirs(curr_video_subdir, exist_ok=True)
        # Get at most 15 DICOM video frames
        frames_paths = convert_dicom_to_frames(
            local_path, curr_video_subdir, f"{video_idx}-",
            uniform_num_samples=15,
            overwrite=overwrite)
        num_frames = len(frames_paths)
        # Create indices for each frame/image
        frame_indices = [f"{video_idx}-{i+1}" for i in range(num_frames)]
        accum_new_metadata["id"].extend(frame_indices)
        accum_new_metadata["video_id"].extend([video_idx] * num_frames)
        # Store new path
        accum_new_metadata["path"].extend([remove_home_dir(x) for x in frames_paths])
        # Store number of frames
        accum_old_metadata["num_frames"].append(num_frames)

    # Remove home directory from old paths
    df_metadata_dicom["old_video_path"] = df_metadata_dicom["old_video_path"].map(lambda x: remove_home_dir(x))
    # Store number of frames for each video
    df_metadata_dicom["num_frames"] = accum_old_metadata["num_frames"]

    # Save dataframe of old metadata to new path
    df_old_new = df_metadata_dicom[["old_video_path", "video_id", "num_frames"]]
    df_old_new.to_csv(os.path.join(metadata_subdir, f"{dataset_key}-old_file_mapping.csv"), index=False)

    # Join new metadata back to metadata table and keep specific columns
    df_new_metadata = pd.DataFrame(accum_new_metadata)
    df_new_metadata = merge_tables(df_new_metadata, df_metadata_dicom, on="video_id")
    cols = [
        "dset", "split", "id", "video_id", "patient_id", "path", "view", "stage",
        "age", "sex", "race", "side", "histopathology",
    ]
    df_new_metadata = df_new_metadata[cols]

    # Save new metadata dataframe
    df_new_metadata.to_csv(os.path.join(metadata_subdir, f"{dataset_key}-metadata.csv"), index=False)

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source":  "The Cancer Imaging Archive (TCIA)",
            "URL": "https://www.cancerimagingarchive.net/collection/remind/",
            "Seed": seed,
            "(Before) Number of Videos": len(df_metadata),
            "(After) Number of Images": len(df_new_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


def process_neck_datasets(overwrite=False):
    """
    Process the Ultrasound Neck datasets: thyroid and nerve datasets.

    Parameters
    ----------
    overwrite : bool, optional
        If True, overwrite existing processed data
    """
    clean_neck_dir = DATASET_TO_DIR["clean"]["neck_thyroid_tn3k"]
    metadata_dir = os.path.join(clean_neck_dir, "metadata")

    # Remove provenance before redoing
    provenance_path = os.path.join(metadata_dir, "provenance.txt")
    if os.path.exists(provenance_path):
        os.remove(provenance_path)

    # Process thyroid dataset
    process_neck_thyroid_tn3k_dataset(overwrite=overwrite)
    # NOTE: No longer included
    # process_neck_thyroid_ddti_dataset(overwrite=overwrite)
    # process_neck_nerve_dataset(overwrite=overwrite)

    # Check that metadata directory exists
    assert os.path.exists(metadata_dir), "Metadata directory does not exist!"

    # Check that metadata exists for each organ
    organs = [key for key in DATASET_TO_DIR["clean"] if key.startswith("neck")]

    # Check that metadata exists for each sub-dataset
    accum_metadata = []
    for organ in organs:
        metadata_path = os.path.join(metadata_dir, f"{organ}-metadata.csv")
        assert os.path.exists(metadata_path), f"Metadata for {organ} does not exist!"
        accum_metadata.append(pd.read_csv(metadata_path))

    # Concatenate metadata
    df_metadata = pd.concat(accum_metadata)

    # Save metadata
    metadata_path = os.path.join(metadata_dir, "neck-metadata.csv")
    df_metadata.to_csv(metadata_path, index=False)
    print("[Neck] Saved combined metadata")

    # Remove of all images that are not in the metadata
    print("[Neck] Removing images that are not in the metadata...")
    img_dir = os.path.join(clean_neck_dir, "images")
    valid_paths = set(df_metadata["path"].map(os.path.basename).tolist())
    for img_path in tqdm(os.listdir(img_dir)):
        if os.path.basename(img_path) not in valid_paths:
            os.remove(os.path.join(img_dir, img_path))
    print("[Neck] Finished processing both Thyroid & Nerve datasets")


def process_neck_thyroid_tn3k_dataset(data_dir=None, save_dir=None, seed=SEED, overwrite=False):
    """
    Process the TN3K: Neck Thyroid Ultrasound dataset from GitHub/Google Drive.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If False, skip processing if the dataset already exists, by default False
    """
    dataset_name = "Neck - Thyroid (TN3K)"
    dataset_key = "neck_thyroid_tn3k"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # If overwriting, delete existing images
    if overwrite:
        cleanup_img_dir(img_subdir)

    # Get all images
    train_img_paths = glob(os.path.join(data_dir, "trainval-image", "*.jpg"))
    test_img_paths = glob(os.path.join(data_dir, "test-image", "*.jpg"))

    # From training images, sample 600 images
    sampled_train_img_paths = random.sample(train_img_paths, 600)
    # Final dataset contains 600 from training set and remaining from test set
    sampled_img_paths = sampled_train_img_paths + test_img_paths

    # Create splits, only for test split
    # NOTE: The remaining data will be split between train and calibration
    splits = [None] * len(sampled_train_img_paths)
    splits.extend(["test"] * len(test_img_paths))

    # Add metadata for view (organ) and split
    extra_metadata_cols = {
        "view": ["thyroid"] * len(sampled_img_paths),
        "split": splits,
    }

    # Save sampled images as PNG
    df_metadata = re_anonymize_and_save_as_png(
        sampled_img_paths, dataset_key, img_subdir, metadata_subdir,
        overwrite=overwrite,
        **extra_metadata_cols,
    )

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source": "GitHub / Google Drive",
            "URL": "https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation",
            "Seed": seed,
            "(Before) Number of Images": len(train_img_paths + test_img_paths),
            "(After) Number of Images": len(df_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


def process_breast_dataset(data_dir=None, save_dir=None, seed=SEED,
                           overwrite=False, n=1000):
    """
    Process the Breast Ultrasound dataset from Kaggle.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If True, overwrite existing files, by default False.
    n : int, optional
        Number of images to sample, by default 1000.
    """
    dataset_name = "Breast"
    dataset_key = "breast"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # Remove provenance before redoing
    provenance_path = os.path.join(metadata_subdir, "provenance.txt")
    if os.path.exists(provenance_path):
        os.remove(provenance_path)

    # If overwriting, delete existing images
    if overwrite:
        cleanup_img_dir(img_subdir)

    # Get all the image files
    print(f"[{dataset_name}] Dataset creation started...")
    orig_img_subdir = os.path.join(data_dir, "versions", "1", "Dataset_BUSI_with_GT")
    orig_img_paths = glob(os.path.join(orig_img_subdir, "benign", "*.png"))
    orig_img_paths = orig_img_paths + glob(os.path.join(orig_img_subdir, "malignant", "*.png"))
    orig_img_paths = orig_img_paths + glob(os.path.join(orig_img_subdir, "normal", "*.png"))
    orig_img_paths = filter_paths_for_masks(orig_img_paths)

    # Randomly sample N images, if only above the amount specified
    sampled_img_paths = orig_img_paths
    if len(orig_img_paths) > n:
        sampled_img_paths = random.sample(orig_img_paths, n)
        print(f"[{dataset_name}] Sampling {n}/{len(orig_img_paths)} images...")

    # Add metadata for view (organ)
    extra_metadata_cols = {"view": ["breast"] * len(sampled_img_paths)}

    # Save sampled images as PNG
    df_metadata = re_anonymize_and_save_as_png(
        sampled_img_paths, dataset_key, img_subdir, metadata_subdir,
        overwrite=overwrite,
        **extra_metadata_cols,
    )

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source":  "Kaggle",
            "URL": "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset",
            "Seed": seed,
            "(Before) Number of Images": len(orig_img_paths),
            "(After) Number of Images": len(df_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


def process_abdomen_dataset(overwrite=False):
    """
    Process Abdominal ultrasound datasets:
        1. Pediatric Appendix (~700 images)
        2. Other Organs

    Parameters
    ----------
    overwrite : bool, optional
        If False, skip processing if the dataset already exists, by default False
    """
    clean_abdomen_dir = DATASET_TO_DIR["clean"]["abdominal_appendix"]
    metadata_dir = os.path.join(clean_abdomen_dir, "metadata")

    # Remove provenance before redoing
    provenance_path = os.path.join(metadata_dir, "provenance.txt")
    if os.path.exists(provenance_path):
        os.remove(provenance_path)

    # Process both appendix and multi-organ abdomen datasets
    process_abdominal_appendix_dataset(overwrite=overwrite)
    process_abdominal_organs_dataset(overwrite=overwrite)

    # Check that metadata directory exists
    assert os.path.exists(metadata_dir), "Metadata directory does not exist!"

    # Check that metadata exists for each organ
    organs = [key for key in DATASET_TO_DIR["clean"] if key.startswith("abdominal_")]

    # Check that metadata exists for each sub-dataset
    accum_metadata = []
    for organ in organs:
        metadata_path = os.path.join(metadata_dir, f"{organ}-metadata.csv")
        assert os.path.exists(metadata_path), f"Metadata for {organ} does not exist!"
        accum_metadata.append(pd.read_csv(metadata_path))

    # Concatenate metadata
    df_metadata = pd.concat(accum_metadata)

    # Save metadata
    metadata_path = os.path.join(metadata_dir, "abdomen-metadata.csv")
    df_metadata.to_csv(metadata_path, index=False)
    print("[Abdomen] Saved combined metadata")

    # Remove of all images that are not in the metadata
    print("[Abdomen] Removing images that are not in the metadata...")
    img_dir = os.path.join(clean_abdomen_dir, "images")
    valid_paths = set(df_metadata["path"].map(os.path.basename).tolist())
    for img_path in tqdm(os.listdir(img_dir)):
        if os.path.basename(img_path) not in valid_paths:
            os.remove(os.path.join(img_dir, img_path))
    print("[Abdomen] Finished processing both Abdomen datasets")


def process_abdominal_appendix_dataset(data_dir=None, save_dir=None, seed=SEED, overwrite=False):
    """
    Process the Abdominal Appendix Ultrasound dataset from Kaggle.

    Note
    ----
    The dataset is downloaded using `kagglehub`. The dataset is processed by
    randomly selecting 1 image per case ID. The images are stored in the
    `images` subdirectory and the metadata is stored in the `metadata`
    subdirectory.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If False, skip processing if the dataset already exists, by default False
    """
    dataset_name = "Abdomen - Appendix"
    dataset_key = "abdominal_appendix"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # Get paths to all images
    orig_img_dir = os.path.join(data_dir, "versions", "1", "US_Pictures", "US_Pictures")
    orig_img_paths = glob(os.path.join(orig_img_dir, "*.png"))
    orig_img_paths += glob(os.path.join(orig_img_dir, "*.bmp"))
    orig_img_paths += glob(os.path.join(orig_img_dir, "*.jpg"))

    # Extract ID for each patient
    # NOTE: This corresponds to the column "US_Number" in `app_data.xlsx`
    patient_id = [os.path.basename(p).split(" ")[0].split(".")[0] for p in orig_img_paths]

    # Create table with IDs and paths
    df_old = pd.DataFrame({"old_id": patient_id, "old_path": orig_img_paths})

    # Sample 1 image for every patient
    sampled_img_paths = df_old.groupby("old_id")["old_path"].sample(n=1, random_state=seed).tolist()
    print(f"[{dataset_name}] Sampling {len(sampled_img_paths)}/{len(df_old)} images...")

    # Add metadata for view (organ)
    extra_metadata_cols = {"view": ["peds_appendix"] * len(sampled_img_paths)}

    # Save sampled images as PNG
    re_anonymize_and_save_as_png(
        sampled_img_paths, dataset_key, img_subdir, metadata_subdir,
        overwrite=overwrite,
        **extra_metadata_cols,
    )

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source": "Kaggle",
            "URL": "https://www.kaggle.com/datasets/joebeachcapital/regensburg-pediatric-appendicitis",
            "Seed": seed,
            "(Before) Number of Images": len(orig_img_paths),
            "(After) Number of Images": len(sampled_img_paths),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


def process_abdominal_organs_dataset(data_dir=None, save_dir=None, seed=SEED, overwrite=False):
    """
    Process the Abdominal Ultrasound classification dataset from GitHub.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If False, skip processing if the dataset already exists, by default False
    """
    dataset_name = "Abdomen - Organs"
    dataset_key = "abdominal_organs"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # Get paths to all images
    base_img_dir = os.path.join(data_dir, "dataset", "img")
    orig_img_paths = glob(os.path.join(base_img_dir, "train", "*.png"))
    orig_img_paths += glob(os.path.join(base_img_dir, "test", "*.png"))

    # Extract view (organ) in image
    extra_metadata_cols = {
        "view": [
            os.path.basename(img_path).split("-")[0].lower()
            for img_path in orig_img_paths
        ]
    }

    # Save sampled images as PNG
    # NOTE: Infix function saves annotated organ specified in image filename
    df_metadata = re_anonymize_and_save_as_png(
        orig_img_paths, dataset_key, img_subdir, metadata_subdir,
        infix_name_func=lambda x: os.path.basename(x).split("-")[0],
        overwrite=overwrite,
        **extra_metadata_cols,
    )

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source": "GitHub",
            "URL": "https://github.com/LeeKeyu/abdominal_ultrasound_classification",
            "Seed": seed,
            "(Before) Number of Images": len(orig_img_paths),
            "(After) Number of Images": len(df_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


def process_knee_dataset(data_dir=None, save_dir=None, seed=SEED, overwrite=False):
    """
    Process the Knee Ultrasound classification dataset from Harvard Dataverse.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If False, skip processing if the dataset already exists, by default False
    """
    dataset_name = "Knee"
    dataset_key = "knee"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # Remove provenance before redoing
    provenance_path = os.path.join(metadata_subdir, "provenance.txt")
    if os.path.exists(provenance_path):
        os.remove(provenance_path)

    # If overwriting, delete existing images
    if overwrite:
        cleanup_img_dir(img_subdir)

    # Load raw metadata and its category encodings
    raw_metadata_subdir = os.path.join(data_dir, "reference")
    df_metadata_raw = pd.read_csv(os.path.join(raw_metadata_subdir, "dataTable.SUBJECT.csv"))
    with open(os.path.join(raw_metadata_subdir, "dvDatasetMetadata.json"), "r") as handler:
        metadata_descriptors = json.load(handler)
        patient_metadata_encoding = metadata_descriptors["files"][0]["variables"]
        knee_metadata_encoding = metadata_descriptors["files"][1]["variables"]

    # Decode all categorical columns
    cols_to_decode = {
        "E03GENDER": "gender",
        "E03AGE": "age"
    }
    for col_metadata in patient_metadata_encoding:
        cat_col = col_metadata["name"]
        if cat_col not in cols_to_decode:
            continue
        # Getting mapping
        decode_map = col_metadata["value"]["category"]
        new_col = cols_to_decode[cat_col]
        df_metadata_raw[new_col] = df_metadata_raw[cat_col].map(lambda x: decode_map[str(x)])

    # Rename id column
    df_metadata_raw["patient_id"] = df_metadata_raw["E03SUBJECTID"]
    df_metadata_raw = df_metadata_raw[["patient_id", "age", "gender"]]

    # Get all existing images
    df_metadata_imgs = pd.DataFrame({
        "img_path": glob(os.path.join(data_dir, "image", "ultrasound", "*.png")),
    })
    df_metadata_imgs["filename"] = df_metadata_imgs["img_path"].map(os.path.basename)
    df_metadata_imgs["patient_id"] = df_metadata_imgs["filename"].map(
        lambda x: x.replace(".png", "").split("_")[0]
    )
    df_metadata_imgs["sub_view"] = df_metadata_imgs["filename"].map(
        lambda x: x.replace(".png", "").split("_")[1]
    )
    # Join tables
    df_metadata_raw = df_metadata_imgs.merge(df_metadata_raw, on="patient_id", how="left")

    # Sample 1 image from each patient, uniformly across the 14 views
    df_metadata_raw = df_metadata_raw.groupby("patient_id").sample(n=1, random_state=seed)

    # Add view and sub-view columns
    view_mapping = knee_metadata_encoding[1]["value"]["category"]
    df_metadata_raw["view"] = "knee"
    df_metadata_raw["sub_view"] = df_metadata_raw["sub_view"].map(
        lambda x: view_mapping[str(x)]
    )

    # Get paths to all images
    orig_img_paths = df_metadata_raw["filename"].map(
        lambda x: os.path.join(data_dir, "image", "ultrasound", x)
    ).tolist()
    extra_metadata_cols = df_metadata_raw[["age", "gender", "view", "sub_view"]].to_dict(orient="list")

    # Save sampled images as PNG
    # NOTE: Infix function saves annotated organ specified in image filename
    df_metadata = re_anonymize_and_save_as_png(
        orig_img_paths, dataset_key, img_subdir, metadata_subdir,
        overwrite=overwrite,
        **extra_metadata_cols
    )

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source": "Harvard Dataverse",
            "URL": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SKP9IB",
            "Seed": seed,
            "(Before) Number of Images": len(df_metadata_imgs),
            "(After) Number of Images": len(df_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


def process_pelvis_ovarian_dataset(data_dir=None, save_dir=None, seed=SEED,
                                   overwrite=False):
    """
    Process the Pelvis - Ovarian Ultrasound dataset from FigShare.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If True, overwrite existing files, by default False.
    """
    dataset_name = "Pelvis - Ovarian"
    dataset_key = "pelvis_ovarian"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # Remove provenance before redoing
    provenance_path = os.path.join(metadata_subdir, "provenance.txt")
    if os.path.exists(provenance_path):
        os.remove(provenance_path)

    # If overwriting, delete existing images
    if overwrite:
        cleanup_img_dir(img_subdir)

    # Get all the image files
    print(f"[{dataset_name}] Dataset creation started...")
    raw_img_subdir = os.path.join(data_dir, "dataset", "dataset", "OTU_2D")
    raw_img_paths = glob(os.path.join(raw_img_subdir, "train", "train_image", "*.JPG"))
    raw_img_paths = raw_img_paths + glob(os.path.join(raw_img_subdir, "test", "image", "*.JPG"))

    # Add metadata for view (organ)
    extra_metadata_cols = {"view": ["pelvis_ovarian"] * len(raw_img_paths)}

    # Save sampled images as PNG
    df_metadata = re_anonymize_and_save_as_png(
        raw_img_paths, dataset_key, img_subdir, metadata_subdir,
        overwrite=overwrite,
        **extra_metadata_cols,
    )

    # Duplicate metadata file
    df_metadata.to_csv(os.path.join(metadata_subdir, "pelvis-metadata.csv"), index=False)

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source":  "FigShare (MMOTU dataset)",
            "URL": "https://figshare.com/articles/dataset/_zip/25058690?file=44222642",
            "Seed": seed,
            "(Before) Number of Images": len(raw_img_paths),
            "(After) Number of Images": len(df_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


def process_fetal_planes_dataset(data_dir=None, save_dir=None, seed=SEED,
                                 overwrite=False):
    """
    Process the Fetal Planes Ultrasound dataset from Zenodo.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If True, overwrite existing files, by default False.
    """
    dataset_name = "Fetal - Planes"
    dataset_key = "fetal_planes"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # Remove provenance before redoing
    provenance_path = os.path.join(metadata_subdir, "provenance.txt")
    if os.path.exists(provenance_path):
        os.remove(provenance_path)

    # If overwriting, delete existing images
    if overwrite:
        cleanup_img_dir(img_subdir)

    # Load raw metadata
    df_metadata_raw = pd.read_csv(os.path.join(data_dir, "dataset", "FETAL_PLANES_DB_data.csv"), sep=";")

    # Sample 400 images from every plane
    df_raw_sampled = df_metadata_raw.groupby("Plane").sample(n=400, random_state=seed)
    # Sample 1 image for every patient
    # NOTE: This is to increase plane diversity and ensure 1 patient for every frame
    df_raw_sampled = df_raw_sampled.groupby("Patient_num").sample(n=1, random_state=seed)

    # Rename columns
    df_raw_sampled = df_raw_sampled.rename(
        columns={
            "Patient_num": "patient_id",
            "US_Machine": "us_machine",
            "Plane": "view",
            "Brain_plane": "sub_view",
        }
    )

    # Add filename and filter columns
    df_raw_sampled["filename"] = df_raw_sampled["Image_name"]
    df_raw_sampled = df_raw_sampled[["patient_id", "filename", "view", "sub_view"]]

    # Process view labels
    df_raw_sampled["view"] = df_raw_sampled["view"].map(
        lambda x: x.lower().replace(" ", "_").replace("other", "fetal_other")
    )
    df_raw_sampled["sub_view"] = df_raw_sampled["sub_view"].map(
        lambda x: x.lower().replace(" ", "_") if x != "Not A Brain" else None
    )
    df_raw_sampled["sub_view"] = "fetal_brain-" + df_raw_sampled["sub_view"]

    # Get paths to all images
    orig_img_paths = df_raw_sampled["filename"].map(
        lambda x: os.path.join(data_dir, "dataset", "Images", x + ".png")
    ).tolist()
    extra_metadata_cols = df_raw_sampled[["view", "sub_view"]].to_dict(orient="list")

    # Save sampled images as PNG
    df_metadata = re_anonymize_and_save_as_png(
        orig_img_paths, dataset_key, img_subdir, metadata_subdir,
        overwrite=overwrite,
        **extra_metadata_cols
    )

    # Duplicate metadata file
    df_metadata.to_csv(os.path.join(metadata_subdir, "fetal-metadata.csv"), index=False)

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source": "Zenodo - Fetal Planes DB",
            "URL": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SKP9IB",
            "Seed": seed,
            "(Before) Number of Images": len(df_metadata_raw),
            "(After) Number of Images": len(df_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


################################################################################
#                    Processing Functions (Video Datasets)                     #
################################################################################
def process_lung_dataset(data_dir=None, save_dir=None, seed=SEED,
                         overwrite=False):
    """
    Process the Covid19 lung ultrasound video dataset from GitHub.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If True, overwrite existing files, by default False.
    """
    dataset_name = "Lung"
    dataset_key = "lung"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    video_subdir = os.path.join(save_dir, "video")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(video_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)
    print(f"[{dataset_name}] Dataset creation started...")

    # Remove provenance before redoing
    provenance_path = os.path.join(metadata_subdir, "provenance.txt")
    if os.path.exists(provenance_path):
        os.remove(provenance_path)

    # Load metadata
    df_metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    # Get all the image files
    local_paths = df_metadata["local_save_path"].map(lambda x: os.path.join(RAW_DATA_DIR, x)).tolist()

    # Convert each video into image frames
    accum_old_new_mapping = {"old_video_path": [], "new_video_id": [], "num_frames": []}
    # NOTE: `id` refer to image/frame ID, while `video_id` refers to video ID
    accum_metadata = {"id": [], "video_id": [], "path": []}
    for idx, local_path in tqdm(enumerate(local_paths)):
        video_idx = f"{dataset_key}-{idx+1}"
        # Create subdirectory for this video's frames
        curr_video_subdir = os.path.join(video_subdir, video_idx)
        os.makedirs(curr_video_subdir, exist_ok=True)

        # Convert video to frames
        frames_paths = convert_video_to_frames(
            local_path, curr_video_subdir, f"{video_idx}-",
            overwrite=overwrite)
        num_frames = len(frames_paths)

        # Create indices for each frame/image
        frame_indices = [f"{video_idx}-{i+1}" for i in range(num_frames)]
        accum_metadata["id"].extend(frame_indices)
        accum_metadata["video_id"].extend([video_idx] * num_frames)

        # Store new path
        accum_metadata["path"].extend([remove_home_dir(x) for x in frames_paths])

        # Store old video filename and new assigned filename
        accum_old_new_mapping["old_video_path"].append(remove_home_dir(local_path))
        accum_old_new_mapping["new_video_id"].append(video_idx)
        accum_old_new_mapping["num_frames"].append(num_frames)

    # Save dataframe of old path to new path
    df_old_new = pd.DataFrame(accum_old_new_mapping)
    df_old_new.to_csv(os.path.join(metadata_subdir, f"{dataset_key}-old_file_mapping.csv"), index=False)

    # Save new metadata dataframe
    df_new_metadata = pd.DataFrame(accum_metadata)
    df_new_metadata["view"] = "lung"
    df_new_metadata.to_csv(os.path.join(metadata_subdir, f"{dataset_key}-metadata.csv"), index=False)

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source":  "GitHub",
            "URL": "https://github.com/jannisborn/covid19_ultrasound/",
            "Seed": seed,
            "(Before) Number of Videos": len(df_metadata),
            "(After) Number of Images": len(df_new_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


def process_pocus_atlas_dataset(data_dir=None, save_dir=None, seed=SEED,
                                overwrite=False):
    """
    Process the scraped POCUS Atlas dataset.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If True, overwrite existing files, by default False.
    """    
    dataset_name = "POCUS Atlas"
    dataset_key = "pocus_atlas"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create directory paths
    video_subdir = os.path.join(save_dir, "video")
    metadata_subdir = os.path.join(save_dir, "metadata")

    # If overwriting, delete existing images
    if overwrite and os.path.exists(video_subdir):
        print(f"[{dataset_name}] Removing existing videos...")
        cleanup_img_dir(video_subdir)

    # Create directories
    os.makedirs(video_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)
    print(f"[{dataset_name}] Dataset creation started...")

    # Load metadata
    df_metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    # Get all the video files
    local_paths = df_metadata["local_save_path"].map(lambda x: os.path.join(RAW_DATA_DIR, x)).tolist()



    # Convert each video into image frames
    accum_old_new_mapping = {"old_video_path": [], "new_video_id": [], "num_frames": []}
    # NOTE: `id` refer to image/frame ID, while `video_id` refers to video ID
    accum_metadata = {
        "id": [],
        "video_id": [],
        "path": [],
        "view": [],
        "description": [],
    }
    for idx, local_path in tqdm(enumerate(local_paths)):
        # Get focus (e.g., organ/pathology)
        focus = df_metadata["focus"].iloc[idx]
        description = df_metadata["description"].iloc[idx]

        # Create video index using dataset, focus and index
        video_idx = f"{dataset_key}-{focus}-{idx+1}"
        # Create subdirectory for this video's frames
        curr_video_subdir = os.path.join(video_subdir, video_idx)
        os.makedirs(curr_video_subdir, exist_ok=True)

        # Convert video to frames
        # NOTE: Separate foreground/background from ultrasound
        frames_paths = convert_video_to_frames(
            local_path, curr_video_subdir, f"{video_idx}-",
            background_dir=DATASET_TO_DIR["clean"]["background"],
            overwrite=overwrite,
        )
        num_frames = len(frames_paths)

        # Create indices for each frame/image
        frame_indices = [f"{video_idx}-{i+1}" for i in range(num_frames)]
        accum_metadata["id"].extend(frame_indices)
        accum_metadata["video_id"].extend([video_idx] * num_frames)
        # Store new path
        accum_metadata["path"].extend([remove_home_dir(x) for x in frames_paths])
        # Store focus (renamed to view) and description
        accum_metadata["view"].extend([focus] * num_frames)
        accum_metadata["description"].extend([description] * num_frames)

        # Store old video filename and new assigned filename
        accum_old_new_mapping["old_video_path"].append(remove_home_dir(local_path))
        accum_old_new_mapping["new_video_id"].append(video_idx)
        accum_old_new_mapping["num_frames"].append(num_frames)

    # Save dataframe of old path to new path
    # NOTE: Rename `focus` to `view`
    df_old_new = pd.DataFrame(accum_old_new_mapping)
    df_old_new["src_url"] = df_metadata["src_url"]
    df_old_new["view"] = df_metadata["focus"]
    df_old_new["description"] = df_metadata["description"]
    df_old_new.to_csv(os.path.join(metadata_subdir, f"{dataset_key}-old_file_mapping.csv"), index=False)

    # Save new metadata dataframe
    df_new_metadata = pd.DataFrame(accum_metadata)
    df_new_metadata.to_csv(os.path.join(metadata_subdir, f"{dataset_key}-metadata.csv"), index=False)

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source":  "POCUS Atlas",
            "URL": "https://www.thepocusatlas.com/",
            "Seed": seed,
            "(Before) Number of Videos": len(df_metadata),
            "(After) Number of Images": len(df_new_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


################################################################################
#                             Aggregation Function                             #
################################################################################
def aggregate_processed_datasets():
    """
    Aggregate all processed datasets into a single metadata dataframe
    """
    save_dir = CLEAN_DATA_DIR
    save_metadata_path = os.path.join(save_dir, "metadata.csv")

    # Load metadata for each dataset and create train/calib/test splits
    print("[OOD Dataset] Aggregating processed datasets...")
    accum_metadata = []
    for dataset in enumerate(CLEAN_DSETS):
        metadata_path = os.path.join(save_dir, dataset, "metadata", f"{dataset}-metadata.csv")
        df_curr = pd.read_csv(metadata_path)
        df_curr["dset"] = dataset

        # Split into train/calib/test
        # SPECIAL CASE 1: POCUS Atlas, completely used for testing
        if dataset == "pocus_atlas":
            df_curr["split"] = "test"
        else:
            df_curr = create_train_calib_test_splits(df_curr)
        accum_metadata.append(df_curr)

    # Concatenate metadata
    df_metadata = pd.concat(accum_metadata, ignore_index=True)

    # Reorder columns
    cols = ["dset", "split", "id", "video_id", "view", "sub_view"]
    cols += [col for col in df_metadata.columns.tolist() if col not in cols]
    df_metadata = df_metadata[cols]

    # Save metadata
    df_metadata.to_csv(save_metadata_path, index=False)
    print("[OOD Dataset] Aggregating processed datasets...DONE")


################################################################################
#                               Helper Functions                               #
################################################################################
def download_kaggle(kaggle_path, save_dir=RAW_DATA_DIR, data_type="dataset"):
    """
    Downloads a Kaggle dataset from KaggleHub and moves it to a specified directory.

    Parameters
    ----------
    kaggle_path : str
        Path to the Kaggle dataset to be downloaded
    save_dir : str, optional
        Destination directory for the downloaded dataset, by default RAW_DATA_DIR
    data_type : str, optional
        One of ("dataset", "competition")
    """
    assert data_type in ("dataset", "competition")

    # Ensure destination exists
    os.makedirs(save_dir, exist_ok=True)

    # Early return, if dataset already exists
    if os.path.exists(os.path.join(save_dir, os.path.basename(kaggle_path))):
        print(f"[Kaggle] Dataset already downloaded: `{kaggle_path}`")
        return

    # Set save directory and download functions
    temp_save_dir = KAGGLE_DATASETS_DIR if data_type == "dataset" else KAGGLE_COMPETITIONS_DIR
    download_func = kagglehub.dataset_download if data_type == "dataset" else kagglehub.competition_download

    # Download from Kaggle
    download_func(kaggle_path)
    saved_path = os.path.join(temp_save_dir, kaggle_path)

    # Assert that it was downloaded in the right place
    assert os.path.exists(saved_path), (
        "Please ensure $HOME is set correctly! Dataset should have been "
        f"downloaded at: `{saved_path}`"
    )

    # Move to destination
    shutil.move(saved_path, save_dir)
    print(f"[Kaggle] Done downloading dataset: `{kaggle_path}`")


def download_github(repo_owner, repo_name, dir_paths, save_dir=RAW_DATA_DIR,
                    fname_regex=None,
                    branch="master"):
    """
    Download files from specified folders from GitHub

    Parameters
    ----------
    repo_owner : str
        Name of GitHub owner
    repo_name : str
        Name of GitHub repo
    dir_paths : list of str
        List of subdirectories to download
    save_dir : str, optional
        Directory to store downloaded files, by default RAW_DATA_DIR
    branch : str, optional
        Name of Git branch , by default "master"
    """
    # Create subdirectory with repo name
    repo_save_dir = os.path.join(save_dir, repo_name)
    os.makedirs(repo_save_dir, exist_ok=True)

    # Get list of filenames
    for dir_path in dir_paths:
        # Create subdirectory
        curr_save_dir = os.path.join(repo_save_dir, dir_path)
        os.makedirs(curr_save_dir, exist_ok=True)

        # Get list of filenames in Git directory
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{dir_path}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to retrieve files from GitHub: `{repo_owner}/{repo_name}/{dir_path}`")
        filenames = [file['name'] for file in response.json() if file['type'] == 'file']

        # If filename regex provided, filter filenames
        if fname_regex:
            filenames = [fname for fname in filenames if re.match(fname_regex, fname)]

        # Download all files
        for filename in tqdm(filenames):
            file_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{dir_path}/{filename}"
            urllib.request.urlretrieve(file_url, os.path.join(curr_save_dir, filename))


def download_from_url(url, save_name, save_dir=RAW_DATA_DIR, unzip=False):
    """
    Download file from specified URL and save to disk.

    Parameters
    ----------
    url : str
        URL of file to download
    save_name : str
        Name to save file as
    save_dir : str, optional
        Directory to save file in, by default RAW_DATA_DIR
    unzip : bool, optional
        If True, unzip file after saving, by default False

    Returns
    -------
    str
        Path to saved file
    """
    # Download file from the internet
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading file from URL: `{url}`")
    urllib.request.urlretrieve(url, save_path)
    print("Downloading file from URL...DONE")

    # If specified, unzip file
    if unzip:
        assert save_name.endswith(".zip"), "If unzipping, initial save name must end with `.zip`"
        with zipfile.ZipFile(save_path, "r") as zip_handler:
            new_save_path = os.path.join(save_dir, save_name.replace(".zip", ""))
            print(f"Unzipping file: `{save_path}` to `{new_save_path}`")
            zip_handler.extractall(new_save_path)
            print("Unzipping file...DONE")

    return new_save_path


def remove_home_dir(path):
    """
    Remove home directory from path

    Parameters
    ----------
    path : str
        Arbitrary path

    Returns
    -------
    str
        Path without home directory
    """
    return path.replace("~/", "").replace(f"{RAW_DATA_DIR}/", "").replace(f"{CLEAN_DATA_DIR}/", "")


def cleanup_img_dir(img_dir):
    """
    Remove all images in the specified directory if it is empty.

    Parameters
    ----------
    img_dir : str
        Path to the directory containing images to be removed.
    """
    if not os.listdir(img_dir):
        return
    print(f"Removing images processed from earlier run...")
    shutil.rmtree(img_dir)
    print(f"Removing images processed from earlier run...DONE")


def write_provenance_file(dataset_name, metadata_dir, text_kwargs):
    """
    Creates a text file with the provenance for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of dataset
    metadata_dir : str
        Path to metadata directory to store provenance in
    text_kwargs : dict
        Any additional text to include in the provenance file. Should be a dictionary
        where each key is a string and each value is a string. For example:
        `{"URL": "https://example.com", "Date": "2023-02-16"}`
    """
    # Create a text file with the provenance
    with open(os.path.join(metadata_dir, "provenance.txt"), "a") as f:
        f.write(
f"""\n################################################################################
                             `{dataset_name}` Dataset                             
################################################################################""")
        f.write("\nCreated with `prep_ood_us_dataset.py`")
        for k, v in text_kwargs.items():
            f.write(f"\n{k}: {v}")
        f.write("\n\n")


def filter_paths_for_masks(paths):
    """
    Filter paths to remove all masks
    """
    return [path for path in paths if "mask" not in os.path.basename(path)]


def re_anonymize_and_save_as_png(
        img_paths, dataset_key, img_subdir, metadata_subdir,
        infix_name_func=None,
        overwrite=False,
        **extra_metadata_cols,
    ):
    """
    Re-anonymize image paths and save images as PNG format with new unique identifiers.

    Parameters
    ----------
    img_paths : list of str
        List of file paths to the original images.
    dataset_key : str
        Unique key for the dataset used as prefix for new image identifiers.
    img_subdir : str
        Directory to save the new PNG images.
    metadata_subdir : str
        Directory to save metadata files containing path mappings and identifiers.
    overwrite : bool, optional
        If True, existing files with the same name will be overwritten, by default False.
    extra_metadata_cols : Keyword arguments
        Each keyword argument is a list of metadata corresponding to each image
        path that can be saved additionally, even after anonymizing

    Side Effects
    ------------
    - Saves the images in the specified subdirectory in PNG format.
    - Generates and saves two CSV metadata files:
        1. Mapping of old image paths to new paths.
        2. Mapping of new identifiers to new image paths.

    Returns
    -------
    pd.DataFrame
        Contains created metadata file with new IDs and paths
    """
    # Save sampled images as PNG
    map_path = {"old_path": [], "new_path": []}
    new_ids = []
    for old_img_path in tqdm(img_paths):
        # Create ID
        # 1. Function specified to process original filename to create infix
        if infix_name_func is not None:
            infix = infix_name_func(old_img_path)
            curr_id = f"{dataset_key}-{infix}-{shortuuid.uuid()}"
        # 2. If no function provided, then simply use dataset name and generated short UUID
        else:
            curr_id = f"{dataset_key}-{shortuuid.uuid()}"

        # Create new path and store old path
        dst_path = os.path.join(img_subdir, f"{curr_id}.png")
        map_path["old_path"].append(old_img_path)
        map_path["new_path"].append(dst_path)
        new_ids.append(curr_id)

        # Skip, if already exists and not overwriting
        if os.path.exists(dst_path) and not overwrite:
            continue

        # Open JPEG and store at destination as PNG
        img = Image.open(old_img_path)
        img.save(dst_path, "png")

    # Ensure IDs are unique
    assert len(new_ids) == len(set(new_ids)), "ShortUUIDs are not unique! Need to fix this manually!"

    # Remove base directory from paths
    for key, paths in map_path.items():
        map_path[key] = [remove_home_dir(path) for path in paths]

    # Store mapping of old path to new filename
    df_old_to_new = pd.DataFrame(map_path)
    df_old_to_new.to_csv(os.path.join(metadata_subdir, f"{dataset_key}-old_file_mapping.csv"), index=False)

    # Store mapping of UUID to new path
    df_metadata = pd.DataFrame({
        "id": new_ids,
        "path": map_path["new_path"],
        **extra_metadata_cols,
    })
    df_metadata.to_csv(os.path.join(metadata_subdir, f"{dataset_key}-metadata.csv"), index=False)

    return df_metadata


def try_video_download_all_exts(path, extensions, save_dir):
    """
    Download online file using various extensions.

    Note
    ----
    Useful if extension online is not known

    Parameters
    ----------
    path : str
        Path to file in Google Drive
    extensions : list of str
        List of extensions to try
    save_dir : str
        Directory to save file in

    Returns
    -------
    str
        Path to saved file, or None if file does not exist
    """
    filename = os.path.basename(path)
    save_path = os.path.join(save_dir, filename)

    # Try all extensions
    for ext in extensions:
        curr_raw_path = f"{path}.{ext}"
        curr_save_path = f"{save_path}.{ext}"
        try:
            urllib.request.urlretrieve(curr_raw_path, curr_save_path)
            return curr_save_path
        except HTTPError:
            pass

    return None


def convert_video_to_frames(
        path, save_dir, prefix_fname="",
        background_dir=None,
        overwrite=False,
    ):
    """
    Convert video to image frames

    Parameters
    ----------
    path : str
        Path to video
    save_dir : str
        Path to directory to save video
    prefix_fname : str
        Prefix to prepend to all image frames
    background_dir : str
        If provided, save extracted background to a directory
    overwrite : bool
        If True, overwrite existing frames, by default False. Otherwise, simply
        return filenames

    Returns
    -------
    list of str
        Path to saved image frames
    """
    os.makedirs(save_dir, exist_ok=True)

    # Simply return filenames if already exists
    if not overwrite and os.listdir(save_dir):
        print("[Video Conversion] Already exists, skipping...")
        num_files = len(os.listdir(save_dir))
        # Recreate filenames
        idx = 1
        paths = [f"{save_dir}/{prefix_fname}{idx+i}.png" for i in range(num_files)]
        assert (
            set(paths) == set(os.listdir(save_dir)),
            f"Unexpected error! Previously extracted video frames have "
            "unexpected file names. Please delete `{save_dir}`"
        )
        return paths

    # Convert video to frames
    vidcap = cv2.VideoCapture(path)
    success, img_arr = vidcap.read()
    idx = 1
    accum_imgs = []
    saved_img_paths = []
    while success:
        curr_img_path = f"{save_dir}/{prefix_fname}{idx}.png"
        # Preprocess image and save to path
        accum_imgs.append(preprocess_and_save_img_array(
            img_arr,
            grayscale=False,
            extract_beamform=True,
            crop=False,
            apply_filter=False,
        ))
        # Load next image
        success, img_arr = vidcap.read()
        idx += 1
        saved_img_paths.append(curr_img_path)

    # Early return, if no images extracted
    if not accum_imgs:
        return []

    # Separate out ultrasound & non-ultrasound part of sequence
    # CASE 1: Only 1 image frame
    if len(accum_imgs) == 1:
        foreground, background = extract_ultrasound_image_foreground(accum_imgs[0])
        foreground = [foreground]
    # CASE 2: Video
    else:
        foreground, background = extract_ultrasound_video_foreground(np.array(accum_imgs))

    # Save extracted ultrasound part to save paths
    for image_idx, save_img_path in enumerate(saved_img_paths):
        cv2.imwrite(save_img_path, foreground[image_idx])

    # If specified, extract background image
    if background_dir:
        # NOTE: Only save if background has at least 25 non-zero pixels
        if background is not None and (background > 30).sum() > 25:
            os.makedirs(background_dir, exist_ok=True)
            cv2.imwrite(f"{background_dir}/{prefix_fname}.png", background)

    return saved_img_paths


def convert_dicom_to_frames(
        path, save_dir, prefix_fname="", grayscale=True,
        uniform_num_samples=-1,
        overwrite=False):
    """
    Convert DICOM image/video to 1+ image frames

    Parameters
    ----------
    path : str
        Path to video
    save_dir : str
        Path to directory to save video
    prefix_fname : str
        Prefix to prepend to all image frames
    grayscale : bool
        If True, save as grayscale image.
    uniform_num_samples : int, optional
        If DICOM contains a video and this value is > 0, sample uniformly across
        the number of frames in the video.
    overwrite : bool
        If True, overwrite existing frames, by default False. Otherwise, simply
        return filenames

    Returns
    -------
    list of str
        Path to saved image frames
    """
    # Lazy import to speed up file loading
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is not installed. Please install it using `pip install pydicom`"
        )
    os.makedirs(save_dir, exist_ok=True)

    # Simply return filenames if already exists
    if not overwrite and os.listdir(save_dir):
        print("[DICOM Conversion] Already exists, skipping...")
        exist_paths = os.listdir(save_dir)
        num_files = len(exist_paths)
        # Recreate filenames
        paths = [f"{save_dir}/{prefix_fname}{1+i}.png" for i in range(num_files)]
        assert (
            set(paths) == set(exist_paths),
            "Unexpected error! Previously extracted video frames have "
            f"unexpected file names. Please delete `{save_dir}`"
        )
        return paths

    # Load DICOM
    assert os.path.exists(path), f"DICOM does not exist at path! \n\tPath: {path}"
    dicom_obj = pydicom.dcmread(path)

    # CASE 1: A single image
    if not hasattr(dicom_obj, "NumberOfFrames"):
        img_arr = dicom_obj.pixel_array

        # Preprocess image and save to path
        preprocess_and_save_img_array(
            img_arr, grayscale,
            save_path=f"{save_dir}/{prefix_fname}1.png"
        )

    # CASE 2: A sequence of image frames
    num_frames = int(dicom_obj.NumberOfFrames)

    # Get frame indices based on sampling choice
    img_indices = list(range(num_frames))
    # 1. Deterministically sample uniformly across the sequence
    if uniform_num_samples > 0:
        if uniform_num_samples > num_frames:
            print("Cannot sample more frames than available! Defaulting to all frames...")
        # Uniformly sampling
        else:
            img_indices = list(np.linspace(0, num_frames-1, uniform_num_samples, dtype=int))

    # Get all image frames and save them
    converted_imgs = []
    for idx, arr_idx in enumerate(img_indices):
        curr_img_path = f"{save_dir}/{prefix_fname}{idx+1}.png"
        img_arr = dicom_obj.pixel_array[arr_idx]

        # Preprocess image and save to path
        preprocess_and_save_img_array(
            img_arr, grayscale,
            save_path=curr_img_path
        )
        converted_imgs.append(curr_img_path)
    return converted_imgs


def create_train_calib_test_splits(df_metadata):
    """
    Given the "cleaned" metadata for 1 organ dataset, create train, calibration,
    and test splits

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Cleaned metadata for 1 organ (e.g., abdomen/lung)
    """
    # Check if split column exists and found splits
    contains_split_col = "split" in df_metadata.columns.tolist()
    found_splits = []
    if contains_split_col:
        found_splits = df_metadata["split"].unique().tolist()
        # Skip, if train/calib/test splits already made
        if len(found_splits) == 3:
            return df_metadata

    # Create copy to avoid in-place modification
    df_metadata = df_metadata.copy()

    # Split on image OR video ID, if applicable
    id_col = "id"
    if "video_id" in df_metadata.columns:
        id_col = "video_id"

    # Split only on IDs
    df_metadata = df_metadata.sample(frac=1, random_state=SEED)
    ids = df_metadata[id_col].unique()

    # Add split column if not already present
    if not contains_split_col:
        df_metadata["split"] = None

    # Create test split, if not already done
    # NOTE: Set aside 50% for held-out testing
    if "test" not in found_splits:
        test_size = int(0.5 * len(ids))
        test_ids = ids[:test_size]
        test_mask = df_metadata[id_col].isin(test_ids)
        df_metadata.loc[test_mask, "split"] = "test"
    other_ids = df_metadata[df_metadata["split"] != "test"][id_col].unique()

    # Create calibration set split, if not already done
    if "calib" not in found_splits:
        calib_size = int(0.5 * len(other_ids))
        calib_ids = other_ids[:calib_size]
        calib_mask = df_metadata[id_col].isin(calib_ids)
        df_metadata.loc[calib_mask, "split"] = "calib"

    # Remaining unassigned samples will be used for training
    df_metadata["split"] = df_metadata["split"].fillna("train")

    # Sort by image IDs
    df_metadata = df_metadata.sort_values(id_col, ignore_index=True)

    return df_metadata


def convert_img_to_uint8(img_arr):
    """
    Convert images (e.g., UINT16) to UINT8.

    Parameters
    ----------
    img_arr : np.ndarray
        Image array to be converted.

    Returns
    -------
    img_arr : np.ndarray
        Converted image array.
    """
    # CASE 0: Image is already UINT8
    if img_arr.dtype == np.uint8:
        return img_arr

    # CASE 1: If image is UINT16, convert to UINT8 by dividing by 256
    if img_arr.dtype == np.uint16:
        img_arr = img_arr.astype(np.float32)
        assert img_arr.max() > 255, f"[UINT16 to UINT8] Image has pixel value > 255! Max: {img_arr.max()}"
        return np.clip((img_arr / 256), 0, 255).astype(np.uint8)
    # CASE 2: If image is between 0 and 1, then multiply by 255
    elif img_arr.min() >= 0 and img_arr.max() <= 1:
        return np.clip((img_arr * 255), 0, 255).astype(np.uint8)

    # Raise error with unhandled dtype
    raise NotImplementedError(f"[UINT16 to UINT8] Unsupported image type! dtype: `{img_arr.dtype}`")


def preprocess_and_save_img_array(
        img_arr, grayscale=True, extract_beamform=False,
        save_path=None, background_save_path=None,
        **kwargs,
    ):
    """
    Preprocess the input image array by converting it to UINT8 and optionally 
    ensuring it is in grayscale format.

    Parameters
    ----------
    img_arr : np.ndarray
        Image array to be preprocessed. Expected to be in either UINT16 or 
        RGB format if conversion is necessary.
    grayscale : bool, optional
        If True, ensures the output image is in grayscale format. Default is True.
    extract_beamform : bool, optional
        If True, extract ultrasound beamform part from image. Default is False.
    save_path : str, optional
        Path to save the preprocessed image array. Default is None.
    background_save_path : str, optional
        If `extract_beamform` and provided, save the non-ultrasound part of the image.
        Default is None.
    **kwargs : Keyword arguments
        Additional keyword arguments to be passed to `extract_ultrasound_image_foreground`

    Returns
    -------
    np.ndarray
        The processed image array in UINT8 format and optionally in grayscale.
    """
    # Preprocess image
    # 1. Convert to UINT8
    img_arr = convert_img_to_uint8(img_arr)

    # If specified, extract beamform part of ultrasound image
    if extract_beamform:
        img_arr, background_arr = extract_ultrasound_image_foreground(img_arr, **kwargs)
        # Save background, if specified
        if background_save_path:
            cv2.imwrite(background_save_path, background_arr)

    # 2. Ensure grayscale image, if specified
    if grayscale and len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    # Save image to file, if specified
    if save_path:
        cv2.imwrite(save_path, img_arr)

    return img_arr


def is_image_dark(img_arr):
    """
    Return True if image is more than 75% of the image is dark/black pixels,
    and False otherwise

    Parameters
    ----------
    img_arr : np.array
        Image array with pixel values in [0, 255]

    Returns
    -------
    bool
        True if image is dark, False otherwise
    """
    # Convert to grayscale if not already
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    # Checks if more than 75% of the image is dark pixels
    return np.mean(img_arr < 30) >= 0.60


def extract_ultrasound_video_foreground(img_sequence, apply_filter=True, crop=True):
    """
    Split ultrasound video into foreground (ultrasound beamform) and background
    (unecessary static parts).

    Parameters
    ----------
    img_sequence : np.ndarray
        Image sequence to separate foreground from background. Of shape (N, H, W, C)
    apply_filter : bool, optional
        If True, apply median blur filter to image
    crop : bool, optional
        If True, return cropped image

    Returns
    -------
    (np.ndarray, np.ndarray)
        (i) Ultrasound video with be beamform extracted of shape (N, H, W)
        (ii) Non-ultrasound part of image frames of shape (H, W) or None if not
             exists
    """
    img_sequence = img_sequence.astype(np.uint8)

    # Convert to grayscale
    if len(img_sequence.shape) == 4 and img_sequence.shape[3] == 3:
        grayscale_imgs = []
        for idx in range(len(img_sequence)):
            grayscale_imgs.append(cv2.cvtColor(img_sequence[idx], cv2.COLOR_RGB2GRAY))
        img_sequence = np.stack(grayscale_imgs, axis=0)

    # Create mask of shape (H, W) that indicates parts of image with no variation
    dynamic_mask = (np.std(img_sequence, axis=0) != 0)
    dynamic_mask = (255*dynamic_mask).astype(np.uint8)

    # Use maximum pixel intensity to fill in the mask
    # NOTE: Bright pixels by the mask should be included
    max_img = img_sequence.max(0)
    dynamic_mask = fill_mask(max_img, dynamic_mask, intensity_threshold=15)

    # If specified, use median blur filter to fill in the gaps and remove noise
    if apply_filter:
        dynamic_mask = cv2.medianBlur(dynamic_mask, 5)
        # Convert back to binary mask
        dynamic_mask = (255*(dynamic_mask > 0)).astype(np.uint8)

    # Split ultrasound video into ultrasound video and non-ultrasound image
    ultrasound_part, non_ultrasound_part = img_sequence.copy(), img_sequence.copy()
    ultrasound_part[:, ~dynamic_mask.astype(bool)] = 0
    non_ultrasound_part[:, dynamic_mask.astype(bool)] = 0
    # NOTE: Assume that non-ultrasound static part only needs 1 image
    non_ultrasound_part = non_ultrasound_part[0]

    # Early return, if not cropping
    if not crop:
        if non_ultrasound_part.astype(bool).sum() == 0:
            non_ultrasound_part = None
        return ultrasound_part, non_ultrasound_part

    # Get tightest crop of ultrasound image
    y_min, y_max, x_min, x_max = create_tight_crop(dynamic_mask)
    ultrasound_part = ultrasound_part[:, y_min:y_max, x_min:x_max]

    # Get tightest crop of background information
    if non_ultrasound_part.astype(bool).sum() > 0:
        y_min, y_max, x_min, x_max = create_tight_crop(non_ultrasound_part)
        non_ultrasound_part = non_ultrasound_part[y_min:y_max, x_min:x_max]
    else:
        non_ultrasound_part = None
    return ultrasound_part, non_ultrasound_part


def extract_ultrasound_image_foreground(img, apply_filter=True, crop=True):
    """
    Split ultrasound image into ultrasound (beamform) and non-ultrasound (unecessary static parts).

    Parameters
    ----------
    img : np.ndarray
        Ultrasound image to separate ultrasound from non-ultrasound part.
        Of shape (H, W, C) or (H, W)
    apply_filter : bool, optional
        If True, apply median blur filter to image
    crop : bool, optional
        If True, return cropped image

    Returns
    -------
    (np.ndarray, np.ndarray)
        (i) Cropped ultrasound part of image of shape (?, ?)
        (ii) Cropped non-ultrasound part of image of shape (?, ?)
    """
    middle_idx = img.shape[1] // 2

    # Convert to grayscale and get is colored mask for center column of image
    is_colored_center_mask = np.zeros_like(len(img), dtype=bool)
    if len(img.shape) == 3 and img.shape[2] == 3:
        is_colored_center_mask = (np.std(img[:, middle_idx], axis=1) >= 5)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # For center pixels that are greater than 50 and not colored, assume it's part of the mask
    active_mask = np.zeros_like(img, dtype=bool)
    active_mask[:, middle_idx] = (img[:, middle_idx] >= 50) & (~is_colored_center_mask)

    # From the center-filled mask, fill in the remaining part of mask
    active_mask = fill_mask(img, active_mask, intensity_threshold=15)

    # If specified, use median blur filter to fill in the gaps and remove noise
    if apply_filter:
        active_mask = cv2.medianBlur(active_mask, 5)
        # Convert back to binary mask
        active_mask = (active_mask > 0)

    # Split ultrasound image into ultrasound part and non-ultrasound part
    active_mask_bool = active_mask.astype(bool)
    ultrasound_part, non_ultrasound_part = img, img.copy()
    ultrasound_part[~active_mask_bool] = 0
    non_ultrasound_part[active_mask_bool] = 0

    # Early return, if not cropping
    if not crop:
        if non_ultrasound_part.astype(bool).sum() == 0:
            non_ultrasound_part = None
        return ultrasound_part, non_ultrasound_part

    # Get tightest crop of ultrasound image
    y_min, y_max, x_min, x_max = create_tight_crop(active_mask)
    ultrasound_part = ultrasound_part[y_min:y_max, x_min:x_max]

    # Get tightest crop of background information
    if non_ultrasound_part.astype(bool).sum() > 0:
        y_min, y_max, x_min, x_max = create_tight_crop(non_ultrasound_part)
        non_ultrasound_part = non_ultrasound_part[y_min:y_max, x_min:x_max]
    else:
        non_ultrasound_part = None
    return ultrasound_part, non_ultrasound_part


def fill_mask(image, mask, intensity_threshold=1):
    """
    Fill the mask by using the pixel intensity values that are greater than the threshold.

    Parameters
    ----------
    image : np.ndarray
        The ultrasound image.
    mask : np.ndarray
        The incomplete mask.
    intensity_threshold : int, optional
        The intensity threshold to consider pixels as part of the mask.

    Returns
    -------
    np.ndarray
        The filled mask.
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # Create a copy of the mask to update
    filled_mask = mask.copy()

    # Get the coordinates of the initial mask pixels
    initial_points = np.argwhere(mask > 0)

    # Define the 8-connected neighborhood
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Use a deque for efficient queue operations
    queue = deque(initial_points)

    # Region growing algorithm
    while queue:
        x, y = queue.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < gray_image.shape[0] and 0 <= ny < gray_image.shape[1]:
                if filled_mask[nx, ny] == 0 and gray_image[nx, ny] > intensity_threshold:
                    filled_mask[nx, ny] = 255
                    queue.append((nx, ny))

    return filled_mask


def create_tight_crop(image):
    """
    Get tightest crop that doesn't remove any non-zero pixels

    Parameters
    ----------
    image : np.ndarray
        The image to be cropped.

    Returns
    -------
    tuple of int
        Coordinates to get the tightest crop (y_min, y_max, x_min, x_max)
        Returns (None, None, None, None) if image has no active pixel
    """
    # Find the coordinates of non-zero pixels
    non_zero_coords = np.argwhere(image > 0)

    # Early return, if image is empty
    if len(non_zero_coords) == 0:
        return None, None, None, None

    # Get the bounding box of the non-zero pixels
    top_left = non_zero_coords.min(axis=0)
    bottom_right = non_zero_coords.max(axis=0)
    # Crop the image using the bounding box coordinates
    y_min, y_max, x_min, x_max = top_left[0], bottom_right[0]+1, top_left[1], bottom_right[1]+1
    return y_min, y_max, x_min, x_max


def merge_tables(left, right, on, how="inner", reset_index=True):
    """
    Merge two tables.

    Parameters
    ----------
    left : pd.DataFrame
        First table to merge
    right : pd.DataFrame
        Second table to merge
    on : str
        Column to join on
    how : str, optional
        Join type, by default "inner"
    reset_index : bool, optional
        Reset index of merged table, by default True`

    Returns
    -------
    pd.DataFrame
        Merged table
    """
    df_all = left.merge(right, how=how, on=on, suffixes=("", "__dup"))
    df_all = df_all.loc[:, ~df_all.columns.str.endswith('__dup')]
    if reset_index:
        df_all = df_all.reset_index(drop=True)
    return df_all


################################################################################
#                             Deprecated Functions                             #
################################################################################
# NOTE: DDTI neck thyroid dataset was replaced by TN3K neck thyroid dataset
def process_neck_thyroid_ddti_dataset(data_dir=None, save_dir=None, seed=SEED, overwrite=False):
    """
    Process the DDTI: Neck Thyroid Ultrasound dataset from Kaggle.

    Note
    ----
    The dataset is downloaded using `kagglehub`. The dataset is processed by
    randomly selecting 1 image per case ID. The images are stored in the
    `images` subdirectory and the metadata is stored in the `metadata`
    subdirectory.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If False, skip processing if the dataset already exists, by default False
    """
    dataset_name = "Neck - Thyroid (Kaggle)"
    dataset_key = "neck_thyroid_ddti"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"][dataset_key]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"][dataset_key]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # If overwriting, delete existing images
    if overwrite:
        cleanup_img_dir(img_subdir)

    # Get all the XML files that correspond to each case
    xml_paths = glob(os.path.join(data_dir, "versions", "1", "*.xml"))
    orig_img_paths = glob(os.path.join(data_dir, "versions", "1", "*.jpg"))

    # Convert each XML file to a case ID
    case_ids = [os.path.basename(xml_path).split(".")[0] for xml_path in xml_paths]

    # If a patient has multiple images, only select one
    sampled_img_paths = []
    for case_id in tqdm(case_ids):
        jpeg_paths = glob(os.path.join(data_dir, "versions", "1", f"{case_id}*.jpg"))
        # If more than 1 JPEG image, randomly choose 1
        if len(jpeg_paths) > 1:
            jpeg_paths = random.sample(jpeg_paths, 1)
        sampled_img_paths.append(jpeg_paths[0])

    # Add metadata for view (organ)
    extra_metadata_cols = {"view": ["thyroid"] * len(sampled_img_paths)}

    # Save sampled images as PNG
    df_metadata = re_anonymize_and_save_as_png(
        sampled_img_paths, dataset_key, img_subdir, metadata_subdir,
        overwrite=overwrite,
        **extra_metadata_cols,
    )

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source": "Kaggle",
            "URL": "https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images",
            "Seed": seed,
            "(Before) Number of Images": len(orig_img_paths),
            "(After) Number of Images": len(df_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


# NOTE: Kaggle neck nerve dataset is not used due to its restrictive license
def process_neck_nerve_dataset(data_dir=None, save_dir=None, seed=SEED,
                               overwrite=False, n=500):
    """
    Process the Neck Nerve Ultrasound dataset from Kaggle.

    Parameters
    ----------
    data_dir : str
        Directory containing the downloaded dataset.
    save_dir : str
        Directory to store the processed dataset.
    seed : int, optional
        Random seed for reproducibility, by default SEED.
    overwrite : bool, optional
        If True, overwrite existing files, by default False.
    n : int, optional
        Number of images to sample, by default 500.
    """
    dataset_name = "Neck - Nerve"
    dataset_key = "neck_nerve"

    # Set default directories
    data_dir = data_dir if data_dir else DATASET_TO_DIR["raw"]["neck_nerve"]
    save_dir = save_dir if save_dir else DATASET_TO_DIR["clean"]["neck_nerve"]

    # Set seed for reproducibility
    random.seed(seed)

    # Create image and metadata subdirectories
    img_subdir = os.path.join(save_dir, "images")
    metadata_subdir = os.path.join(save_dir, "metadata")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(metadata_subdir, exist_ok=True)

    # If overwriting, delete existing images
    if overwrite:
        cleanup_img_dir(img_subdir)

    # Get all the image files
    print(f"[{dataset_name}] Dataset creation started...")
    orig_img_paths = glob(os.path.join(data_dir, "train", "*.tif"))
    orig_img_paths = orig_img_paths + glob(os.path.join(data_dir, "test", "*.tif"))
    orig_img_paths = filter_paths_for_masks(orig_img_paths)

    # Randomly sample N images
    sampled_img_paths = random.sample(orig_img_paths, n)
    print(f"[{dataset_name}] Sampling {n}/{len(orig_img_paths)} images...")

    # Add metadata for view (organ)
    extra_metadata_cols = {"view": ["neck_nerve"] * len(sampled_img_paths)}

    # Save sampled images as PNG
    df_metadata = re_anonymize_and_save_as_png(
        sampled_img_paths, dataset_key, img_subdir, metadata_subdir,
        overwrite=overwrite,
        **extra_metadata_cols,
    )

    # Create a text file with the provenance
    write_provenance_file(
        dataset_name, metadata_subdir,
        {
            "Source":  "Kaggle",
            "URL": "https://www.kaggle.com/c/ultrasound-nerve-segmentation/data",
            "Seed": seed,
            "(Before) Number of Images": len(orig_img_paths),
            "(After) Number of Images": len(df_metadata),
        }
    )
    print(f"[{dataset_name}] Dataset creation started...DONE")


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire()
