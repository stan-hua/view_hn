"""
prep_ood_us_dataset.py

Description: Script to prepare image and metadata for OOD-US dataset

Note
----
The following datasets are used to create the "Ultrasound Out-Of-View Dataset"

Partially Unseen OOD:
1. Neck Ultrasound (1K images)
    1. Adult [Thyroid dataset](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) (480 US images)
    2. [Nerve dataset](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data) (8K US images)
2. Breast Ultrasound (1K images)
    1. Adult [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) (1.57K US images)
3. Lung Ultrasound (1K images)
    1. Adult [COVID POCUS](https://github.com/jannisborn/covid19_ultrasound) dataset (200 US videos; US images)
4. Abdominal Ultrasound (1K images)
    1. [Pediatric Appendix](https://www.kaggle.com/datasets/joebeachcapital/regensburg-pediatric-appendicitis) (2.1K US images; appendix)
    2. Adult [USNotAI](https://github.com/LeeKeyu/abdominal_ultrasound_classification) (360 US images; bladder, kidney, bowel, gallbladder, liver, spleen)
5. Knee Ultrasound (1K images)
    1. Adult (35-70 years) [Knee Ultrasound](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SKP9IB) (>15K US images; knee)
6. Pelvic Ultrasound (1K images)
    1. [Adult ovarian ultrasound dataset](https://figshare.com/articles/dataset/_zip/25058690?file=44222642) (1.2K US images; pelvis)
7. Fetal Ultrasound (1K images)
    1. [Fetal Planes dataset](https://zenodo.org/records/3904280) (12K US images; abdomen, brain, femur and thorax)

Held-Out Unseen OOD Datasets:
1. [POCUS Atlas](https://www.thepocusatlas.com/) (1.7K videos)
2. [Clarius Clinical Gallery](https://clarius.com/about/clinical-gallery/)
"""

# Standard libraries
import json
import os
import random
import shutil
import urllib.request
import requests
import zipfile
from glob import glob
from urllib.error import HTTPError

# Non-standard libraries
import cv2
import kagglehub
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
        "neck_thyroid": os.path.join(RAW_DATA_DIR, "ddti-thyroid-ultrasound-images"),
        "neck_nerve": os.path.join(RAW_DATA_DIR, "ultrasound-nerve-segmentation"),
        "breast": os.path.join(RAW_DATA_DIR, "breast-ultrasound-images-dataset"),
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
        "neck_thyroid": os.path.join(CLEAN_DATA_DIR, "neck"),
        "neck_nerve": os.path.join(CLEAN_DATA_DIR, "neck"),
        "breast": os.path.join(CLEAN_DATA_DIR, "breast"),
        "lung": os.path.join(CLEAN_DATA_DIR, "lung"),
        "abdominal_appendix": os.path.join(CLEAN_DATA_DIR, "abdomen"),
        "abdominal_organs": os.path.join(CLEAN_DATA_DIR, "abdomen"),
        "knee": os.path.join(CLEAN_DATA_DIR, "knee"),
        "pelvis_ovarian": os.path.join(CLEAN_DATA_DIR, "pelvis"),
        "fetal_planes": os.path.join(CLEAN_DATA_DIR, "fetal"),

        # Held-out test sets
        "pocus_atlas": os.path.join(CLEAN_DATA_DIR, "pocus_atlas"),
        "clarius": os.path.join(CLEAN_DATA_DIR, "clarius"),
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
    download_map = {
        "neck_thyroid": {
            "func": download_kaggle,
            "kwargs": {
                "kaggle_path": "dasmehdixtr/ddti-thyroid-ultrasound-images",
                "data_type": "dataset",
                "save_dir": RAW_DATA_DIR,
            }
        },
        "neck_nerve": {
            "func": download_kaggle,
            "kwargs": {
                "kaggle_path": "ultrasound-nerve-segmentation",
                "data_type": "competition",
                "save_dir": RAW_DATA_DIR,
            }
        },
        "breast": {
            "func": download_kaggle,
            "kwargs": {
                "kaggle_path": "aryashah2k/breast-ultrasound-images-dataset",
                "data_type": "dataset",
                "save_dir": RAW_DATA_DIR,
            }
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
        "clarius": {
            "func": download_clarius,
        },
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
def process_datasets(*organs):
    """
    Processes downloaded ultrasound datasets using predefined processing functions.

    Parameters
    ----------
    *organs : Any
        List of organ datasets to download. If None, all available datasets are downloaded.
    """
    # Mapping of organ to processing method
    process_map = {
        "neck": process_neck_datasets,
        "breast": process_breast_dataset,
        "lung": process_lung_dataset,
        "abdomen": process_abdomen_dataset,
        "pelvis": process_pelvis_ovarian_dataset,
        "fetal": process_fetal_planes_dataset,

        # Held-out test sets
        "pocus_atlas": process_pocus_atlas_dataset,
    }

    # If no organs specified, then download all
    if not organs:
        organs = list(process_map.keys())

    # Process each dataset
    print(f"[OOD Ultrasound] Processing data for the following organs: {organs}")
    for organ in organs:
        print(f"Processing dataset: {organ}")
        process_map[organ]()

    # Aggregate all datasets
    aggregate_processed_datasets()


def process_neck_datasets(overwrite=False):
    """
    Process the Ultrasound Neck datasets: thyroid and nerve datasets.

    Parameters
    ----------
    overwrite : bool, optional
        If True, overwrite existing processed data
    """
    clean_neck_dir = DATASET_TO_DIR["clean"]["neck_thyroid"]
    metadata_dir = os.path.join(clean_neck_dir, "metadata")

    # Remove provenance before redoing
    provenance_path = os.path.join(metadata_dir, "provenance.txt")
    if os.path.exists(provenance_path):
        os.remove(provenance_path)

    # Process both neck and thyroid datasets
    process_neck_thyroid_dataset(overwrite=overwrite)
    process_neck_nerve_dataset(overwrite=overwrite)

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


def process_neck_thyroid_dataset(data_dir=None, save_dir=None, seed=SEED, overwrite=False):
    """
    Process the Neck Thyroid Ultrasound dataset from Kaggle.

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
    dataset_name = "Neck - Thyroid"
    dataset_key = "neck_thyroid"

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

    # Create image and metadata subdirectories
    video_subdir = os.path.join(save_dir, "video")
    metadata_subdir = os.path.join(save_dir, "metadata")
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
    accum_metadata = []
    for dataset in CLEAN_DSETS:
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


def convert_video_to_frames(path, save_dir, prefix_fname="", overwrite=False):
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
        return paths

    # Convert video to frames
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    idx = 1
    converted_imgs = []
    while success:
        curr_img_path = f"{save_dir}/{prefix_fname}{idx}.png"
        cv2.imwrite(curr_img_path, image)
        success, image = vidcap.read()
        idx += 1
        converted_imgs.append(curr_img_path)

    return converted_imgs


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
    for img_path in tqdm(os.listdir(img_dir)):
        os.remove(os.path.join(img_dir, img_path))
    print(f"Removing images processed from earlier run...DONE")


def create_train_calib_test_splits(df_metadata):
    """
    Given the "cleaned" metadata for 1 organ dataset, create train, calibration,
    and test splits

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Cleaned metadata for 1 organ (e.g., abdomen/lung)
    """
    # Create copy to avoid in-place modification
    df_metadata = df_metadata.copy()

    # Split on image OR video ID, if applicable
    id_col = "id"
    if "video_id" in df_metadata.columns:
        id_col = "video_id"

    # Split only on IDs
    df_metadata = df_metadata.sample(frac=1, random_state=SEED)
    ids = df_metadata[id_col].unique()

    # Set aside 50% for held-out testing
    test_size = int(0.5 * len(ids))
    test_ids, other_ids, = ids[:test_size], ids[test_size:]

    # Set aside remaining 25% for calibration and 25% for training
    calib_size = int(0.5 * len(other_ids))
    calib_ids, train_ids = other_ids[:calib_size], other_ids[calib_size:]

    # Assign splits
    df_metadata["split"] = None
    for split, ids in [("train", train_ids), ("calib", calib_ids), ("test", test_ids)]:
        mask = df_metadata[id_col].isin(ids)
        df_metadata.loc[mask, "split"] = split

    return df_metadata


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire()
