"""
extract_embeddings.py

Description: Contains function to load embeddings from trained models.
"""
# Standard libraries
import argparse

# Non-standard libraries
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

# Custom libraries
from src.data import constants
from src.data_prep import segment_dataset
from src.models import embedders


################################################################################
#                                  Constants                                   #
################################################################################
EMBED_SUFFIX = "_embeddings(histogram_norm).h5"
EMBED_SUFFIX_RAW = "_embeddings(raw).h5"


################################################################################
#                                  Functions                                   #
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
        "random" : "If flagged, extracts embeddings from a randomly initialized"
                   " EfficientNet model",
        "hn" : "If flagged, extracts embeddings with HN model.",
        "cytoimagenet" : "If flagged, extracts embeddings with CytoImageNet "
                         "model.",
        "imagenet" : "If flagged, extracts embeddings with ImageNet model.",
        "cpc" : "If flagged, extracts embeddings with CPC model.",
        "moco" : "If flagged, extracts embeddings with MoCo model.",
        "raw" : "If flagged, extracts embeddings for raw images.",
        "segmented": "If flagged, extracts embeddings for segmented images"
    }
    parser.add_argument("--random", action="store_true",
                        help=arg_help["random"])
    parser.add_argument("--hn", action="store_true", help=arg_help["hn"])
    parser.add_argument("--cytoimagenet", action="store_true",
                        help=arg_help["cytoimagenet"])
    parser.add_argument("--imagenet", action="store_true",
                        help=arg_help["imagenet"])
    parser.add_argument("--cpc", action="store_true", help=arg_help["cpc"])
    parser.add_argument("--moco", action="store_true", help=arg_help["moco"])
    parser.add_argument("--raw", action="store_true", help=arg_help["raw"])
    parser.add_argument("--segmented", action="store_true",
                        help=arg_help["segmented"])


def extract_embeds(raw=False, segmented=False, reverse_mask=False, **kwargs):
    """
    Extract embeddings using both ImageNet and CytoImageNet-trained models.

    Parameters
    ----------
    raw : bool, optional
        If True, extracts embeddings for raw images. Otherwise, uses
        preprocessed images, by default False.
    segmented : bool, optional
        If True, extracts embeddings for segmented images, by default False.
    reverse_mask : bool, optional
        If True, reverses mask for segmented images, by default False
    kwargs : keyword arguments
        hn : bool, optional
            If True, extracts embeddings using HN model, by default False.
        cytoimagenet : bool, optional
            If True, extracts embeddings using CytoImageNet model, by default
            False.
        imagenet : bool, optional
            If True, extracts embeddings using ImageNet model, by default False.
        cpc : bool, optional
            If True, extracts embeddings using CPC model, by default False.
        moco : bool, optional
            If True, extracts embeddings using MoCo model, by default False.
    """
    # Input sanitization
    assert not (raw and segmented), "Specify only one of (raw, segmented)!"

    # Prepare data-related arguments
    img_dir = None
    img_dataloader = None
    if raw:
        img_dir = constants.DIR_IMAGES_RAW
    elif segmented:
        data_module = segment_dataset.SegmentedUSModule(
            mask_img_dir=constants.DIR_SEGMENT_MASK,
            src_img_dir=constants.DIR_SEGMENT_SRC,
            reverse_mask=reverse_mask)
        img_dataloader = data_module.train_dataloader()
    else:
        img_dir = constants.DIR_IMAGES

    # Check which models to extract embeddings with
    models = [model for model in constants.MODELS if model in kwargs]

    # Extract embeddings
    for model in models:
        save_path = get_save_path(model, raw, segmented, reverse_mask)
        embedders.main(model, save_path,
                       img_dir=img_dir,
                       img_dataloader=img_dataloader)


def get_embeds(model, raw=False, segmented=False, reverse_mask=False):
    """
    Retrieve extracted deep embeddings using model specified.

    Parameters
    ----------
    model : str
        Name of pretraining dataset for model. Either "cytoimagenet" or
        "imagenet"
    raw : bool, optional
        If True, gets extracted embeddings for raw images. Otherwise, uses
        preprocessed images, by default False.
    segmented : bool, optional
        If True, gets extracted embeddings for segmented images, by default False.
    reverse_mask : bool, optional
        If True, gets extracted embeddings where segmentation masks are
        reversed, by default False
    """
    def _get_embeds(model):
        """
        Loads dataframe containing image embeddings (features).

        Parameters
        ----------
        model : str
            Name of pretraining dataset, used to train model

        Returns
        -------
        pd.DataFrame
            Contains extracted image features and paths to files
        """
        suffix = EMBED_SUFFIX_RAW if raw else EMBED_SUFFIX
        embed_path = constants.DIR_EMBEDS + f"/{model}{suffix}"

        embed_path = get_save_path(model, raw, segmented, reverse_mask)
        df = pd.read_hdf(embed_path, "embeds")

        return df

    assert model in constants.MODELS or model == "both"

    if model != "both":
        df_embeds = _get_embeds(model)
    else:
        run = False
        embed_lst = []

        for m in ("cytoimagenet", "imagenet"):
            embed_lst.append(_get_embeds(m, run))
            run = True

        df_embeds = pd.concat(embed_lst, axis=1)

    return df_embeds


def get_umap_embeddings(df_embeds):
    """
    Get 2D-UMAP embeddings.

    Parameters
    ----------
    df_embeds : pd.DataFrame
        Contains embeddings for images
    
    Returns
    -------
    numpy.array
        2-dimensional UMAP embeddings
    """
    # Standardize embeddings
    df_scaled = StandardScaler().fit_transform(df_embeds)

    # Perform dimensionality reduction
    reducer = umap.UMAP(random_state=0)
    umap_embeds = reducer.fit_transform(df_scaled)

    return umap_embeds


def get_save_path(model_name, raw=False, segmented=False, reverse_mask=False):
    """
    Create expected save path from model name and parameters.

    Parameters
    ----------
    model_name : str
        Name of model
    raw : bool, optional
        If True, extracts embeddings for raw images. Otherwise, uses
        preprocessed images, by default False.
    segmented : bool, optional
        If True, extracts embeddings for segmented images, by default False.
    reverse_mask : bool, optional
        If True, reverses mask for segmented images, by default False

    Returns
    -------
    str
        Full path to save embeddings
    """
    embed_suffix = EMBED_SUFFIX_RAW if raw else EMBED_SUFFIX
    segmented_suffix = f"_segmented{'_reverse' if reverse_mask else ''}"
    save_path = f"{constants.DIR_EMBEDS}/{model_name}"\
                f"{segmented_suffix if segmented else ''}{embed_suffix}"

    return save_path


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Extract embeddings
    extract_embeds(**vars(ARGS))
