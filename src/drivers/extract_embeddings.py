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
        "hn" : "If flagged, extracts embeddings with HN model.",
        "cytoimagenet" : "If flagged, extracts embeddings with CytoImageNet "
                         "model.",
        "imagenet" : "If flagged, extracts embeddings with ImageNet model.",
        "cpc" : "If flagged, extracts embeddings with CPC model.",
        "moco" : "If flagged, extracts embeddings with MoCo model.",
        "raw" : "If flagged, extracts embeddings for raw images."
    }
    parser.add_argument("--hn", action="store_true", help=arg_help["hn"])
    parser.add_argument("--cytoimagenet", action="store_true",
                        help=arg_help["cytoimagenet"])
    parser.add_argument("--imagenet", action="store_true",
                        help=arg_help["imagenet"])
    parser.add_argument("--cpc", action="store_true", help=arg_help["cpc"])
    parser.add_argument("--moco", action="store_true", help=arg_help["moco"])
    parser.add_argument("--raw", action="store_true", help=arg_help["raw"])


def extract_embeds(raw=False, **kwargs):
    """
    Extract embeddings using both ImageNet and CytoImageNet-trained models.

    Parameters
    ----------
    raw : bool, optional
        If True, extracts embeddings for raw images. Otherwise, uses
        preprocessed images, by default False.
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
    img_dir = constants.DIR_IMAGES if not raw else constants.DIR_IMAGES_RAW

    # Check which models to extract embeddings with
    models = [model for model in constants.MODELS if kwargs.get(model)]

    # Extract embeddings
    for model in models:
        embed_suffix = EMBED_SUFFIX_RAW if raw else EMBED_SUFFIX
        save_dir = constants.DIR_EMBEDS + f"/{model}{embed_suffix}"
        embedders.main(model, save_dir, img_dir=img_dir)


def get_embeds(model, raw=False):
    """
    Retrieve extracted deep embeddings using model specified.

    Parameters
    ----------
    model : str
        Name of pretraining dataset for model. Either "cytoimagenet" or
        "imagenet"
    raw : bool, optional
        If True, extracts embeddings for raw images. Otherwise, uses
        preprocessed images, by default False.
    """
    def _get_embeds(model, drop_path=False):
        """
        Loads dataframe containing image embeddings (features).

        Parameters
        ----------
        model : str
            Name of pretraining dataset, used to train model
        drop_path : bool, optional
            If True, drops column related to file path, by default, False.

        Returns
        -------
        pd.DataFrame
            Contains extracted image features and paths to files
        """
        suffix = EMBED_SUFFIX_RAW if raw else EMBED_SUFFIX
        embed_path = constants.DIR_EMBEDS + f"/{model}{suffix}"

        df = pd.read_hdf(embed_path, "embeds")

        if drop_path:
            df = df.drop(columns=["files"])

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


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()


    # 2. Extract embeddings
    extract_embeds(**vars(ARGS))
