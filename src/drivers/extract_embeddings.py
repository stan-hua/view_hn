"""
extract_embeddings.py

Description: Contains function to load embeddings from trained models.
"""
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
def extract_all_embeds(raw=False):
    """
    Extract embeddings using both ImageNet and CytoImageNet-trained models.

    Parameters
    ----------
    raw : bool, optional
        If True, extracts embeddings for raw images. Otherwise, uses
        preprocessed images, by default False.
    """
    img_dir = constants.DIR_IMAGES if not raw else constants.DIR_IMAGES_RAW

    for model in ("hn", "imagenet", "cytoimagenet"):
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


    MODELS = ("imagenet", "cytoimagenet", "hn")

    assert model in MODELS or model == "both"

    if model != "both":
        df_embeds = _get_embeds(model)
    else:
        run = False
        embed_lst = []

        for m in MODELS:
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
    extract_all_embeds(raw=True)