"""
extract_embeddings.py

Description: Contains function to load embeddings from trained models.
"""
# Non-standard libraries
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

# Custom libraries
from src.data.constants import DIR_EMBEDS, DIR_IMAGES
from src.models import embedders


################################################################################
#                                  Constants                                   #
################################################################################
EMBED_SUFFIX = "_embeddings(histogram_norm).h5"


################################################################################
#                                  Functions                                   #
################################################################################
def extract_all_embeds():
    """
    Extract embeddings using both ImageNet and CytoImageNet-trained models.
    """
    for model in ("imagenet", "cytoimagenet"):
        embedders.main(model, DIR_EMBEDS + f"/{model}{EMBED_SUFFIX}",
                       img_dir=DIR_IMAGES)


def get_embeds(model):
    """
    Retrieve extracted deep embeddings using model specified.

    Parameters
    ----------
    model : str
        Name of pretraining dataset for model. Either "cytoimagenet" or
        "imagenet"
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
        embed_path = DIR_EMBEDS + f"/{model}{EMBED_SUFFIX}"

        df = pd.read_hdf(embed_path, "embeds")

        if drop_path:
            df = df.drop(columns=["files"])

        return df


    MODELS = ("imagenet", "cytoimagenet")

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
    extract_all_embeds()