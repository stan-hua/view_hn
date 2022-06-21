"""
extract_embeddings.py

Description: Extracts embeddings from pretrained models, and plots data on 2D
             plots.
"""
# Standard libraries
import os

# Non-standard libraries
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Custom libraries
from src.data.constants import (DIR_EMBEDS, DIR_IMAGES, DIR_FIGURES,
                                METADATA_FILE)
from src.models import embedders


################################################################################
#                                Plot Settings                                 #
################################################################################
sns.set_style("dark")
plt.style.use('dark_background')
plt.rc('font', family='serif')

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


def plot_umap(embeds, labels, line=False, legend=True, title="",
              save=False, save_dir=DIR_FIGURES, filename="umap"):
    """
    Plot 2D U-Map of extracted image embeddings, colored by <labels>.

    Parameters
    ----------
    embeds : numpy.array
        A 2-dimensional array of N samples.
    labels : list
        Labels to color embeddings by. Must match number of samples (N).
    line : bool, optional
        Connects points in scatterplot sequentially, by default False
    legend : bool, optional
        If True, shows legend, by default True.
    title : str, optional
        Plot title, by default "".
    save : bool, optional
        Filename , by default False
    save_dir : str, optional
        Directory to save plot image in
    filename : str, optional
        Filename to save plot as. Not including extension, by default "umap"
    """
    plt.figure()

    if line:
        sns.lineplot(x=embeds[:, 0], y=embeds[:, 1], sort=False,
                     size=1, alpha=0.3, legend=False)

    sns.scatterplot(x=embeds[:, 0], y=embeds[:, 1],
                    hue=labels,
                    legend="full" if legend else legend,
                    alpha=1,
                    palette="tab20",
                    s=5,
                    linewidth=0)

    if legend:
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    
    if title:
        plt.title(title)

    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    plt.tight_layout()

    # Save Figure
    if save:
        if not os.path.isdir(f"{save_dir}umap/"):
            os.mkdir(f"{save_dir}umap/")
        plt.savefig(f"{save_dir}umap/{filename}.png", bbox_inches='tight',
                    dpi=400)


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


def main_umap(model):
    """
    Retrieves embeddings from specified pretrained model. Then plot UMAPs.

    Parameters
    ----------
    model : str
        Pretraining dataset for model. Either "imagenet" or "cytoimagenet"
    """
    # Load view labels
    df_labels = pd.read_csv(METADATA_FILE).rename(
        columns={"IMG_FILE": "filename", "revised_labels": "label"})

    # Load embeddings
    df_embeds = get_embeds(model)

    # Separate file paths from embeddings
    df_embeds["filename"] = df_embeds["files"].map(lambda x: os.path.basename(x))
    filenames = df_embeds["filename"]
    patients = filenames.map(lambda x: x.split("_")[0])
    us_nums = filenames.map(lambda x: int(x.split("_")[-1].split(".jpg")[0]))

    # Get UMAP embeddings (all patients)
    df_embeds_only = df_embeds.drop(columns=["files", "filename"])
    umap_embeds = get_umap_embeddings(df_embeds_only)

    # Plot by patient
    plot_umap(umap_embeds, patients, title="UMAP (colored by patients)",
              save=True, filename=f"{model}_umap(patient)")

    # Plot sequentially by US nums for 1 patient
    patient_selected = patients.unique()[11]
    idx_patient = (patients == patient_selected)
    
    # Sort data points by US number. Re-extract UMAP embeddings (for 1 patient)
    patient_us_nums = us_nums[idx_patient].reset_index(drop=True).sort_values()
    idx_sorted = patient_us_nums.index.tolist()
    patient_embeds_sorted = get_umap_embeddings(
        df_embeds_only[idx_patient][idx_sorted])

    plot_umap(patient_embeds_sorted, patient_us_nums, line=True, legend=False,
              title=f"UMAP (patient {patient_selected}, colored by US number)",
              save=True, filename=f"{model}_umap(us_num)")

    # Filter for image files with view labels
    filename_to_label = dict(zip(df_labels["filename"], df_labels["label"]))
    views = (filenames.map(lambda x: filename_to_label.get(x, None)))
    idx = views.notna()
    umap_embeds_views = get_umap_embeddings(df_embeds_only[idx])
    view_labels = views[idx]

    # Plot all images by view
    plot_umap(umap_embeds_views, view_labels, title="UMAP (colored by view)",
              save=True, filename=f"{model}_umap(views)")


if __name__ == '__main__':
    for model in ("both", "imagenet", "cytoimagenet", ):
        main_umap(model)
