"""
plot_umap.py

Description: Plots 2D UMAP embeddings
"""
# Standard libraries
import os

# Non-standard libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Custom libraries
from src.data.constants import DIR_FIGURES, METADATA_FILE
from src.drivers.extract_embeddings import get_umap_embeddings, get_embeds


################################################################################
#                                Plot Settings                                 #
################################################################################
sns.set_style("dark")
plt.style.use('dark_background')
plt.rc('font', family='serif')


################################################################################
#                                  Functions                                   #
################################################################################
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


def plot_umap_all_patients(model, patients, df_embeds_only, color="patient"):
    """
    Plots UMAP for all patients, coloring by patient ID.

    Parameters
    ----------
    model : str
        Name of model, or pretraining dataset used to pretrain model
    patients : pd.Series
        Contains all patient IDs
    df_embeds_only : pd.DataFrame
        Extracted deep embeddings. Does not have file paths in any column.
    color : str
        Color points by "patient" or by "hospital"
    """
    assert color in ("patient", "hospital")

    # Get hospital labels
    hospitals = patients.map(
        lambda x: "Stanford" if x.startswith("SU2") else "SickKids")
    
    # Choose label based on flag
    label = patients if color == "patient" else hospitals

    # Get UMAP embeddings
    umap_embeds_all = get_umap_embeddings(df_embeds_only)

    # Plot by patient
    plot_umap(umap_embeds_all, label, title=f"UMAP (colored by {color})",
              save=True, filename=f"{model}_umap({color})")


def plot_umap_by_view(model, df_labels, filenames, df_embeds_only):
    """
    Plots UMAP for all view-labeled patients, coloring by view.

    Parameters
    ----------
    model : str
        Name of model, or pretraining dataset used to pretrain model
    df_labels : pd.DataFrame
        Contains file paths and view labels
    filenames : pd.Series
        Contains file name of image whose features were extracted
    df_embeds_only : pd.DataFrame
        Extracted deep embeddings. Does not have file paths in any column.
    """
    # Filter for image files with view labels
    filename_to_label = dict(zip(df_labels["filename"], df_labels["label"]))
    views = (filenames.map(lambda x: filename_to_label.get(x, None)))
    idx = views.notna()
    umap_embeds_views = get_umap_embeddings(df_embeds_only[idx])
    view_labels = views[idx]

    # Plot all images by view
    plot_umap(umap_embeds_views, view_labels, title="UMAP (colored by view)",
              save=True, filename=f"{model}_umap(views)")


def plot_umap_for_one_patient(model, patients, us_nums, df_embeds_only):
    """
    Plots UMAP for a single patient.

    Parameters
    ----------
    model : str
        Name of model, or pretraining dataset used to pretrain model
    patients : pd.Series
        Contains all patient IDs
    us_nums : pd.Series
        Contains number in ultrasound sequence capture
    df_embeds_only : pd.DataFrame
        Extracted deep embeddings. Does not have file paths in any column.
    """
    # Plot sequentially by US nums for 1 patient
    patient_selected = patients.unique()[11]
    idx_patient = (patients == patient_selected)
    
    # Sort data points by US number. Re-extract UMAP embeddings (for 1 patient)
    patient_us_nums = us_nums[idx_patient].reset_index(drop=True).sort_values()
    idx_sorted = patient_us_nums.index.tolist()
    umap_embeds_patient = get_umap_embeddings(
        df_embeds_only[idx_patient][idx_sorted])

    plot_umap(umap_embeds_patient, patient_us_nums, line=True, legend=False,
              title=f"UMAP (patient {patient_selected}, colored by US number)",
              save=True, filename=f"{model}_umap(us_num)")


def main(model):
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
    df_embeds = df_embeds.rename(columns={"paths": "files"})

    # Separate file paths from embeddings
    df_embeds["filename"] = df_embeds["files"].map(lambda x: os.path.basename(x))
    filenames = df_embeds["filename"]
    patients = filenames.map(lambda x: x.split("_")[0])
    us_nums = filenames.map(lambda x: int(x.split("_")[-1].split(".jpg")[0]))

    # Isolate UMAP embeddings (all patients)
    df_embeds_only = df_embeds.drop(columns=["files", "filename"])

    plot_umap_all_patients(model, patients, df_embeds_only, color="hospital")
    plot_umap_for_one_patient(model, patients, us_nums, df_embeds_only)
    plot_umap_by_view(model, df_labels, filenames, df_embeds_only)


if __name__ == '__main__':
    for model in ("hn", ):      # "both", "imagenet", "cytoimagenet", 
        main(model)
