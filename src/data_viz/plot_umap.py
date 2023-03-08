"""
plot_umap.py

Description: Plots 2D UMAP embeddings
"""
# Standard libraries
import logging
import os
import random

# Non-standard libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Custom libraries
from src.data import constants
from src.data_prep import utils
from src.drivers.embed import get_embeds
from src.data_viz import utils as viz_utils
from src.data_viz.eda import gridplot_images


################################################################################
#                                    Setup                                     #
################################################################################
# Disable logging
logging.disable()

# Set random seed
random.seed(constants.SEED)

# Plot theme (light/dark)
THEME = "dark"

# Order of labels in plot
VIEW_LABEL_ORDER = ["Sagittal_Right", "Transverse_Right", "Sagittal_Left",
                    "Transverse_Left", "Bladder"]
SIDE_LABEL_ORDER = ["Left", "None", "Right"]
PLANE_LABEL_ORDER = ["Sagittal", "Transverse", "Bladder"]


################################################################################
#                           UMAP Plotting Functions                            #
################################################################################
def plot_umap(embeds, labels, highlight=None, label_order=None, s=7,
              line=False, legend=True, title="", palette="tab10",
              save=False, save_dir=constants.DIR_FIGURES, filename="umap",
              filename_suffix=""):
    """
    Plot 2D U-Map of extracted image embeddings, colored by <labels>.

    Parameters
    ----------
    embeds : numpy.array
        A 2-dimensional array of N samples.
    labels : list
        Labels to color embeddings by. Must match number of samples (N).
    highlight : list, optional
        List of boolean values, corresponding to N samples. Those marked as
        False will appear softer (lower alpha value).
    label_order : list, optional
        Order of unique label values in legend, by default None.
    s : int, optional
        Size of scatterplot points, by default 5.
    line : bool, optional
        Connects points in scatterplot sequentially, by default False
    legend : bool, optional
        If True, shows legend, by default True.
    title : str, optional
        Plot title, by default "".
    palette : str, optional
        Seaborn color palette, by default "tab10".
    save : bool, optional
        Filename , by default False
    save_dir : str, optional
        Directory to save plot image in
    filename : str, optional
        Filename to save plot as. Not including extension, by default "umap"
    filename_suffix : str, optional
        If provided, attach as suffix to filename provided, by default ""
    """
    # Plot configurations
    viz_utils.set_theme(THEME)

    # Create figure
    plt.figure()

    # Draw line
    if line:
        sns.lineplot(x=embeds[:, 0], y=embeds[:, 1], sort=False,
                     size=max(1, s//5), alpha=0.3, legend=False)

    # Draw scatterplot
    if highlight is None:
        sns.scatterplot(x=embeds[:, 0], y=embeds[:, 1],
                        hue=labels,
                        hue_order=label_order,
                        legend="full" if legend else legend,
                        alpha=0.8,
                        palette=palette,
                        s=s,
                        linewidth=0)
    elif len(highlight) != len(embeds):
        raise RuntimeError("Length of `embeds` and `highlight` do not match!")
    else:
        # Draw highlighted points
        sns.scatterplot(x=embeds[highlight][:, 0], y=embeds[highlight][:, 1],
                        hue=labels[highlight],
                        hue_order=label_order,
                        legend=False,
                        alpha=0.8,
                        palette=palette,
                        s=s,
                        linewidth=0)

        # Draw non-highlighted points
        sns.scatterplot(x=embeds[~highlight][:, 0], y=embeds[~highlight][:, 1],
                        hue=labels[~highlight],
                        hue_order=label_order,
                        legend="full" if legend else legend,
                        alpha=0.25,
                        palette=palette,
                        s=s,
                        linewidth=0)

    # Create legend
    if legend:
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    # Create title
    if title:
        plt.title(title)

    # Clear axes labels and ticks
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    # Pack plot
    plt.tight_layout()

    # Save Figure
    if save:
        full_path = f"{save_dir}umap/{filename}{filename_suffix}.png"

        # Check if UMAP directory exists
        if not os.path.isdir(f"{save_dir}umap/"):
            os.mkdir(f"{save_dir}umap/")

        # Check if subdirectory exists
        if not os.path.isdir(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))

        plt.savefig(full_path,
                    bbox_inches='tight',
                    dpi=400)


def plot_umap_all_patients(model, patients, df_embeds_only, color="patient",
                           raw=False,
                           **plot_kwargs):
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
    raw : bool, optional
        If True, loaded embeddings extracted from raw images. Otherwise, uses
        preprocessed images, by default False.
    **plot_kwargs : Keyword arguments to pass into `plot_umap`
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
              filename=f"{model}/{model}_umap{'_raw' if raw else ''}({color})",
              save=True,
              **plot_kwargs)


def plot_umap_by_view(model, view_labels, df_embeds_only,
                      highlight=None,
                      raw=False,
                      hospital="SickKids",
                      **plot_kwargs):
    """
    Plots UMAP for all view-labeled patients, coloring by view.

    Parameters
    ----------
    model : str
        Name of model, or pretraining dataset used to pretrain model
    view_labels : numpy.array
        Contains view labels, corresponding to embeddings extracted. Images
        without label (null value) will be excluded
    df_embeds_only : pd.DataFrame
        Extracted deep embeddings. Does not have file paths in any column.
    raw : bool, optional
        If True, loaded embeddings extracted from raw images. Otherwise, uses
        preprocessed images, by default False.
    hospital : str, optional
        Name of hospital, by default "SickKids".
    **plot_kwargs : Keyword arguments to pass into `plot_umap`
    """
    # Map hospital to more readable string
    map_hospital_str = {
        "sickkids": "SickKids",
        "stanford": "Stanford",
        "chop": "CHOP",
        "uiowa": "UIowa",
        "stanford_non_seq": "Stanford(Non-Seq)",
        "sickkids_silent_trial": "SickKids (Silent Trial)",
    }
    assert hospital in map_hospital_str
    hospital_str = map_hospital_str[hospital]

    # Filter out image files w/o labels 
    idx_unlabeled = ~pd.isnull(view_labels)
    view_labels = view_labels[idx_unlabeled]
    df_embeds_only = df_embeds_only[idx_unlabeled]
    highlight = highlight if highlight is None else highlight[idx_unlabeled]

    # Extract UMAP embeddings
    umap_embeds_views = get_umap_embeddings(df_embeds_only)

    # Label order
    label_order = get_label_order(view_labels)

    # Plot all images by view
    plot_umap(umap_embeds_views, view_labels,
              highlight=highlight,
              label_order=label_order,
              save=True,
              title=f"UMAP ({hospital_str}, colored by view)",
              filename=f"{model}/{hospital}/{model}_umap"
                       f"{'_raw' if raw else ''}(views"
                       f"{', highlighted' if highlight is not None else ''})",
              palette="tab10" if len(label_order) < 5 else "tab20",
              **plot_kwargs)


def plot_umap_by_machine(model, machine_labels, filenames, df_embeds_only,
                         raw=False,
                         hospital="SickKids",
                         **plot_kwargs):
    """
    Plots UMAP for all machine-labeled patients, coloring by machine.

    Parameters
    ----------
    model : str
        Name of model, or pretraining dataset used to pretrain model
    machine_labels : numpy.array
        Contains machine labels, corresponding to embeddings extracted. Images
        without label (null value) will be excluded
    filenames : pd.Series
        Contains file name of image whose features were extracted
    df_embeds_only : pd.DataFrame
        Extracted deep embeddings. Does not have file paths in any column.
    raw : bool, optional
        If True, loaded embeddings extracted from raw images. Otherwise, uses
        preprocessed images, by default False.
    hospital : str, optional
        Name of hospital with labels for images. One of (SickKids, Stanford,
        Both), by default "SickKids".
    **plot_kwargs : Keyword arguments to pass into `plot_umap`
    """
    assert hospital in ("SickKids", "Stanford", "Both")

    # Filter out image files w/o labels
    idx_unlabeled = ~pd.isnull(machine_labels)
    machine_labels = machine_labels[idx_unlabeled]
    filenames = filenames[idx_unlabeled]
    df_embeds_only = df_embeds_only[idx_unlabeled]

    # Extract UMAP embeddings
    umap_embeds_views = get_umap_embeddings(df_embeds_only)

    # Plot all images by view
    plot_umap(umap_embeds_views, machine_labels,
              save=True,
              title=f"UMAP ({hospital}, colored by machine)",
              filename=f"{model}/{model}_umap_{hospital.lower()}"
                       f"{'_raw' if raw else ''}(machine)",
              **plot_kwargs)


def plot_umap_for_one_patient_seq(model, view_labels, patient_visit,
                                  us_nums, df_embeds_only, color="us_nums",
                                  raw=False,
                                  **plot_kwargs):
    """
    Plots UMAP for a single patient, colored by ultrasound number.

    Parameters
    ----------
    model : str
        Name of model, or pretraining dataset used to pretrain model
    view_labels : numpy.array
        Contains view labels, corresponding to embeddings extracted. Images
        without label (null value) will be excluded
    patient_visit : pd.Series
        Contains string of (patient ID)_(visit number), which forms a unique
        sequence identifier
    us_nums : pd.Series
        Contains number in ultrasound sequence capture
    df_embeds_only : pd.DataFrame
        Extracted deep embeddings. Does not have file paths in any column.
    color : str
        Option to color by US sequence number or view label. One of "us_nums"
        or "views", by default "us_nums".
    raw : bool, optional
        If True, loaded embeddings extracted from raw images. Otherwise, uses
        preprocessed images, by default False.
    **plot_kwargs : Keyword arguments to pass into `plot_umap`
    """
    assert color in ("us_nums", "views")

    # If coloring by label, filter out unlabeled
    if color != "us_nums":
        idx_unlabeled = ~pd.isnull(view_labels)
        view_labels = view_labels[idx_unlabeled]
        patient_visit = patient_visit[idx_unlabeled]
        us_nums = us_nums[idx_unlabeled]
        df_embeds_only = df_embeds_only[idx_unlabeled]

    # Select a unique sequence (patient-visit)
    patient_selected = patient_visit.unique()[0]
    idx_patient = (patient_visit == patient_selected)
    view_labels = view_labels[idx_patient]
    us_nums = us_nums[idx_patient].reset_index(drop=True)
    df_embeds_only = df_embeds_only[idx_patient].reset_index(drop=True)
    
    # Sort data points by US number
    patient_us_nums = us_nums.sort_values()
    idx_sorted = patient_us_nums.index.tolist()
    df_embeds_only = df_embeds_only[idx_sorted]
    view_labels = view_labels[idx_sorted]

    # Extract UMAP embeddings (for 1 patient)
    umap_embeds_patient = get_umap_embeddings(df_embeds_only)

    plot_umap(umap_embeds_patient,
              patient_us_nums if color == "us_nums" else view_labels,
              line=True,
              legend=False if color == "us_nums" else True,
              s=12,
              title=f"UMAP (patient {patient_selected}, colored by US number)",
              palette="Blues" if color == "us_nums" else "tab10",
              save=True,
              filename=f"{model}/{model}_umap_single{'_raw' if raw else ''}"
                       f"({color})",
              **plot_kwargs)


def plot_umap_for_n_patient(model, patients, df_embeds_only, n=3,
                            raw=False,
                            **plot_kwargs):
    """
    Plots UMAP for images from N chosen patient, colored by patient.

    Parameters
    ----------
    model : str
        Name of model, or pretraining dataset used to pretrain model
    patients : pd.Series
        Contains all patient IDs
    df_embeds_only : pd.DataFrame
        Extracted deep embeddings. Does not have file paths in any column.
    n : int
        Number of patients to choose, by default 3
    raw : bool, optional
        If True, loaded embeddings extracted from raw images. Otherwise, uses
        preprocessed images, by default False.
    **plot_kwargs : Keyword arguments to pass into `plot_umap`
    """
    # NOTE: Extract UMAP embeddings for ALL patients
    umap_embeds_all = get_umap_embeddings(df_embeds_only)

    # Choose N patient
    patients_selected = patients.unique()[:n]
    
    # Filter UMAP embeddings for the chosen patients
    idx_patients = patients.isin(patients_selected)
    umap_embeds_patients = umap_embeds_all[idx_patients]
    patient_ids = patients[idx_patients]

    plot_umap(umap_embeds_patients, patient_ids,
              legend=True,
              title=f"UMAP (patients {patients_selected}, "
                    "colored by patient ID)",
              save=True,
              filename=f"{model}/{model}_umap{'_raw' if raw else ''}"
                       "(patient_id)",
              **plot_kwargs)


def plot_images_in_umap_clusters(model, filenames, df_embeds_only, raw=False,
                                 **plot_kwargs):
    """
    Plots images in UMAP clusters

    Parameters
    ----------
    model : str
        Name of model, or pretraining dataset used to pretrain model
    filenames : pd.Series
        Contains file name of image whose features were extracted
    df_embeds_only : pd.DataFrame
        Extracted deep embeddings. Does not have file paths in any column.
    raw : bool, optional
        If True, loaded embeddings extracted from raw images. Otherwise, uses
        preprocessed images, by default False.
    **plot_kwargs : Keyword arguments to pass into `plot_umap`
    """
    # Get UMAP embeddings
    umap_embeds_all = get_umap_embeddings(df_embeds_only)

    # Get cluster embeddings
    cluster_labels = cluster_by_density(umap_embeds_all)

    # Plot example images in each cluster 
    for cluster in np.unique(cluster_labels):
        # Filter by cluster, and sample 25 images
        idx_cluster = (cluster_labels == cluster)
        cluster_filenames = filenames[idx_cluster][:25]

        print(f"""
################################################################################
#                              Cluster {cluster}                               #
################################################################################
        """)

        # Add directory to image paths
        cluster_img_paths = [constants.DIR_IMAGES + filename \
            for filename in cluster_filenames]

        print("\n".join(cluster_img_paths))

        # Load images
        imgs = np.array([cv2.imread(path) for path in cluster_img_paths])

        # Grid plot cluster images
        gridplot_images(
            imgs,
            filename=f"{model}_cluster_{cluster}{'_raw' if raw else ''}",
            title=f"Cluster {cluster}"
            )

    # Plot UMAP with cluster labels
    plot_umap(umap_embeds_all, cluster_labels,
              label_order=sorted(np.unique(cluster_labels)),
              save=True,
              title=f"UMAP (colored by cluster label)",
              filename=f"{model}/{model}_umap{'_raw' if raw else ''}"
                       "(cluster_labels)",
              **plot_kwargs)


################################################################################
#                               Helper Functions                               #
################################################################################
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


def cluster_by_density(embeds):
    """
    Cluster embeddings by density.

    Parameters
    ----------
    embeds : pd.Series or np.array
        Extracted embeddings of the shape: (num_samples, num_features)

    Returns
    -------
    np.array
        Assigned cluster labels
    """
    clusters = DBSCAN().fit(embeds)
    return clusters.labels_


def get_label_order(labels):
    """
    Given example labels, return the label.

    Parameters
    ----------
    labels : list
        Example labels

    Returns
    -------
    list
        Order of labels, or None, if unable to find the label order
    """
    labels = set(labels)
    for label_order in (VIEW_LABEL_ORDER, SIDE_LABEL_ORDER, PLANE_LABEL_ORDER):
        if labels.issubset(set(label_order)):
            return label_order
    return None


################################################################################
#                                 Main Method                                  #
################################################################################
def main(exp_name,
         raw=False,
         segmented=False,
         reverse_mask=False,
         hospital_umap=False,
         view_umap=True,
         highlight_label_boundary=False,
         one_seq_umap=False,
         machine_umap=False,
         n_patient_umap=False,
         cluster_umap=False,
         dset=constants.DEFAULT_EVAL_DSET,
         ):
    """
    Retrieves embeddings from specified pretrained model. Then plot UMAPs.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    raw : bool, optional
        If True, loads embeddings extracted from raw images. Otherwise, uses
        preprocessed images, by default False.
    segmented : bool, optional
        If True, loads embeddings extracted from segmented images, by default
        False.
    reverse_mask : bool, optional
        If True, loads embeddings extracted from segmented images where the mask
        is reversed, by default False.
    dset : str, optional
        Dataset split or test dataset name, whose embeddings to plot, by default
        constants.DEFAULT_EVAL_DSET
    """
    # INPUT: Infer hospital from dset specified
    hospital = "sickkids" if dset in ("train", "val", "test") else dset

    # Load embeddings
    df_embeds = get_embeds(
        exp_name,
        raw=raw,
        segmented=segmented,
        reverse_mask=reverse_mask,
        dset=dset)
    # Rename old filename column
    df_embeds = df_embeds.rename(columns={"paths": "filename",
                                          "files": "filename"})

    # Extract metadata from image file paths
    df_metadata = pd.DataFrame({"filename": df_embeds["filename"]})
    df_metadata = utils.extract_data_from_filename_and_join(
        df_metadata,
        hospital=hospital,
        label_part=None,
    )

    patients = df_metadata["id"]
    patient_visits = df_metadata.apply(
        lambda row: "_".join([row["id"], row["visit"]]), axis=1)
    us_nums = df_metadata["seq_number"]
    filenames = df_metadata["filename"].map(os.path.basename).to_numpy()
    view_labels = df_metadata["label"].to_numpy()

    # Isolate UMAP embeddings (all patients)
    df_embeds_only = df_embeds.drop(columns=["filename"])

    # 0. Shared UMAP kwargs
    plot_kwargs = {"filename_suffix": f"({dset})" if dset else ""}

    # 1. Plot UMAP of all patients, colored by hospital (SickKids / Stanford)
    if hospital_umap:
        plot_umap_all_patients(
            exp_name, patients, df_embeds_only, color="hospital", raw=raw,
            **plot_kwargs)

    if one_seq_umap:
        # 2. Plot UMAP of one patient, colored by number in sequence
        plot_umap_for_one_patient_seq(
            exp_name, view_labels, patient_visits, us_nums, df_embeds_only,
            color="us_nums", raw=raw,
            **plot_kwargs)
        # 3. Plot UMAP for one unique US sequence, colored by view
        # NOTE: Only keeps images with labels in provided df_labels
        plot_umap_for_one_patient_seq(
            exp_name, view_labels, patient_visits, us_nums, df_embeds_only,
            color="views", raw=raw,
            **plot_kwargs)

    # 4. Plot UMAP of patients, colored by view
    if view_umap:
        # 4.1 Plot UMAP with view labels
        plot_umap_by_view(
            exp_name, view_labels,
            df_embeds_only=df_embeds_only,
            raw=raw, hospital=hospital,
            **plot_kwargs)

        # 4.2 Plot UMAP, highlighting label boundaries
        if highlight_label_boundary:
            highlight = utils.get_label_boundaries(df_metadata)
            plot_umap_by_view(
                exp_name, view_labels,
                df_embeds_only=df_embeds_only,
                highlight=highlight,
                raw=raw, hospital=hospital,
                **plot_kwargs)

    # 5. Plot UMAP of patients, colored by machine label
    if machine_umap:
        # Get machine labels (if any) for all extracted images
        machine_labels = utils.get_machine_for_filenames(filenames)
        # Plot UMAP
        plot_umap_by_machine(exp_name, machine_labels, filenames, df_embeds_only,
                             raw=raw, hospital="SickKids",
                             **plot_kwargs)

    # 6. Plot UMAP for N patients, colored by patient ID
    if n_patient_umap:
        plot_umap_for_n_patient(exp_name, patients, df_embeds_only, n=3, raw=raw,
                                **plot_kwargs)

    # 7. Plot example images from UMAP clusters
    if cluster_umap:
        plot_images_in_umap_clusters(exp_name, filenames, df_embeds_only,
                                     raw=False,
                                     **plot_kwargs)

    # Close all figures
    plt.close("all")


if __name__ == '__main__':
    for model in ("moco",):      # must be in constants.MODELS
        main(model, raw=False, segmented=False, reverse_mask=False)
