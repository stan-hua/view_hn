"""
utils.py

Description: Contains general helper functions for data visualization.
"""

# Standard libraries
import math
import os

# Non-standard libraries
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from tabulate import tabulate


################################################################################
#                                  Constants                                   #
################################################################################
# Constants used in calibration plots
CALIB_COUNT = 'count'
CALIB_CONF = 'conf'
CALIB_ACC = 'acc'
CALIB_BIN_ACC = 'bin_acc'
CALIB_BIN_CONF = 'bin_conf'


################################################################################
#                               Helper Functions                               #
################################################################################
def set_theme(theme="light"):
    """
    Set matplotlib theme

    Parameters
    ----------
    theme : str, optional
        One of "light" or "dark", by default "light"
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc('font', family='serif')
    if theme == "light":
        sns.set_style("white")
        plt.style.use('seaborn-white')
    else:
        sns.set_style("dark")
        plt.style.use('dark_background')


def print_table(df, show_cols=True, show_index=True):
    """
    Prints table to stdout in a pretty format.

    Parameters
    ----------
    df : pandas.DataFrame
        A table
    show_cols : bool
        If True, prints column names, by default True.
    show_index : bool
        If True, prints row index, by default True.
    """
    print(tabulate(df, tablefmt="psql",
                   headers="keys" if show_cols else None,
                   showindex=show_index))


def gridplot_images(imgs, labels=None, title=None, filename=None, save_dir=None):
    """
    Plot example images on a grid plot

    Parameters
    ----------
    imgs : np.array
        Images to visualize
    labels : np.array, optional
        Labels corresponding to images
    title : str, optional
        Plot title, by default None
    filename : str, optional
        Path to save figure to
    save_dir : str, optional
        Path to directory to save images

    Returns
    -------
    plt.Figure
        Matplotlib Figure
    """
    # Determine number of images to plot
    num_imgs_sqrt = int(np.sqrt(len(imgs)))
    num_imgs = num_imgs_sqrt ** 2
    nrows = ncols = num_imgs_sqrt

    # If very few images, print all images in 1 row
    if num_imgs < len(imgs):
        nrows = len(imgs)
        ncols = 1
        num_imgs = len(imgs)

    # Create grid plot
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.01,      # padding between axes
    )

    for idx, (ax, img_arr) in enumerate(zip(grid, imgs[:num_imgs])):
        # Set label as title
        if labels:
            ax.set_title(labels[idx])

        # Set x and y axis to be invisible
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        # If first dimension is the channels, move to end
        if img_arr.shape[0] in (1, 3):
            img_arr = np.moveaxis(img_arr, 0, -1)

        # Add image to grid plot
        vmin, vmax = (0, 255) if img_arr.max() > 1 else (None, None)
        ax.imshow(img_arr, cmap='gray', vmin=vmin, vmax=vmax)

    # Set title
    fig.suptitle(title)

    # Make plot have tight layout
    plt.tight_layout()

    # Early return, if no path is provided
    if not filename:
        return fig

    # Create subdirectories, if not exists
    save_path = os.path.join(save_dir, filename)
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save image
    plt.savefig(save_path)

    return fig


def gridplot_images_from_paths(paths, title=None, filename=None, save_dir=None):
    """
    Create grid plot from provided list of image paths

    Parameters
    ----------
    paths : list
        List of full paths to images
    title : str
        Plot title, by default None
    filename : str
        Path to save figure to
    save_dir : str
        Path to directory to save images

    Returns
    -------
    plt.Figure
        Matplotlib Figure
    """
    imgs = np.stack([cv2.imread(path) for path in paths])
    ax = gridplot_images(imgs, filename=filename, save_dir=save_dir, title=title)
    return ax


def grouped_barplot(data, x, y, hue, yerr_low, yerr_high, legend=False,
                    xlabel=None, ylabel=None, ax=None,
                    **plot_kwargs):
    """
    Create grouped bar plot with custom confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Data
    x : str
        Name of primary column to group by
    y : str
        Name of column with bar values
    hue : str
        Name of secondary column to group by
    yerr_low : str
        Name of column to subtract y from to create LOWER bound on confidence
        interval
    yerr_high : str
        Name of column to subtract y from to create UPPER bound on confidence
        interval
    legend : bool, optional
        If True, add legend to figure, by default False.
    ax : matplotlib.pyplot.Axis, optional
        If provided, draw plot into this Axis instead of creating a new Axis, by
        default None.
    **plot_kwargs : keyword arguments to pass into `matplotlib.pyplot.bar`

    Returns
    -------
    matplotlib.pyplot.Axis.axis
        Grouped bar plot with custom confidence intervals
    """
    # Get unique values for x and hue
    x_unique = data[x].unique()
    xticks = np.arange(len(x_unique))
    hue_unique = data[hue].unique()

    # Bar-specific constants
    offsets = np.arange(len(hue_unique)) - np.arange(len(hue_unique)).mean()
    offsets /= len(hue_unique) + 1.
    width = np.diff(offsets).mean()

    # Create figure
    if ax is None:
        _, ax = plt.subplots()

    # Create bar plot per hue group
    for i, hue_group in enumerate(hue_unique):
        df_group = data[data[hue] == hue_group]
        ax.bar(
            x=xticks+offsets[i],
            height=df_group[y].values,
            width=width,
            label="{} {}".format(hue, hue_group),
            yerr=abs(df_group[[yerr_low, yerr_high]].T.to_numpy()),
            **plot_kwargs)

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set x-axis ticks
    ax.set_xticks(xticks, x_unique)

    if legend:
        ax.legend()

    return ax


################################################################################
#                     Calibration-Related Helper Functions                     #
################################################################################
# Adapted from https://github.com/torrvision/focal_calibration/blob/main/Metrics/plots.py
def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][CALIB_COUNT] = 0
        bin_dict[i][CALIB_CONF] = 0
        bin_dict[i][CALIB_ACC] = 0
        bin_dict[i][CALIB_BIN_ACC] = 0
        bin_dict[i][CALIB_BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][CALIB_COUNT] = bin_dict[binn][CALIB_COUNT] + 1
        bin_dict[binn][CALIB_CONF] = bin_dict[binn][CALIB_CONF] + confidence
        bin_dict[binn][CALIB_ACC] = bin_dict[binn][CALIB_ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][CALIB_COUNT] == 0):
            bin_dict[binn][CALIB_BIN_ACC] = 0
            bin_dict[binn][CALIB_BIN_CONF] = 0
        else:
            bin_dict[binn][CALIB_BIN_ACC] = float(
                bin_dict[binn][CALIB_ACC]) / bin_dict[binn][CALIB_COUNT]
            bin_dict[binn][CALIB_BIN_CONF] = bin_dict[binn][CALIB_CONF] / \
                float(bin_dict[binn][CALIB_COUNT])
    return bin_dict


def plot_reliability_diagram(confs, preds, labels, num_bins=10):
    """
    Plot Reliability Diagram.

    Note
    ----
    Used to assess if model's output probabilities accurately represent the
    probability of the outcome in the calibration set (e.g., Does a model
    predict a 70% prob. for an event that occurs 70% of the time?)

    Parameters
    ----------
    confs : list or array-like
        Probability of each prediction
    preds : list or array-like
        List of predicted labels
    labels : list or array-like
        List of true labels
    num_bins : int, optional
        Number of bins across probability range (0 to 1), by default 10
    """
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][CALIB_BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align='edge', width=0.05, color='pink', label='Expected')
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    plt.show()


def plot_confidence_histogram(confs, preds, labels, num_bins=10):
    """
    Plot Confidence Histogram.

    Note
    ----
    Used to see the percentage of calibration samples in each confidence bin.

    Parameters
    ----------
    confs : list or array-like
        Probability of each prediction
    preds : list or array-like
        List of predicted labels
    labels : list or array-like
        List of true labels
    num_bins : int, optional
        Number of bins across probability range (0 to 1), by default 10
    """
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][CALIB_COUNT] / float(num_samples)) * 100
        y.append(n)
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Percentage samples')
    plt.ylabel('Percentage of samples')
    plt.xlabel('Confidence')
    plt.show()
