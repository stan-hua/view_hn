"""
utils.py

Description: Contains general helper functions for data visualization.
"""

# Standard libraries
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


def gridplot_images(imgs, filename, save_dir, title=None):
    """
    Plot example images on a grid plot

    Parameters
    ----------
    example_imgs : np.array
        Images to visualize
    filename : str
        Path to save figure to
    save_dir : str
        Path to directory to save images
    title : str, optional
        Plot title, by default None
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

    for ax, img_arr in zip(grid, imgs[:num_imgs]):
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

    # Create subdirectories, if not exists
    save_path = os.path.join(save_dir, filename)
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save image
    plt.savefig(save_path)


def gridplot_images_from_paths(paths, filename, save_dir, title=None):
    """
    Create grid plot from provided list of image paths

    Parameters
    ----------
    paths : list
        List of full paths to images
    filename : str
        Path to save figure to
    save_dir : str
        Path to directory to save images
    title : str
        Plot title, by default None
    """
    imgs = np.stack([cv2.imread(path) for path in paths])
    gridplot_images(imgs, filename, save_dir, title=title)


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
