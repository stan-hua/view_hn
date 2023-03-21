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

    # Create grid plot
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(num_imgs_sqrt, num_imgs_sqrt),
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
