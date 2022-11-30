"""
utils.py

Description: Contains general helper functions for data visualization.
"""

# Non-standard libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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
