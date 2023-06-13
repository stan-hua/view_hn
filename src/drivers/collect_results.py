"""
collect_results.py

Description: Used to get metric results for eval. models of multiple experiments
             from the `inference` directory, and place them in an ordered table.
"""

# Standard libraries
import argparse
import logging
import os
import re
from collections import OrderedDict

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Custom libraries
from src.data import constants
from src.data_viz import utils as viz_utils


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

# Default evaluation sets to retrieve
DSETS = ["val", "test", "stanford", "uiowa", "chop"]

# Default tasks to retrieve for
TASKS = ["side", "plane"]

# Default filename to save CSV as
DEFAULT_FNAME = "multiple_exp_summary.csv"


################################################################################
#                               Helper Functions                               #
################################################################################
def create_eval_model_dir(exp_name, task, dset):
    """
    Given the experiment name and evaluation task, create the path to the
    expected directory.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    task : str
        Task (side/plane/None) to find models for
    dset : str
        Dataset split and/or external dataset evaluated on

    Returns
    -------
    str
        Full path to existing evaluation model directory
    """
    # Raise RuntimeError if `exp_name` does not contain (TASK)
    if "(TASK)" not in exp_name:
        raise RuntimeError("Experiment name regex provided (%s) does NOT "
                           "contain expected (TASK) in name to fill-in with "
                           "task" % (exp_name,))

    # Fill-in with actual task name
    exp_name = exp_name.replace("(TASK)", task)

    # If UIowa or CHOP, only look at predictions masking bladder
    if dset in constants.DSETS_MISSING_BLADDER:
        exp_name += "__mask_bladder"

    # Prepend `inference` directory path
    path = os.path.join(constants.DIR_INFERENCE, exp_name)

    # Raise RuntimeError if created path does not exist
    if not os.path.exists(path):
        raise RuntimeError("Cannot find evaluation model inference directory! "
                           "Failed eval. model: %s" % (exp_name,))

    return path


def prettify_metric(metric_str):
    """
    Given metric string of form: "0.6921 (0.681, 0.791)", return the following
    string: "69.21 [68.1, 79.1]".

    Parameters
    ----------
    metric_str : str
        Metric string with confidence intervals

    Returns
    -------
    str
        Prettified metric string
    """
    # Replace parenthesis with brackets
    metric_str = metric_str.replace("(", "[").replace(")", "]")

    # Early return, if metric doesn't contain confidence interval
    if "[" not in metric_str and "]" not in metric_str:
        return metric_str

    # Extract values
    # 1. Get the mean value
    mean, rest = metric_str.split(" [")
    # 2. Get the lower bound
    lower, rest = rest.split(", ")
    # 3. Get upper bound
    upper = rest.replace("]", "")
    vals = [mean, lower, upper]

    # Convert to float values
    vals = map(float, vals)

    # Only multiply by 100 if it's less than 1
    cond_multiply = lambda x: (x * 100) if x < 1 else x
    vals = map(cond_multiply, vals)

    # Reformat as string
    mean, lower, upper = vals
    new_metric_str = f"{mean:.2f} [{lower:.2f}, {upper:.2f}]"

    return new_metric_str


def get_eval_metrics(exp_name, dsets=DSETS, tasks=TASKS, ignore_bladder=False):
    """
    Get accuracy metrics for task-specific evaluation model, for the specified
    experiment and evaluation datasets.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dsets : list, optional
        Dataset splits and/or external datasets evaluated on, by default DSETS
    tasks : list, optional
        List of tasks (side/plane/None) to find models for, by default TASKS
    ignore_bladder : bool, optional
        If True, specify to ignore the images labeled as Bladder (if possible)

    Returns
    -------
    pd.DataFrame
        Row corresponding to evaluation datasets for 1 experiment
    """
    # Accumulate metrics for each dset
    metrics_per_dset = []
    for dset in dsets:
        dset_metrics = OrderedDict()
        for task in tasks:
            # Get path to model eval. directory
            eval_model_dir = create_eval_model_dir(exp_name, task, dset)

            # Load metrics file for dset
            metrics_path = os.path.join(eval_model_dir, f"{dset}_metrics.csv")
            df_metrics = pd.read_csv(metrics_path, index_col=0)

            # Specify column to get from
            # NOTE: Column specifies subset of images to get metrics from
            col = "All"
            if ignore_bladder and "Without Bladder" in df_metrics.columns:
                col = "Without Bladder"

            # Get metric of interest
            metric_str = df_metrics.loc["Overall Accuracy", col]

            # Clean up metric
            metric_str = prettify_metric(metric_str)

            # Store data
            dset_metrics[f"{dset}_{task}"] = metric_str
        metrics_per_dset.append(pd.Series(dset_metrics))

    # Add experiment name
    stored_exp_name = exp_name if exp_name != "|" else ""
    metrics_per_dset.insert(0, pd.Series({"exp_name": stored_exp_name}))

    # Concatenate results
    metric_ser = pd.concat(metrics_per_dset)

    # Transpose to create dataframe row
    metric_row = metric_ser.to_frame().T

    return metric_row


def add_space_between_dsets(df_metrics, dsets=DSETS):
    """
    Add column of space between metrics from different dsets

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Each row contains evaluation metrics for multiple datasets for the same
        experiment
    dsets : list, optional
        Evaluation datasets stored

    Returns
    -------
    pd.DataFrame
        Metrics table with added space between columns of different datasets
    """
    # Create copy to avoid overwriting
    df_metrics = df_metrics.copy()

    # Expected number of columns after adding spacers
    future_n = len(df_metrics.columns) + (len(dsets) - 1)

    col_idx = 0
    curr_dset = "NOT_EXISTS"
    while col_idx < future_n:
        columns = df_metrics.columns.tolist()
        # If same dset, skip
        if "_".join(columns[col_idx].split("_")[:-1]) == curr_dset \
                or col_idx == 0:
            col_idx += 1
            continue
        # If last is the `exp_name`, skip
        if columns[col_idx-1] == "exp_name":
            curr_dset = "_".join(columns[col_idx].split("_")[:-1])
            col_idx += 1
            continue

        # Otherwise, add column spacer
        spacer_col = pd.DataFrame({"": [""] * len(df_metrics)})
        df_metrics = pd.concat([
            df_metrics.iloc[:, 0:col_idx],
            spacer_col,
            df_metrics.iloc[:, col_idx:]], axis=1)

        # New dataset
        curr_dset = "_".join(columns[col_idx].split("_")[:-1])
        # NOTE: Need to skip added spacer column
        col_idx += 2

    return df_metrics


def parse_prettified_metric_col(metric_col):
    """
    Given a (prettified) metric column, extract the mean metric value, lower CI
    and upper CI values.

    Parameters
    ----------
    metric_col : pd.Series, np.array, list
        Metric string column of the form: "MEAN [LOWER, UPPER]"

    Returns
    -------
    tuple of (np.array, np.array, np.array)
        List of MEAN, list of LOWER CI and list of UPPER CI values
    """
    # Convert to list
    if isinstance(metric_col, pd.Series):
        metric_col = metric_col.tolist()
    vals = list(metric_col)

    # Extract values
    mean, lower, upper = [], [], []
    for val in vals:
        match = re.match(r"(.*) \[(.*), (.*)\]", val)
        # Raise error, if invalid formatted string found
        if match is None:
            raise RuntimeError("Invalid formatted string in metric column!")

        mean.append(match.group(1))
        lower.append(match.group(2))
        upper.append(match.group(3))

    # Convert to numeric arrays
    mean, lower, upper = np.array(mean), np.array(lower), np.array(upper)

    mean = mean.astype(float)
    lower = lower.astype(float)
    upper = upper.astype(float)

    return mean, lower, upper


################################################################################
#                        Analysis on Collected Metrics                         #
################################################################################
def barplot_all_models(prettified_metrics, ylabel=None, ax=None):
    """
    Create a barplot of metrics with all models for ONE hospital.

    Parameters
    ----------
    prettified_metrics : pd.Series
        Each row is a distinct model with ONE string metric column
        evaluated on one hospital for a particular label type. Indexed by model
        name.
    ylabel : str, optional
        Label of y-axis, by default None.
    ax : matplotlib.pyplot.Axis, optional
        If provided, draw plot into this Axis instead of creating a new Axis, by
        default None.

    Returns
    -------
    matplotlib.pyplot.Axis.axis
        Grouped bar plot with custom confidence intervals
    """
    # Raise error, if Series is not provided
    if not isinstance(prettified_metrics, pd.Series):
        raise RuntimeError("Invalid input! Please provide Series to function..")

    # Parse out mean, lower and upper CI values
    mean, lower, upper = parse_prettified_metric_col(prettified_metrics)

    # Place back into table format
    df = pd.DataFrame({
        "Accuracy": mean,
        "Accuracy_5": lower,
        "Accuracy_95": upper,
    })

    # Convert to decimals
    df[:] *= 0.01

    # Calculate delta for CI
    df["Accuracy_5_delta"] = df["Accuracy_5"] - df["Accuracy"]
    df["Accuracy_95_delta"] = df["Accuracy_95"] - df["Accuracy"]

    # Reindex by model name
    df["Model"] = prettified_metrics.index

    # Create figure, if not provided
    if ax is None:
        _, ax = plt.subplots()

    # Create bar plot with CI
    ax.bar(
        data=df,
        x="Model",
        height="Accuracy",
        yerr=abs(df[["Accuracy_5_delta", "Accuracy_95_delta"]].to_numpy().T),
        alpha=0.8,
        capsize=5,
    )

    # Set limits
    ax.set_ylim(0.25, 1)

    # Add grid background
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", alpha=0.6)

    # Set y-axis label
    if ylabel:
        ax.set_ylabel(ylabel, rotation=25)

    return ax


def plot_all_models_on_dsets(df_prettified_metrics, task="plane",
                             save=False,
                             save_dir=None,
                             fname="barplot.png"):
    """
    For each eval. dataset, create a barplot of performance metrics for all
    models.

    Parameters
    ----------
    df_prettified_metrics : pd.Series
        Each row is a distinct model with a string metric column for each
        dataset evaluated on.
    task : str, optional
        Prediction task (side/plane). Corresponding metric columns end with
        "_TASK", by default "plane".
    save : bool, optional
        If True, save plot to file, by default False.
    save_dir : str, optional
        Directory to save plot in, working directory by default.
    fname : str, optional
        Filename to save plot as, by default 'barplot.png'

    Returns
    -------
    matplotlib.pyplot.Axis.axis
        Grouped bar plot with custom confidence intervals
    """
    assert task in ("plane", "side", None)

    # Create copy and set index to experiment name
    # NOTE: Treated as model name
    df_prettified_metrics = df_prettified_metrics.copy()
    df_prettified_metrics = df_prettified_metrics.set_index("exp_name")

    # Extract metric columns and eval. dataset names
    metric_cols = [col for col in df_prettified_metrics.columns
                   if f"_{task}" in col]
    dsets = [re.match(rf"(.*)_{task}", col).group(1) for col in metric_cols]

    # Filter for metric columns and non-NA rows
    df_prettified_metrics = df_prettified_metrics[metric_cols]
    df_prettified_metrics = df_prettified_metrics.replace(r'^\s*$', np.nan,
                                                          regex=True)
    df_prettified_metrics = df_prettified_metrics.dropna()

    # Create plots
    fig, axs = plt.subplots(
        nrows=len(metric_cols), ncols=1,
        sharex=True, sharey=True,
        figsize=(5, 4))

    # Create bar plot for each eval. dataset
    for i, metric_col in enumerate(metric_cols):
        # NOTE: Make dataset name the title for each plot
        dset = re.match(rf"(.*)_{task}", metric_col).group(1)
        barplot_all_models(
            df_prettified_metrics[metric_col],
            ylabel=dset,
            ax=axs[i],
        )

    # Configure space between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Rotate x and y labels
    plt.xticks(rotation=90)

    # Figure x-axis and y-axis labels
    fig.supxlabel("Model")
    fig.supylabel("Model Accuracies for Each Dataset")

    fig.tight_layout()

    # Save plot
    if save:
        path = os.path.join(save_dir, fname) if save_dir else fname
        plt.savefig(path)


################################################################################
#                                Main Functions                                #
################################################################################
def init(parser):
    """
    Initialize ArgumentParser object.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser object
    """
    # Descriptions for each argument
    arg_help = {
        "exp_name": "Experiment name to find results for, with location to "
                    "replace task. Example: "
                    "'supervised_linear_lstm_(TASK)__from_imagenet'"
                    "Providing '|' symbolizes row space.",
        "dset": "List of dataset split or test dataset name to evaluate",
        "task": "Tasks whose evaluation models to look for (side/plane/None)",
        "ignore_bladder": "If True, get metrics computed only on non-Bladder "
                          "images.",

        # Plotting
        "barplot": "If True, create bar plot with all results",
        "plotname": "Name of plot, by default 'barplot.png'",

        # Saving
        "save_dir": "Path to directory to save created file/s in",
        "fname": "Filename of CSV to save gathered results to",
    }

    # Add arguments
    parser.add_argument("--exp_name", nargs="+", required=True,
                        help=arg_help["exp_name"])
    parser.add_argument("--dset", nargs="+",
                        default=DSETS,
                        help=arg_help["dset"])
    parser.add_argument("--task", nargs="+",
                        default=TASKS,
                        help=arg_help["task"])
    parser.add_argument("--ignore_bladder",
                        default=False,
                        action="store_true",
                        help=arg_help["ignore_bladder"])

    parser.add_argument("--barplot", action="store_true",
                        help=arg_help["barplot"])
    parser.add_argument("--plotname", default="barplot.png",
                        help=arg_help["plotname"])

    parser.add_argument("--save_dir", type=str,
                        default=constants.DIR_INFERENCE,
                        help=arg_help["save_dir"])
    parser.add_argument("--fname", type=str,
                        default=DEFAULT_FNAME,
                        help=arg_help["fname"])


def main(args):
    """
    Perform the following:
        (1) Get results of evaluation models for each experiment, and
        (2) Store them in a CSV file

    Parameters
    ----------
    args : argparser.Namespace
        Parsed arguments
    """
    # Get evaluation metrics for each experiment
    exp_names = args.exp_name
    metric_rows = []
    for exp_name in list(exp_names):
        # Check if row space is desired
        if exp_name == "|":
            metric_row = pd.DataFrame({"exp_name": [""]})
        else:
            metric_row = get_eval_metrics(
                exp_name,
                dsets=args.dset,
                tasks=args.task,
                ignore_bladder=args.ignore_bladder)

        # Accumulate
        metric_rows.append(metric_row)

    # Concatenate results
    df_metrics = pd.concat(metric_rows, ignore_index=True)

    # Add spacer between dset metrics
    df_metrics = add_space_between_dsets(df_metrics, args.dset)

    # Fill missing with empty string
    df_metrics = df_metrics.fillna("")

    # Save to file path
    save_path = os.path.join(args.save_dir, args.fname)
    df_metrics.to_csv(save_path, index=False)

    # Create plot
    if args.barplot:
        plot_all_models_on_dsets(
            df_metrics, save=True, save_dir=args.save_dir,
            fname=args.plotname)


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Format results
    main(ARGS)
