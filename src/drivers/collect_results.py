"""
collect_results.py

Description: Used to get metric results for eval. models of multiple experiments
             from the `inference` directory, and place them in an ordered table.
"""

# Standard libraries
import argparse
import logging
import os
from collections import OrderedDict

# Non-standard libraries
import pandas as pd

# Custom libraries
from src.data import constants


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
    if dset in constants.HOSPITAL_MISSING_BLADDER:
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


def get_eval_metrics(exp_name, dsets=DSETS, tasks=TASKS):
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

            # Get metric of interest
            metric_str = df_metrics.loc["Overall Accuracy", "All"]

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
    future_n = len(df_metrics.columns) + len(dsets) - 1

    col_idx = 0
    curr_dset = "NOT_EXISTS"
    while col_idx < future_n:
        columns = df_metrics.columns.tolist()
        # If same dset, skip
        if columns[col_idx].startswith(curr_dset) or col_idx == 0:
            col_idx += 1
            continue
        # If the last one was `exp_name`, ignore
        if col_idx and columns[col_idx-1] == "exp_name":
            col_idx += 1
            curr_dset = columns[col_idx].split("_")[0]
            continue

        # Otherwise, add column spacer
        spacer_col = pd.DataFrame({"": [""] * len(df_metrics)})
        df_metrics = pd.concat([
            df_metrics.iloc[:, 0:col_idx],
            spacer_col,
            df_metrics.iloc[:, col_idx:]], axis=1)

        # New dataset
        curr_dset = columns[col_idx].split("_")[0]
        # NOTE: Need to skip added spacer column
        col_idx += 2

    return df_metrics


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
        "save_dir": "Path to directory to save created file in",
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
                tasks=args.task)

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


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Format results
    main(ARGS)
