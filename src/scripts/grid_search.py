"""
grid_search.py

Description: Used to perform hyperparameter search via RandomizedGridSearch
"""

# Standard libraries
import logging
import os
import random
import shutil
import subprocess
from datetime import datetime
from glob import glob
from pathlib import Path

# Non-standard libraries
import pandas as pd
import yaml

# Custom libraries
from src.data import constants


# Configure logging
logging.basicConfig(level=logging.DEBUG)

################################################################################
#                                  Constants                                   #
################################################################################ 
# Logger
LOGGER = logging.getLogger(__name__)

# Results directory containing model directories
RESULTS_DIR = constants.DIR_RESULTS

# Name of model (appended to folder name)
MODEL_NAME = "relative_side"

# Metric to optimize 
METRIC = "loss"

# If True, discards all model weights that aren't the best model 
KEEP_BEST_WEIGHTS = False

# Parameters to test
SEARCH_SPACE = {
        'lr': [1e-3, 1e-4, 1e-5],
        # 'adam': [True, False],
        'momentum': [0.8, 0.9],
        'weight_decay': [5e-3, 5e-4],
        "grad_clip_norm": [None, 1., 2.],
        'precision': 16, 
        "pin_memory": True,
        "batch_size": [1, 8, 16],
        "train": True,
        "test": True,
        "train_val_split": 0.75,
        "train_test_split": 0.75,
        "full_seq": True,
        "relative_side": True,
        "n_lstm_layers": [1, 2, 3],
        "hidden_dim": [256, 512, 1024],
        "bidirectional": False,
    }

# Example command string
COMMAND = "python -m src.drivers.model_training"


################################################################################
#                               GridSearch Class                               #
################################################################################
class GridSearch:
    """
    Implementation of RandomizedGridSearchCV.
    """

    def __init__(self, model_name, timestamp, grid_search_dir, metric='loss'):
        self.model_name = model_name
        self.timestamp = timestamp
        self.grid_search_dir = grid_search_dir
        self.metric = metric
        self.tested_combinations = self.get_tested_hyperparams()


    def sample_hyperparams(self, possible_hyperparams):
        """
        Randomly choose hyperparameter combinations that is unused.

        Returns
        -------
        dict
            Mapping of argument to hyperparameter value.
        """
        for _ in range(1000):
            sampled_params = {}
            for u in possible_hyperparams.keys():
                v = possible_hyperparams[u]
                if isinstance(v, list):
                    v = random.choice(v)
                sampled_params[u] = v

            if self.is_hyperparam_used(sampled_params):
                continue

            row = pd.DataFrame(sampled_params, index=[0])
            self.tested_combinations = pd.concat(
                [self.tested_combinations, row], ignore_index=True)

            return sampled_params
        return None


    def is_hyperparam_used(self, sample_hyperparams):
        """
        Checks if sampled hyperparams have already been tested

        Returns
        -------
        bool
            If True, hyperparameters have already been tested. Otherwise, False.
        """
        # Get intersecting columns
        search_columns = set(list(sample_hyperparams.keys()))
        available_columns = set(self.tested_combinations.columns.tolist())
        cols = list(search_columns.intersection(available_columns))

        return [sample_hyperparams[c] for c in cols] in self.tested_combinations[cols].values.tolist()


    def get_tested_hyperparams(self):
        """
        Gets all tested hyperparameter combinations so far.

        Returns
        -------
        pandas.DataFrame
            Contains hyperparameters on each row
        """
        result_directories = self.get_training_dirs()

        df_accum = pd.DataFrame()
        for result_dir in result_directories:
            _hyperparams = get_hyperparams(result_dir)
            row = pd.DataFrame([_hyperparams], index=[0])
            df_accum = pd.concat([df_accum, row])

        return df_accum


    def get_training_dirs(self):
        """
        Find all directories made during hyperparameter search. Move to grid
        search directory if not already done.

        Note
        ----
        All folders containing the model name will be moved into the grid search
        directory.

        Returns
        -------
        list
            Absolute paths to each directory in the grid search directory
        """
        temp_folders = []
        for result_dir in glob(f"{RESULTS_DIR}/*"):
            if ("grid_search" not in result_dir) and \
                    (self.model_name in result_dir):
                temp_folders.append(result_dir)

        # Move folders to grid search directory
        for result_dir in temp_folders:
            shutil.move(result_dir, self.grid_search_dir)

        # Get list of model directory names
        model_dirs = []
        for result_dir in glob(f"{self.grid_search_dir}/*"):
            if "." in result_dir or "csv" in result_dir or "json" in result_dir:
                continue
            model_dirs.append(result_dir)

        return model_dirs


    def find_best_models(self):
        """
        Get row with the best validation metric. If cross-fold validation done,
        average metrics over folds. 

        Returns
        -------
        pandas.dataframe
            Containing the best validation set results, where each row
            corresponds to tested model hyperparameters
        """
        result_directories = self.get_training_dirs()

        df_accum = pd.DataFrame()
        for result_dir in result_directories:
            df = aggregate_fold_histories(result_dir)

            best_score = min(df[f"val_{self.metric}"]) if self.metric == 'loss' else max(df[f"val_{self.metric}"])
            df_best_epoch = df[df[f"val_{self.metric}"] == best_score]

            # Add in parameters
            params = get_hyperparams(result_dir)
            for key in params:
                if key == "insert_where" and datetime.strptime(result_dir.split('_')[-2], "%Y-%m-%d").day < 8:
                    params[key] = 0
                df_best_epoch[key] = params[key]

            df_best_epoch["dir"] = result_dir
            df_accum = pd.concat([df_accum, df_best_epoch])

        return df_accum


    def perform_grid_search(self, search_space, n=None):
        """
        Performs randomized grid search on given specified parameter lists.
        
        Note
        ----
        Tested parameter combinations are saved.

        Parameters
        ----------
        search_space: dict
            Keys are argument names for training script and values are potential
            values to search.
        n : int
            Number of trials to run
        """
        while n > 0:
            n -= 1

            # Randomly sample hyperparameters
            params = self.sample_hyperparams(search_space)

            if params is None:
                return

            LOGGER.info(params)

            # Accumulate commands on string
            command_str = COMMAND
            for u in params.keys():
                v = params[u]

                if v is None:
                    continue
                elif isinstance(v, bool):
                    if v:
                        command_str += f" --{u}"
                else:
                    command_str += f" --{u} {v}"

            LOGGER.info(command_str) 
 
            # Add current time as experiment name
            exp_timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M")
            exp_name = f"{MODEL_NAME}({exp_timestamp})"
            command_str += f" --exp_name {exp_name}"

            # Run model training
            subprocess.run(command_str, shell=True)


    def save_grid_search_results(self):
        """
        Saves the best hyperparameters to a JSON file, and saves a summary of
        model metrics in a CSV file.
        """
        df = self.find_best_models()

        if len(df) == 0:
            return

        # Optimize best metric
        if self.metric == 'loss':
            best_score = min(df[f"val_{self.metric}"])
        else:
            best_score = max(df[f"val_{self.metric}"])

        # Save best parameters
        best_model = df[df[f"val_{self.metric}"] == best_score].iloc[0]
        best_model.to_json(f"{self.grid_search_dir}/best_parameters.json")

        # Save summary of model results
        df = df.sort_values(by=[f"val_{self.metric}"])
        df.to_csv(f"{self.grid_search_dir}/grid_search({self.timestamp}).csv",
                  index=False)


################################################################################
#                               Helper Functions                               #
################################################################################
def aggregate_fold_histories(result_dir: str):
    """
    Averages the metric of interest for each epoch, across folds.

    Parameters
    ----------
    result_dir : str
        Path to model training directory

    Returns
    -------
    pandas.DataFrame
        Contains the averaged values per epoch
    """
    histories = glob(f"{result_dir}/*/history.csv")

    df = pd.DataFrame()
    for history in histories:
        fold = history.split(os.sep)[-3][-1]
        df_fold = pd.read_csv(history)
        df_fold['fold'] = fold
        df = pd.concat([df, df_fold], ignore_index=True)

    df = df.groupby(by=["epoch"]).mean().reset_index()

    for col in df.columns.tolist():
        if "_" not in col or "loss" in col:
            continue
        if ('train' not in col) and ('test' not in col) and ('val' not in col):
            continue

        if all(df[col] < 1):
            df[col] = (df[col] * 100).round(decimals=2)

    return df


def get_hyperparams(result_dir: str):
    """
    Get hyperparameters from a training directory.

    Returns
    -------
    dict
        Contains hyperparameters from run
    """
    hparam_paths = list(Path(result_dir).rglob("hparams.yaml"))

    if not hparam_paths:
        return None

    with open(hparam_paths[0], "r") as stream:
        hparams = yaml.load(stream, yaml.Loader)

    return hparams


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d")
    grid_search_dir = f"{RESULTS_DIR}/{MODEL_NAME}_grid_search({timestamp})/"

    if not os.path.exists(grid_search_dir):
        os.mkdir(grid_search_dir)

    gridSearch = GridSearch(MODEL_NAME, timestamp, grid_search_dir,
                            metric=METRIC)
    gridSearch.perform_grid_search(SEARCH_SPACE, n=20)
    gridSearch.save_grid_search_results()
