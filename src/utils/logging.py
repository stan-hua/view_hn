"""
logging.py

Description: Wrapper over PyTorch Lightning's CSVLogger to output a simpler CSV
             file (history.csv) after training.
"""

# Standard libraries
import os

# Non-standard libraries
import pandas as pd
from comet_ml import ExistingExperiment
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import rank_zero_only


################################################################################
#                              Custom CSV Logger                               #
################################################################################
class FriendlyCSVLogger(CSVLogger):
    @rank_zero_only
    def finalize(self, status=None) -> None:
        metrics_path = os.path.join(self.experiment.metrics_file_path)

        def collapse_epoch(df):
            """
            Collapse epoch results to one row

            Parameters
            ----------
            df : pandas.DataFrame
                Results from the same epoch, across rows with null values

            Returns
            -------
            pandas.DataFrame
                Epoch results in one row
            """
            # Flatten structure
            all_values = pd.melt(df).dropna().drop_duplicates(
                subset="variable", keep="last")
            # Reorganize back to row
            row = all_values.set_index("variable").T
            # Remove extra index names added
            row = row.reset_index(drop=True)
            row.columns.name = None
            return row

        if not os.path.exists(metrics_path):
            return

        df = pd.read_csv(metrics_path)
        df = df.drop(columns=["step"])
        df = df.groupby(by=['epoch'], as_index=False).apply(collapse_epoch)

        for col in df.columns.tolist():
            if "_" not in col or "loss" in col:
                continue
            df[col] = (df[col] * 100).round(decimals=2)

        df.to_csv(os.path.join(self.log_dir, "history.csv"), index=False)


################################################################################
#                               Helper Functions                               #
################################################################################
def load_comet_logger(exp_key):
    """
    Load Comet ML logger for existing experiment.

    Parameters
    ----------
    exp_key : str
        Experiment key for existing experiment

    Returns
    -------
    comet_ml.ExistingExperiment
        Can be used for logging
    """
    assert "COMET_API_KEY" in os.environ, "Please set `COMET_API_KEY` before running this script!"
    logger = ExistingExperiment(
        previous_experiment=exp_key,
    )
    return logger
