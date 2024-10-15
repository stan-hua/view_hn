"""
attr_encoding.py

Description: Given embeddings from trained classifiers, determine how much they
             encode for a specified attribute (e.g., disease severity, age)
"""

# Standard libraries
import json
import logging
import os
import sys

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Custom libraries
from src.scripts.embed import get_embeds_with_metadata
from src.scripts.model_eval import create_save_dir_by_flags


################################################################################
#                                    Setup                                     #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


################################################################################
#                                  Constants                                   #
################################################################################
# Flag to overwrite existing results
OVERWRITE = False

# Default flags for inference directory
DEFAULT_SAVE_FLAGS = {
    "ckpt_option": "best",
    "mask_bladder": False,
    "test_time_aug": False,
    "da_transform_name": None,
}

# Filename to save as
ATTR_ENCODING_FNAME = "attr_encoding.json"

################################################################################
#                               Helper Functions                               #
################################################################################
def prep_hn_severity_column(df_embeds, hn_col="hn", surgery_col="surgery"):
    """
    Prepare hydronephrosis severity column from "hn" and "surgery" column

    Note
    ----
    Column `hn_col` denotes if kidney has hydronephrosis or not
    Column `surgery_col` denotes if hydronephrosis surgery is required for
    kidney, in other words more severe hydronephrosis cases

    Parameters
    ----------
    df_embeds : pd.DataFrame
        DataFrame with both embeddings and metadata
    hn_col : str, optional
        Column name for having hydronephrosis (1/0)
    surgery_col : str, optional
        Column name for hydronephrosis requiring surgery (1/0)
    """
    # Ensure columns exist
    for col in (hn_col, surgery_col):
        assert col in df_embeds.columns, f"Column `{col}` not in DataFrame!"

    # Convert into 3 classes
    # 0 = no hydronephrosis, 1 = mild hydronephrosis, 2 = severe hydronephrosis
    df_embeds["hn_severity"] = df_embeds.apply(
        lambda row:
            (2 if float(row[surgery_col]) else (1 if float(row[hn_col]) else 0))
            if not pd.isnull(row[hn_col]) and not pd.isnull(row[surgery_col])
            else None,
        axis=1
    )

    return df_embeds


def extract_only_embed_cols(df_embeds):
    """
    Isolate embedding columns from a DataFrame with both embeddings and metadata.

    Parameters
    ----------
    df_embeds : pd.DataFrame
        DataFrame with both embeddings and metadata

    Returns
    -------
    pd.DataFrame
        DataFrame with only the embedding features
    """
    feature_cols = [col for col in df_embeds.columns
                    if isinstance(col, int) or col.isnumeric()]
    df_embeds_only = df_embeds[feature_cols]
    return df_embeds_only


def compute_attr_encoding(df_embeds, attr_col, split="val", cm_ax=None):
    """
    Compute macro F1-score for logistics regression of features on
    a given sensitive attribute.

    Parameters
    ----------
    df_embeds : pd.DataFrame
        DataFrame with both image embeddings and metadata
    attr_col : str
        Column name containing the categorical sensitive attribute to predict
        from embeddings
    split : str, optional
        Name of split to compute encoding with (validation/test set), by default
        "val"
    cm_ax : plt.Axes, optional
        If provided, plot Confusion Matrix to this axis

    Returns
    -------
    float
        Macro F1-score of classified sensitive attribute on validation or test
    """
    # Drop all rows with missing values
    LOGGER.info(f"Dropping {df_embeds[attr_col].isna().sum()} of {len(df_embeds)} rows with missing values!")
    df_embeds = df_embeds.dropna(subset=[attr_col])

    # Get train/val/test embeddings
    df_train = df_embeds[df_embeds["split"] == "train"]
    df_val = df_embeds[df_embeds["split"] == "val"]
    df_test = df_embeds[df_embeds["split"] == "test"]

    # Get X (embeddings) and y (sensitive attribute)
    X_train = extract_only_embed_cols(df_train).to_numpy()
    y_train = df_train[attr_col].to_numpy()
    X_val = extract_only_embed_cols(df_val).to_numpy()
    y_val = df_val[attr_col].to_numpy()
    X_test = extract_only_embed_cols(df_test).to_numpy()
    y_test = df_test[attr_col].to_numpy()

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # To handle imbalanced classes, compute class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Tune regularization strength using validation set
    LOGGER.info(f"[Attr Encoding] Perform L2 logistic reg. with varying L2 penalty strength...")
    Cs = 10**np.linspace(-5, 1, 10)
    val_f1s = []
    for C in Cs:
        log_reg = LogisticRegression(
            class_weight=class_weights_dict,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42,
            C=C
        )
        log_reg.fit(X_train, y_train)
        val_preds = log_reg.predict_proba(X_val)
        val_f1 = f1_score(y_val, val_preds.argmax(axis=1), average='macro')
        val_f1s.append(val_f1)
    LOGGER.info(f"[Attr Encoding] Perform L2 logistic reg. with varying L2 penalty strength...DONE")

    # Get "best" L2 penalty strength based on F1-score
    best_C = Cs[np.argmax(val_f1s)]
    best_log_reg = LogisticRegression(
        class_weight=class_weights_dict,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
        C=best_C
    )
    best_log_reg.fit(X_train, y_train)

    # Create a confusion matrix
    if cm_ax is not None:
        ConfusionMatrixDisplay.from_estimator(best_log_reg, X_val, y_val, ax=cm_ax)

    # CASE 1: Returning averaged validation results
    if split  == "val":
        LOGGER.info(f"[Attr Encoding] Returning avg. validation F1 across all L2 penalty strengths...")
        return round(sum(val_f1s) / len(val_f1s), 4)

    # CASE 2: Returning test results using best L2 penalty score from validation
    LOGGER.info(f"[Attr Encoding] Returning test F1 using best L2 penalty score from validation...")
    test_preds = best_log_reg.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, test_preds, multi_class='ovr')
    return test_auc


################################################################################
#                                  Functions                                   #
################################################################################
def main(exp_name, dset="sickkids_beamform", split="val", attr_cols=("hn_severity",),
         ckpt_option="best"):
    """
    Compute attribute encodings for all sensitive attributes specified in
    `attr_cols` for given experiment and dataset.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Name of dataset
    attr_cols : list of str, optional
        Column names of sensitive attributes to determine feature encodings.
    split : str, optional
        Name of split to compute encoding with (validation/test set), by default
        "val"
    ckpt_option : str, optional
        Which checkpoint to use: 'best' or 'last', by default "best"
    """
    LOGGER.info("Computing attribute encodings...")
    # Ensure inference directory exists
    # NOTE: Assumed to have already performed predictions and evaluations
    save_flags = DEFAULT_SAVE_FLAGS.copy()
    save_flags.update({"ckpt_option": ckpt_option})
    save_dir = create_save_dir_by_flags(exp_name, dset, **save_flags)
    if not os.path.exists(save_dir):
        raise RuntimeError("Save directory does not exist: %s" % (save_dir,))

    # Add attr_encoding subdirectory
    save_dir = os.path.join(save_dir, "attr_encoding")

    # Load existing attr encoding metrics, if available
    metric_path = os.path.join(save_dir, ATTR_ENCODING_FNAME)
    if os.path.exists(metric_path) and not OVERWRITE:
        with open(metric_path, 'r') as f:
            attr_metrics = json.load(f)

    # Get embeddings
    df_embeds = get_embeds_with_metadata(
        exp_name, dset=dset, split="all",
        ckpt_option=ckpt_option,
    )

    # Drop "Other" images
    df_embeds = df_embeds[df_embeds["label"] != "Other"]

    # Parse null columns
    df_embeds = df_embeds.replace(["nan", "NaN"], None)

    # If `hn_severity` specified, ensure column is created
    if "hn_severity" in attr_cols and "hn_severity" not in df_embeds.columns:
        df_embeds = prep_hn_severity_column(df_embeds)

    # Compute attribute encoding
    attr_metrics = {}
    for attr_col in attr_cols:
        # Add dataset and split
        if attr_col not in attr_metrics:
            attr_metrics[attr_col] = {}
        if dset not in attr_metrics[attr_col]:
            attr_metrics[attr_col][dset] = {}

        # Skip, if already computed
        if dset in attr_metrics[attr_col] and split in attr_metrics[attr_col][dset]:
            continue

        # Create copy to avoid in-place edits
        df_embeds_curr = df_embeds.copy()

        # Prepare axis to plot confusion matrix
        fig, cm_ax = plt.subplots()

        # Compute attribute encoding
        LOGGER.info(f"Computing attribute encodings ({attr_col})...")
        attr_metrics[attr_col][dset][split] = compute_attr_encoding(
            df_embeds_curr,
            attr_col=attr_col,
            cm_ax=cm_ax,
        )
        LOGGER.info(f"Computing attribute encodings ({attr_col})...DONE")

        # Add title
        fig.suptitle(f"Confusion Matrix for Attribute Encoding ({attr_col})")

        # Save confusion matrix
        cm_save_dir = os.path.join(save_dir, attr_col)
        os.makedirs(cm_save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(cm_save_dir, f"{dset}-{split}-confusion_matrix.png"),
            bbox_inches="tight",
        )
        plt.clf()


    # Save computed attribute encodings
    with open(metric_path, 'w') as f:
        json.dump(attr_metrics, f, indent=4)
    
    LOGGER.info("Computing attribute encodings...DONE")


if __name__ == "__main__":
    Fire(main)
