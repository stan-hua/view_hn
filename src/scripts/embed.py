"""
embed.py

Description: Contains function to extract embeddings from pretrained models.
"""

# Standard libraries
import argparse
import logging
import os

# Non-standard libraries
import numpy as np
import pandas as pd
# import tensorflow as tf
import torch
from tqdm import tqdm

# Custom libraries
from config import constants
from src.data_prep import utils
from src.data_prep.dataset import UltrasoundDataModule
from src.scripts import load_data, load_model


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

# Verbosity
VERBOSE = True

# Embedding path suffixes
EMBED_SUFFIX = "_embeddings(histogram_norm).h5"
EMBED_SUFFIX_RAW = "_embeddings(raw).h5"


################################################################################
#                                   Classes                                    #
################################################################################
class ImageEmbedder:
    """
    Class used to create image embeddings
    """

    def __init__(self, model):
        """
        Initialize ImageEmbedder with feature extractor model

        Parameters
        ----------
        model : object
            Image feature extractor containing a feature extraction method.
        """
        self.model = model


    def embed(self, save_path, **kwargs):
        """
        Embed images specified in a dataframe, directory or image dataloader.

        Parameters
        ----------
        save_path : str
            Path to save embeddings to

        Returns
        -------
        pandas.DataFrame
            Embedded features
        """
        # Separate by PyTorch and Tensorflow models
        if isinstance(self.model, torch.nn.Module):
            return self.embed_torch_batch(save_path, **kwargs)
        else:
            return self.embed_tf_batch(save_path, **kwargs)


    def embed_torch(self, data):
        """
        Extract embeddings for image, using a PyTorch model.

        Parameters
        ----------
        data : torch.Tensor
            Image tensor

        Returns
        -------
        np.array
            1280-dimensional embedding
        """
        assert isinstance(self.model, torch.nn.Module)

        # Set model to evaluation mode
        self.model.eval()

        # Check if custom embed function present
        with torch.no_grad():
            if hasattr(self.model, "forward_embed"):
                features = self.model.forward_embed(data)
            elif hasattr(self.model, "extract_embeds"):
                features = self.model.extract_embeds(data)
            else:
                self.model.extract_embeds
                raise RuntimeError("No feature extraction function defined!")
        
        # If more than 1 image, attempt to flatten extra 3rd dimension
        if features.shape[0] > 1:
            features = features.squeeze()
        # Convert to numpy
        if not isinstance(features, np.ndarray):
            features = features.cpu().numpy()

        return features


    def embed_tf_batch(self, save_path, img_dir=None,
                       img_dataloader=None,
                       device=None):
        """
        Extract embeddings for all images in the directory, using Tensorflow
        data loading libraries.

        Parameters
        ----------
        save_path : str
            File path to save embeddings at. Saved as h5 file.
        img_dir : str, optional
            Path to directory containing images, by default None.
        img_dataloader : torch.utils.data.DataLoader
            Image dataloader with metadata dictionary containing `filename`, by
            default None

        Returns
        -------
        pd.DataFrame
            Contains a column for the path to the image file. Other columns are
            embedding columns
        """
        raise NotImplementedError
    #     def process_path(file_path):
    #         """
    #         Loads image from file path

    #         Parameters
    #         ----------
    #         file_path : str
    #             Path to image

    #         Returns
    #         -------
    #         tf.Tensor
    #             Tensor containing image
    #         """
    #         # Load the raw data from the file as a string
    #         img = tf.io.read_file(file_path)
    #         img = tf.io.decode_jpeg(img, channels=3)
    #         return img

    #     # Input sanitization
    #     assert img_dir or img_dataloader is not None, \
    #         "Must specify img_dir/img_dataloader!"
    #     # Verify model provided
    #     assert not isinstance(self.model, torch.nn.Module), \
    #         "Provided model is not a Tensorflow model!"

    #     if img_dir:
    #         assert os.path.isdir(img_dir), "Path provided does not lead to a "\
    #                                        "directory!"
    #         # Get file paths
    #         files_ds = tf.data.Dataset.list_files(
    #             os.path.join(img_dir, "*"), shuffle=False)
    #         file_paths = list(files_ds.as_numpy_iterator())

    #         # Get images from file paths
    #         img_ds = files_ds.map(process_path,
    #                               num_parallel_calls=tf.data.AUTOTUNE)

    #         # Make image generator efficient via prefetching
    #         img_ds = img_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    #         # Extract embeddings in batches of 32
    #         all_embeds = self.model.predict(img_ds)
    #     else:   # DataLoader
    #         all_embeds = []
    #         file_paths = []

    #         # Extract embeddings in batches
    #         for img, metadata in tqdm(img_dataloader):
    #             x = img.detach().cpu().numpy()
    #             if len(x.shape) == 5:
    #                 x = x.squeeze(axis=0)
    #             # Flip around channel dimension
    #             if x.shape[1] == 3:
    #                 x = np.moveaxis(x, 1, 3)
    #             embeds = self.model.predict(x)

    #             all_embeds.append(embeds)
    #             file_paths.extend(metadata["filename"])

    #         # Concatenate batched embeddings
    #         all_embeds = np.concatenate(all_embeds)

    #     # Save embeddings
    #     df_features = pd.DataFrame(np.array(all_embeds))
    #     assert not df_features.isna().all(axis=None)
    #     df_features["filename"] = np.array(file_paths).flatten()
    #     df_features.to_hdf(save_path, "embeds")


    def embed_torch_batch(self, save_path,
                          img_dir=None, img_dataloader=None,
                          device=constants.DEVICE):
        """
        Extract embeddings for all images in the directory, using PyTorch
        libraries.

        Note
        ----
        One of (img_dir, img_dataloader) must be specified!

        Parameters
        ----------
        save_path : str
            File path to save embeddings at. Saved as h5 file.
        img_dir : str, optional
            Path to directory containing images, by default None.
        img_dataloader : torch.utils.data.DataLoader, optional
            Image dataloader with metadata dictionary containing `filename`, by
            default None
        device : str, optional
            Device to send data to, by default constants.DEVICE.

        Returns
        -------
        pd.DataFrame
            Contains a column for the path to the image file. Other columns are
            embedding columns
        """
        # INPUT: If GPU specified but no GPU available, default to CPU
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
        self.model = self.model.to(device)
        self.model.eval()

        # Input sanitization
        assert img_dir or img_dataloader, "Must specify at least image "\
                                          "directory or image dataloader"
        assert isinstance(self.model, torch.nn.Module), \
            "Provided model is not a PyTorch model!"

        # Get data loader
        if img_dir:               # by directory
            assert os.path.isdir(img_dir), "Path provided does not lead to a "\
                                           "directory!"
            # Prepare to load data
            data_module = UltrasoundDataModule(img_dir=img_dir, mode=3)
            data_module.setup()
            # NOTE: Extracts embeddings from all images in directory
            dataloader = data_module.train_dataloader()
        else:
            dataloader = img_dataloader

        all_embeds = []
        file_paths, ids, visits, seq_numbers = [], [], [], []
        dsets = []

        # Extract embeddings in batches
        with torch.no_grad():
            for data, metadata in tqdm(dataloader):
                # If shape is (1, seq_len, C, H, W), flatten first dimension
                if len(data.size()) == 5:
                    data = data.squeeze(dim=0)

                if device != "cpu":
                    data = data.to(device)
                embeds = self.embed_torch(data)

                # Remove possibly added extra dimension
                if len(embeds.shape) == 3 and embeds.shape[0] == 1:
                    embeds = embeds.squeeze(axis=0)
                elif len(embeds.shape) == 4 and \
                        embeds.shape[-2] == 1 and embeds.shape[-1] == 1:
                    embeds = embeds.squeeze(axis=2).squeeze(axis=2)
                    assert len(embeds.shape) == 2, \
                        "Unexpected case, where embed. output has 4 dimensions!"

                all_embeds.append(embeds)
                # Store metadata
                file_paths.extend(metadata["filename"])
                ids.extend(metadata["id"])
                visits.extend(metadata["visit"])
                seq_numbers.extend(metadata["seq_number"])
                dsets.extend(metadata["dset"])

        # Save embeddings. Each row is a feature vector
        df_features = pd.DataFrame(np.concatenate(all_embeds))
        assert not df_features.isna().all(axis=None)
        df_features["filename"] = np.array(file_paths).flatten()
        df_features["id"] = np.array(ids).flatten()
        df_features["visit"] = np.array(visits).flatten()
        df_features["seq_number"] = np.array(seq_numbers).flatten()
        df_features["dset"] = np.array(dsets).flatten()
        df_features.to_hdf(save_path, "embeds", mode="w")

        return df_features


################################################################################
#                              Feature Extraction                              #
################################################################################
def extract_embeds(model,
                   save_embed_path=None,
                   img_file=None,
                   **embedder_kwargs):
    """
    Given a Tensorflow/Torch model, extract features from

    Note
    ----
    img_file, img_dir and img_dataloader cannot both be empty.

    Parameters
    ----------
    model : tf.Model or torch.nn.Module
        Pretrained model
    save_embed_path : str, optional
        If path provided, stores embeddings at path, by default None.
    exp_name : str, optional
        Name of experiment. Can be used to generate `save_embed_path`, if it's
        not provided, by default None.
    img_file : str
        Path to image file
    img_dir : str
        Path to directory of images
    img_dataloader : torch.utils.data.DataLoader
        Image dataloader with metadata dictionary containing `filename`
    """
    # Wrap model in ImageEmbedder
    embedder = ImageEmbedder(model)

    # If an image file
    if img_file:
        embeds = embedder.embed_torch(img_file, **embedder_kwargs)
        df_embed = pd.DataFrame(embeds).T
        df_embed['filename'] = img_file
        df_embed.to_hdf(save_embed_path)
        return

    # Embed image directory/dataloader/dataframe
    return embedder.embed(save_embed_path, **embedder_kwargs)


################################################################################
#                              Loading Embeddings                              #
################################################################################
def get_embeds(name, **kwargs):
    """
    Retrieve extracted deep embeddings using model specified.

    Parameters
    ----------
    name : str or list
        Model/experiment name, or list of model/experiment names to concatenate
        embeddings.
    **kwargs : Keyword arguments for `get_save_path`
        raw : bool, optional
        segmented : bool, optional
        reverse_mask : bool, optional
    """
    assert isinstance(name, (str, list)), "Invalid type for `name`!"

    # If getting embeddings for 1 model
    if isinstance(name, str):
        embed_path = get_save_path(name, **kwargs)
        df_embeds = pd.read_hdf(embed_path, "embeds")
        return df_embeds

    # If getting embeddings for 1+ models
    embed_lst = []
    for name_i in name:
        embed_path = get_save_path(name_i, **kwargs)
        df_embed_i = pd.read_hdf(embed_path, "embeds")
        embed_lst.append(df_embed_i)
    df_embeds = pd.concat(embed_lst, axis=1)
    return df_embeds


def get_embeds_with_metadata(exp_name, dset="sickkids_beamform", split="all",
                             **kwargs):
    """
    Retrieves embeddings from specified pretrained model. Then adds metadata from
    image file paths.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    dset : str, optional
        Name of dataset
    split : str, optional
        Split of dataset, or "all" if doing train/val/test
    **kwargs : Any
        Keyword arguments to pass into `get_embeds`

    Returns
    -------
    pd.DataFrame
        Table with both embedding features and metadata
    """
    # Load embeddings for all splits
    all_embed_lst = []
    splits = ["train", "val", "test"] if split == "all" else [split]
    for curr_split in splits:
        df_embeds = get_embeds(exp_name, dset=dset, split=curr_split, **kwargs)
        # Rename old filename column
        df_embeds = df_embeds.rename(columns={"paths": "filename",
                                                "files": "filename"})
        # Add dset and split
        df_embeds["dset"] = dset
        df_embeds["split"] = curr_split
        all_embed_lst.append(df_embeds)

    # Concatenate embeddings
    df_embeds_all = pd.concat(all_embed_lst, ignore_index=True)

    # Extract metadata from image file paths
    try:
        df_metadata = utils.extract_data_from_filename_and_join(
            df_embeds_all,
            dset=dset,
            label_part=None,
        )
    # If errored, print experiment name
    except RuntimeError as error_msg:
        LOGGER.critical("[plot_umap] Failed to extract metadata for experiment:"
                        " %s", exp_name)
        raise error_msg

    # Isolate embedding columns
    feature_cols = [col for col in df_embeds_all.columns
                    if isinstance(col, int)]
    df_embeds_only = df_embeds_all[feature_cols]

    # Create table with both embeddings and metadata
    df_data = pd.concat([df_embeds_only, df_metadata], axis=1)
    df_data = df_data.loc[:, ~df_data.columns.duplicated()]

    return df_data


################################################################################
#                                Main Function                                 #
################################################################################
def main(args):
    """
    Extract embeddings for the following experiments and datasets

    Parameters
    ----------
    args : argparse.Namespace
        Contains command-line arguments
    """
    dsets = args.dsets
    splits = args.splits
    # If only one of dset/split is > 1, assume it's meant to be broadcast
    if len(dsets) == 1 and len(splits) > 1:
        LOGGER.info("Only 1 `dset` provided! Assuming same `dset` for all `splits`...")
        dsets = dsets * len(splits)
    if len(splits) == 1 and len(dsets) > 1:
        LOGGER.info("Only 1 `split` provided! Assuming same `split` for all `dsets`...")
        splits = splits * len(dsets)

    # Ensure number of datasets match number of splits
    assert len(dsets) == len(splits), "Length of `dsets` and `splits` do not match!"

    # Extract embeddings for each experiment name
    for exp_name in args.exp_name:
        # 1. Load pre-trained model
        model = load_model.load_pretrained_from_exp_name(
            exp_name,
            ckpt_option=args.ckpt_option,
        )

        # Load experiment hyperparameters
        exp_hparams = load_model.get_hyperparameters(exp_name=exp_name)

        # Overwrite hyperparameters for data loading
        exp_hparams["self_supervised"] = False
        exp_hparams["augment_training"] = False
        exp_hparams["shuffle"] = False
        exp_hparams["include_labeled_other"] = True
        exp_hparams["imbalanced_sampler"] = False

        # Extract embeddings for each dataset
        for idx, dset in enumerate(dsets):
            split = splits[idx]

            # If dataset was used in training, then load DM as usual
            if dset in exp_hparams.get("dsets", ["sickkids"]):
                dm = load_data.setup_data_module(hparams=exp_hparams, full_path=True)
                img_dataloader = dm.get_filtered_dataloader(split=split, dset=dset)
            else:
                # Get image dataloader
                img_dataloader = load_data.setup_default_dataloader_for_dset(
                    dset=dset, split=split,
                    full_path=True,
                    use_defaults=True,
                )

            # Create path to save embeddings
            save_embed_path = get_save_path(
                exp_name, dset=dset, split=split,
                ckpt_option=args.ckpt_option
            )
            # Early return, if embeddings already made
            if os.path.isfile(save_embed_path):
                LOGGER.info(f"Embeddings for exp_name: ({exp_name}), "
                            f"dset: ({dset}, {split}) already exists! Skipping...")
                continue

            # Perform extraction
            extract_embeds(model=model,
                           save_embed_path=save_embed_path,
                           img_dataloader=img_dataloader,
                           device=constants.DEVICE,)

            LOGGER.info(f"Success! Created embeddings for exp_name: "
                        f"({exp_name}), dset: ({dset}, {split})")


################################################################################
#                               Helper Functions                               #
################################################################################
def init(parser):
    """
    Initialize ArgumentParser arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser object
    """
    arg_help = {
        "exp_name": "Name of experiment/s to create embeddings for",
        "dsets": "Name of datasets to embed",
        "splits": "For each `dset`, what data split to embed",
        "ckpt_option": "Which checkpoint to use: 'best' or 'last'",
    }

    parser.add_argument("--exp_name", required=True, nargs="+",
                        help=arg_help["exp_name"])
    parser.add_argument("--dsets", required=True, nargs="+",
                        help=arg_help["dsets"])
    parser.add_argument("--splits", required=True, nargs="+",
                        help=arg_help["splits"])
    parser.add_argument("--ckpt_option", default="best",
                        help=arg_help["ckpt_option"])


def get_save_path(name, dset, split, ckpt_option="best",
                  raw=False, segmented=False, reverse_mask=False):
    """
    Create expected save path from model name and parameters.

    Parameters
    ----------
    name : str
        Model/Experiment name
    dset : str, optional
        Name of dataset (e.g., sickkids)
    split : str, optional
        Specific data split (i.e., train/val/test)
    ckpt_option : str, optional
        Which checkpoint to use: 'best' or 'last', by default "best"
    raw : bool, optional
        If True, extracts embeddings for raw images. Otherwise, uses
        preprocessed images, by default False.
    segmented : bool, optional
        If True, extracts embeddings for segmented images, by default False.
    reverse_mask : bool, optional
        If True, reverses mask for segmented images, by default False

    Returns
    -------
    str
        Full path to save embeddings
    """
    # Ensure that embedding directory exists
    if not os.path.exists(constants.DIR_EMBEDS):
        LOGGER.info("Embedding directory not found! Creating...")
        os.makedirs(constants.DIR_EMBEDS)

    embed_suffix = EMBED_SUFFIX_RAW if raw else EMBED_SUFFIX
    segmented_suffix = f"_segmented{'_reverse' if reverse_mask else ''}"
    # HACK: Only add infix for "last" ckpt embeddings
    save_path = f"{constants.DIR_EMBEDS}/{name}"\
                f"{'_last' if ckpt_option == 'last' else ''}" \
                f"{segmented_suffix if segmented else ''}"\
                f"{f'({dset}-{split})'}"\
                f"{embed_suffix}"

    return save_path


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    init(PARSER)

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Run main flow to extract embeddings
    main(ARGS)
