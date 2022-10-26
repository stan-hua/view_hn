"""
embedders.py

Description: Used to create deep embeddings for images.
"""
# Standard libraries
import argparse
import logging
import os

# Non-standard libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tqdm import tqdm

# Custom libraries
from src.data import constants
from src.data_prep.dataset import UltrasoundDataModule
from src.models.cpc import CPC
from src.models.moco import MoCo
from src.models.siamnet import load_siamnet
from src.models.efficientnet_pl import EfficientNetPL


logging.basicConfig(level=logging.DEBUG)

################################################################################
#                                  Constants                                   #
################################################################################
VERBOSE = True
LOGGER = logging.getLogger(__name__)


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
            Image feature extractor containing "predict" method.
        """
        self.model = model


    def predict_torch(self, img):
        """
        Extract embeddings for image, using a PyTorch model.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor

        Returns
        -------
        np.array
            1280-dimensional embedding
        """
        assert isinstance(self.model, torch.nn.Module)

        # Set model to evaluation mode
        self.model.eval()

        # Check if custom embed function present. If not, use forward pass
        with torch.no_grad():
            if hasattr(self.model, "forward_embed"):
                features = self.model.forward_embed(img)
            elif hasattr(self.model, "extract_embeds"):
                features = self.model.extract_embeds(img)
            else:
                features = self.model(img)
        
        # If more than 1 image, attempt to flatten extra 3rd dimension
        if features.shape[0] > 1:
            features = features.squeeze()

        return features


    def predict_dir_tf(self, save_path, img_dir=None, img_dataloader=None):
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
        def process_path(file_path):
            """
            Loads image from file path

            Parameters
            ----------
            file_path : str
                Path to image

            Returns
            -------
            tf.Tensor
                Tensor containing image
            """
            # Load the raw data from the file as a string
            img = tf.io.read_file(file_path)
            img = tf.io.decode_jpeg(img, channels=3)
            return img

        # Input sanitization
        assert img_dir or img_dataloader, "Must specify at least image "\
                                          "directory or image dataloader"
        # Verify model provided
        assert not isinstance(self.model, torch.nn.Module), \
            "Provided model is not a Tensorflow model!"

        if img_dir:
            assert os.path.isdir(img_dir), "Path provided does not lead to a "\
                                           "directory!"
            # Get file paths
            files_ds = tf.data.Dataset.list_files(
                os.path.join(img_dir, "*"), shuffle=False)
            file_paths = list(files_ds.as_numpy_iterator())

            # Get images from file paths
            img_ds = files_ds.map(process_path,
                                  num_parallel_calls=tf.data.AUTOTUNE)

            # Make image generator efficient via prefetching
            img_ds = img_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

            # Extract embeddings in batches of 32
            all_embeds = self.model.predict(img_ds)
        else:   # DataLoader
            all_embeds = []
            file_paths = []

            # Extract embeddings in batches
            for img, metadata in tqdm(img_dataloader):
                embeds = self.model.predict(img.detach().cpu())

                all_embeds.append(embeds)
                file_paths.extend(metadata["filename"])

            # Concatenate batched embeddings
            all_embeds = np.concatenate(all_embeds)

        # Save embeddings
        df_features = pd.DataFrame(np.array(all_embeds))
        df_features["files"] = file_paths
        df_features["files"] = df_features["files"].str.decode("utf-8")
        df_features.to_hdf(save_path, "embeds")


    def predict_dir_torch(self, save_path, img_dir=None, img_dataloader=None):
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
        img_dataloader : torch.utils.data.DataLoader
            Image dataloader with metadata dictionary containing `filename`, by
            default None

        Returns
        -------
        pd.DataFrame
            Contains a column for the path to the image file. Other columns are
            embedding columns
        """
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
        file_paths = []

        # Extract embeddings in batches
        for img, metadata in tqdm(dataloader):
            embeds = self.predict_torch(img)

            all_embeds.append(embeds)
            file_paths.extend(metadata["filename"])

        # Save embeddings. Each row is a feature vector
        df_features = pd.DataFrame(np.concatenate(all_embeds))
        df_features['files'] = file_paths
        df_features.to_hdf(save_path, "embeds", mode="w")

        return df_features


################################################################################
#                                  Functions                                   #
################################################################################
def get_arguments():
    """
    Get arguments from the command-line.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="imagenet",
                        help="Type of model", choices=constants.MODELS)
    parser.add_argument("--img_file", type=str, default=None,
                         help="Path to an image file")
    parser.add_argument("--img_dir", type=str, default=None,
                         help="Path to a directory of images")
    parser.add_argument("--save_embed_path", type=str,
                        help="File path to save embeddings at")
    parser.add_argument("--cyto_weights_path", type=str,
                        default=constants.CYTO_WEIGHTS_PATH,
                        help="Path to CytoImageNet-"
                        "trained EfficientNetB0 model weights")
    parser.add_argument("--cpc_ckpt_path", type=str,
                        default=constants.CPC_CKPT_PATH,
                        help="Path to CPC model checkpoint.")
    parser.add_argument("--moco_ckpt_path", type=str,
                        default=constants.MOCO_CKPT_PATH,
                        help="Path to MoCo model checkpoint.")

    return parser.parse_args()


def instantiate_embedder(model_name, weights):
    """
    Instantiates embedder.

    Parameters
    ----------
    model_name : str
        Either 'cytoimagenet' or 'imagenet'.
    weights : str, optional
        Path to model's weights
    
    Returns
    -------
    ImageEmbedder
    """
    if model_name == "cytoimagenet":
        feature_extractor = EfficientNetB0(weights=weights,
                                           include_top=False,
                                           input_shape=(None, None, 3),
                                           pooling="avg")
    elif model_name == "imagenet":
        feature_extractor = EfficientNetPL.from_pretrained(
            model_name="efficientnet-b0")
    elif model_name == "hn":
        feature_extractor = load_siamnet()
    elif model_name == "cpc":
        feature_extractor = CPC.load_from_checkpoint(weights)
    elif model_name == "moco":
        feature_extractor = MoCo.load_from_checkpoint(weights)
    elif model_name == "random":
        # Randomly initialized EfficientNet model
        feature_extractor = EfficientNetPL()

    # Wrap model in ImageEmbedder
    embedder = ImageEmbedder(feature_extractor)

    return embedder


def main(model_name, save_embed_path,
         img_file=None, img_dir=None, img_dataloader=None,
         cyto_weights_path=constants.CYTO_WEIGHTS_PATH,
         cpc_ckpt_path=constants.CPC_CKPT_PATH,
         moco_ckpt_path=constants.MOCO_CKPT_PATH):
    """
    Instantiates embedder class with appropriate feature extractor, and extracts
    embedding to path.

    Note
    ----
    img_file, img_dir and img_dataloader cannot both be empty.

    Parameters
    ----------
    model_name : str
        Either 'cytoimagenet' or 'imagenet'.
    save_embed_path : str
        File path to store embeddings at
    img_file : str
        Path to image file
    img_dir : str
        Path to directory of images
    img_dataloader : torch.utils.data.DataLoader
        Image dataloader with metadata dictionary containing `filename`
    cyto_weights_path : str, optional
        Path to cytoimagenet weights, by default constants.CYTO_WEIGHTS_PATH
    cpc_ckpt_path : str, optional
        Path to CPC checkpoint, by default constants.CPC_CKPT_PATH
    moco_ckpt_path : str, optional
        Path to MoCo checkpoint, by default constants.MOCO_CKPT_PATH
    """
    assert img_file or img_dir or img_dataloader, \
        "At least one of (img_file, img_dir, img_dataloader) must be given!"

    name_to_weights_path = {
        "cytoimagenet": cyto_weights_path,
        "cpc": cpc_ckpt_path,
        "moco": moco_ckpt_path
    }
    weights = name_to_weights_path.get(model_name)

    # Get feature extractor
    embedder = instantiate_embedder(model_name, weights=weights)
    
    if img_file:
        embeds = embedder.predict_torch(img_file)
        df_embed = pd.DataFrame(embeds).T
        df_embed['filename'] = img_file
        df_embed.to_hdf(save_embed_path)
        return

    # Separate by Tensorflow and PyTorch models
    if model_name == "cytoimagenet":
        embedder.predict_dir_tf(save_embed_path,
                                img_dir=img_dir,
                                img_dataloader=img_dataloader)
    else:
        embedder.predict_dir_torch(save_embed_path,
                                   img_dir=img_dir,
                                   img_dataloader=img_dataloader)


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == '__main__':
    args = get_arguments()

    main(args.model_name, args.save_embed_path,
         args.img_file, args.img_dir, args.cyto_weights_path)
