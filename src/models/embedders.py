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
from src.data.constants import CYTO_WEIGHTS_PATH
from src.data_prep.dataset import UltrasoundDataModule
from src.models.siamnet import load_siamnet


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

        # Check if custom embed function present. If not, use forward pass
        if hasattr(self.model, "forward_embed"):
            features = self.model.forward_embed(img)
        else:
            features = self.model(img)
        
        # If more than 1 image, attempt to flatten extra 3rd dimension
        if features.shape[0] > 1:
            features = features.squeeze()

        return features

    def predict_dir_tf(self, dir, save_path):
        """
        Extract embeddings for all images in the directory, using Tensorflow
        data loading libraries.

        Parameters
        ----------
        dir : str
            Path to directory containing images
        save_path : str
            File path to save embeddings at. Saved as h5 file.

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
        
        # Get file paths
        files_ds = tf.data.Dataset.list_files(dir + "/*", shuffle=False)

        # Get images from file paths
        img_ds = files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

        # Make image generator efficient via prefetching
        img_ds = img_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Extract embeddings in batches of 32
        all_embeds = self.model.predict(img_ds)

        # Save embeddings
        df_features = pd.DataFrame(np.array(all_embeds))
        df_features["files"] = list(files_ds.as_numpy_iterator())
        df_features["files"] = df_features["files"].str.decode("utf-8")
        df_features.to_hdf(save_path, "embeds")

    def predict_dir_torch(self, dir, save_path):
        """
        Extract embeddings for all images in the directory, using PyTorch
        libraries.

        Parameters
        ----------
        dir : str
            Path to directory containing images
        save_path : str
            File path to save embeddings at. Saved as h5 file.

        Returns
        -------
        pd.DataFrame
            Contains a column for the path to the image file. Other columns are
            embedding columns
        """
        assert os.path.isdir(dir), "Path provided does not lead to a directory!"
        assert self.model not in ("cytoimagenet", "imagenet"), \
            "Do not use this function if not extracting Cyto/ImageNet features!"

        # Load data
        data_module = UltrasoundDataModule(dir=dir)
        data_module.setup()
        train_set = data_module.train_dataloader()

        all_embeds = []
        file_paths = []

        for img, metadata in tqdm(train_set):
            # Extract embeddings
            embeds = self.predict_torch(img)

            all_embeds.append(embeds)
            file_paths.extend(metadata["filename"])

        # Save features. Each row is a feature vector
        df_features = pd.DataFrame(np.concatenate(all_embeds))
        df_features['files'] = file_paths
        df_features.to_hdf(save_path, "embeds")
        
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
                        help="cytoimagenet or imagenet")
    parser.add_argument("--img_file", type=str, default=None,
                         help="Path to an image file")
    parser.add_argument("--img_dir", type=str, default=None,
                         help="Path to a directory of images")
    parser.add_argument("--save_embed_path", type=str,
                        help="File path to save embeddings at")
    parser.add_argument("--cyto_weights_path", type=str,
                        default=CYTO_WEIGHTS_PATH, help="Path to CytoImageNet-"
                        "trained EfficientNetB0 model weights")

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
        feature_extractor = EfficientNetB0(weights='imagenet',
                                           include_top=False,
                                           input_shape=(None, None, 3),
                                           pooling="avg")
    elif model_name == "hn":
        feature_extractor = load_siamnet()

    embedder = ImageEmbedder(feature_extractor)

    return embedder


def main(model_name, save_embed_path, img_file=None, img_dir=None,
         cyto_weights_path=CYTO_WEIGHTS_PATH):
    """
    Instantiates embedder class with appropriate feature extractor, and extracts
    embedding to path.

    Note
    ----
    img_file and img_dir cannot both be empty.

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
    cyto_weights_path : str, optional
        Path to cytoimagenet weights, by default CYTO_WEIGHTS_PATH
    """
    assert img_file or img_dir, \
        "At least one of img_file or img_dir must be given!"

    weights = None
    if model_name == "cytoimagenet":
        weights = cyto_weights_path

    # Get feature extractor
    embedder = instantiate_embedder(model_name, weights=weights)
    
    if img_file:
        embeds = embedder.predict_torch(img_file)
        df_embed = pd.DataFrame(embeds).T
        df_embed['filename'] = img_file
        df_embed.to_hdf(save_embed_path)
        return

    if model_name == "hn":
        embedder.predict_dir_torch(img_dir, save_embed_path)
    else:
        embedder.predict_dir_tf(img_dir, save_embed_path)


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == '__main__':
    args = get_arguments()

    main(args.model_name, args.save_embed_path,
         args.img_file, args.img_dir, args.cyto_weights_path)