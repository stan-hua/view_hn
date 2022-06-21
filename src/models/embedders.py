"""
embedders.py

Description: Used to create deep embeddings for images.
"""
# Standard libraries
import argparse
import cv2
import glob
from math import ceil
import os

# Non-standard libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from src.data.constants import CYTO_WEIGHTS_PATH


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

    def predict(self, img_path):
        """
        Extract embeddings for image at path.

        Parameters
        ----------
        img_path : str
            Path to image

        Returns
        -------
        np.array
            1280-dimensional embedding
        """
        img = cv2.imread(img_path)
        features = self.model.predict(img)

        return features

    def predict_dir(self, dir, save_path):
        """
        Extract embeddings for all images in the directory.

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

        all_embeds = []
        file_paths = []

        # Get embeddings for each image
        for img_path in glob.glob(os.path.join(dir, "*")):
            try:
                embeds = self.predict(img_path)
                all_embeds.append(embeds)
                file_paths.append(img_path)
            except Exception as error_msg:
                print(error_msg)
                print(f"{img_path} skipped!")
                pass

        # Save features. Each row is a 1280-dim feature vector corresponding to
        # an image.
        df_features = pd.DataFrame(np.array(all_embeds)).T
        df_features['paths'] = file_paths
        df_features.to_hdf(save_path)
        
        return df_features

    def predict_dir_tf(self, dir, save_path):
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


def instantiate_embedder(model_name, cyto_weights_path=CYTO_WEIGHTS_PATH):
    """
    Instantiates embedder.

    Parameters
    ----------
    model_name : str
        Either 'cytoimagenet' or 'imagenet'.
    cyto_weights_path : str, optional
        Path to cytoimagenet weights, by default CYTO_WEIGHTS_PATH
    
    Returns
    -------
    ImageEmbedder
    """
    if model_name == "cytoimagenet":
        feature_extractor = EfficientNetB0(weights=cyto_weights_path,
                                           include_top=False,
                                           input_shape=(None, None, 3),
                                           pooling="avg")
    else:
        feature_extractor = EfficientNetB0(weights='imagenet',
                                           include_top=False,
                                           input_shape=(None, None, 3),
                                           pooling="avg")

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

    embedder = instantiate_embedder(model_name, cyto_weights_path)
    
    if img_file:
        embeds = embedder.predict(img_file)
        df_embed = pd.DataFrame(embeds).T
        df_embed['filename'] = img_file
        df_embed.to_hdf(save_embed_path)
    else:
        embedder.predict_dir_tf(img_dir, save_embed_path)


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == '__main__':
    args = get_arguments()

    main(args.model_name, args.save_embed_path,
         args.img_file, args.img_dir, args.cyto_weights_path)