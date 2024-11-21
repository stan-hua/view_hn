"""
app.py

Description: Contains functions to help serve a model locally, on a connected
             video capture.
"""

# Non-standard libraries
import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.transforms.v2 import Resize, ToTensor

# Custom libraries
from src.data import constants
from src.scripts.load_model import load_pretrained_from_exp_name, get_hyperparameters


################################################################################
#                                  Constants                                   #
################################################################################
# Video Streamer
STREAMER = None

# Name of model to use
EXP_NAME = "exp_param_sweep-supervised_baseline-only_beamform"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


################################################################################
#                                   Classes                                    #
################################################################################
class VideoStreamer():
    """
    VideoStreamer class.

    Note
    ----
    Class to get current image frame from Video Capture device
    """

    def __init__(self, index=0):
        """
        Initializes the StreamImage class with a video capture device.

        Parameters
        ----------
        index : int, optional
            Index of the video capture device, by default 0.
        """
        self.video_cap = cv2.VideoCapture(index)


    def get_frame(self):
        """
        Retrieves a frame from the video capture device

        Returns
        -------
        np.ndarray or None
            A frame from the video capture device if read is successful, otherwise None
        """
        assert self.video_cap.isOpened(), "Video capture device is not opened!"
        ret, img_arr = self.video_cap.read()
        # Early return, if unable to get frame
        if not ret:
            return None

        # Convert to grayscale
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

        # Perform histogram equalization
        # img_arr = cv2.equalizeHist(img_arr)

        # Convert back to 3 channels
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)

        # Center crop
        # img_arr = center_crop(img_arr)

        return img_arr


    def __del__(self):
        self.video_cap.release()


################################################################################
#                                  Functions                                   #
################################################################################
def center_crop(img_arr):
    """
    Center crop an image

    Parameters
    ----------
    img_arr : np.ndarray
        Input image of shape (H, W, C)

    Returns
    -------
    np.ndarray
        Cropped image
    """
    # Get the dimensions of the image
    height, width = img_arr.shape[0], img_arr.shape[1]
    assert height != 3 and width != 3, f"Expects (H, W, 3)! Got shape {img_arr.shape}"
    center_x, center_y = width // 2, height // 2

    # Define crop width as the length of the smaller side
    crop_width = crop_height = min(height, width)

    # Calculate the starting and ending points
    start_x = center_x - (crop_width // 2)
    start_y = center_y - (crop_height // 2)
    end_x = start_x + crop_width
    end_y = start_y + crop_height

    # Crop the image
    cropped_img = img_arr[start_y:end_y, start_x:end_x]
    return cropped_img


def main(device=DEVICE):
    # Set up streamer
    STREAMER = VideoStreamer()

    # Get model hyperparameters
    hparams = get_hyperparameters(exp_name=EXP_NAME)
    label_part = hparams["label_part"]

    # Get mapping between label index and class name
    idx_to_class = constants.LABEL_PART_TO_CLASSES[label_part]["idx_to_class"]

    # Load model
    model = load_pretrained_from_exp_name(EXP_NAME).to(device)
    model.eval()

    # Predict on each images in real-time
    img_arr = STREAMER.get_frame()
    with st.empty():
        while img_arr is not None:
            # Convert to torch.Tensor
            img_tensor = ToTensor()(img_arr)
            # Resize image
            img_tensor = Resize(constants.IMG_SIZE)(img_tensor)
            # Add batch dimension, convert to float32 type and send to device
            img_tensor = img_tensor.unsqueeze(0).to(torch.float32).to(device)

            # Perform inference
            out = model(img_tensor)

            # Get index of predicted label
            pred = torch.argmax(out, dim=1)
            pred = int(pred.detach().cpu())

            # Convert to probabilities
            prob = torch.nn.functional.softmax(out, dim=1)
            prob_numpy = prob.detach().cpu().numpy().flatten().round(4)

            # Convert from encoded label to label name
            pred_label = idx_to_class[pred]

            # Display image
            st.image(
                img_arr,
                use_container_width=True,
                caption=f"Prediction: {pred_label} (p={prob_numpy[pred]})",
            )

            # Get next frame
            img_arr = STREAMER.get_frame()


if __name__ == "__main__":
    st.title("RenalView: Renal Ultrasound Kidney/Bladder Plane Labeler")
    st.write("")
    main()
