"""
influence_function.py

Description: This file contains a wrapper over influence functions for model
             debugging
"""

# Standard libraries
import random

# Non-standard libaries
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import torch
import torch.nn.functional as F
from torch_influence import (
    AutogradInfluenceModule, CGInfluenceModule, LiSSAInfluenceModule,
    BaseObjective,
)
from tqdm import tqdm

# Custom libaries
from config import constants


################################################################################
#                                  Constants                                   #
################################################################################
# Get device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping of influence modules
INF_MODULES = {
    "autograd": AutogradInfluenceModule,
    "cg": CGInfluenceModule,
    "lissa": LiSSAInfluenceModule
}
# Influence module default hyperparameters
INF_HPARAMS = {
    "cg": {
        "atol": 1e-8,
        "maxiter": 1000,
    },
    "lissa": {
        "repeat": 5,
        "depth": 10000,
        "scale": 100,       # NOTE: Typically treated as a hyperparameter
    }
}


################################################################################
#                                  Functions                                   #
################################################################################
class MyCrossEntropyLoss(BaseObjective):
    """
    L2-regularized cross entropy loss

    Note
    ----
    Reproducing the same loss is important 
    """

    def __init__(self, hparams=None):
        """
        Initialize MyCrossEntropyLoss object.

        Parameters
        ----------
        hparams : dict, optional
            Model hyperparameters, by default None
        """
        super().__init__()
        self.hparams = hparams or {}
        self.loss = torch.nn.NLLLoss()


    def train_outputs(self, model, batch):
        """
        Pass data through model
        """
        data, _ = batch

        # Get logits
        out = model(data)

        return out


    def train_loss_on_outputs(self, outputs, batch):
        """
        Compute training loss, excluding regularization terms.
        """
        _, metadata = batch

        # Get label
        y_true = metadata["label"]

        # Get loss
        loss = self.loss(F.log_softmax(outputs, dim=1), y_true)
        return loss


    def train_regularization(self, params):
        """
        Compute loss regularization terms.
        """
        # Compute loss regularization terms
        reg = 0.

        # 1. Weight decay
        weight_decay = self.hparams.get("weight_decay")
        if weight_decay:
            reg += weight_decay * torch.square(params.norm())

        return reg


    def test_loss(self, model, params, batch):
        """
        Compute test loss, excluding regularization terms.
        """
        # NOTE: No regularization in the test loss
        out = self.train_outputs(model, batch)
        loss = self.train_loss_on_outputs(out, batch)
        return loss


def plot_most_helpful_harmful_examples(model, hparams, train_loader, test_loader,
                                       how="cg", num_examples=9):
    """
    Get most helpful and harmful training examples.

    Parameters
    ----------
    model : torch.nn.Module
        Arbitrary model
    hparams : dict
        Model hyperparameters
    train_loader : torch.utils.data.DataLoader
        Training set dataloader
    test_loader : torch.utils.data.DataLoader
        Validation/Test set dataloader
    how : str, optional
        Method to compute inverse Hessian-vector product (autograd=AutoGrad,
        cg=Conjugate Gradients, lissa=Linear time Stochastic Second-Order Algo.)
    num_examples : int, optional
        Maximum number of examples to show for each class.

    Returns
    -------
    dict of {kind (helpful/harmful): {label: plt.Figure}}
        Contains grid plots of the most helpful/harmful images for each label.
    """
    assert how in INF_MODULES, f"Must be in {tuple(INF_MODULES.keys())}"

    # Get mapping of integer-encoded label to string label
    idx_to_class = constants.LABEL_PART_TO_CLASSES[hparams["label_part"]]["idx_to_class"]

    # Get size of datasets
    train_set_size = len(train_loader.dataset)
    test_set_size = len(test_loader.dataset)

    # Create indices for train/test set
    train_indices = list(range(train_set_size))
    test_indices = list(range(test_set_size))
    # Randomly shuffle test set indices
    random.shuffle(test_indices)

    # Create objective
    objective = MyCrossEntropyLoss(hparams)

    # Send model to device
    model = model.to(DEVICE)

    # Get the top test images from each label with the highest test loss
    label_to_test_sample = {}
    for idx, (data, metadata) in enumerate(test_loader):
        # Get test point data
        data = data.to(DEVICE)
        metadata["label"] = metadata["label"].to(DEVICE)
        ret = model.test_step((data, metadata), -1)
        decoded_label = idx_to_class[metadata["label"][0].cpu().item()]

        # Skip, if correctly classified
        if (ret["y_pred"] == ret["y_true"]).all():
            continue

        # Store, only if test loss is worse than previous
        loss = float(ret["loss"].item())
        if loss > label_to_test_sample.get(decoded_label, {}).get("loss", -1000):
            label_to_test_sample[decoded_label] = { 
                "idx": idx,
                "loss": loss,
                "y_pred": ret["y_pred"][0].item(),
                "y_true": ret["y_true"][0].item(),
            }

    # Create influence module
    module = INF_MODULES[how](
        model=model,
        objective=objective,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        damp=0.001,
        **INF_HPARAMS.get(how, {}),
    )

    # Get the indices of the most helpful/harmful training images for each test sample
    # NOTE: Split by true label
    test_imgs_and_captions = []
    helpful_imgs_and_captions = []
    harmful_imgs_and_captions = []
    for label, metadata in tqdm(label_to_test_sample.items(), desc="Computing Influences"):
        # Store test sample
        test_img = get_image_from_index(metadata["idx"], test_loader)
        test_caption = f"Pred: {metadata['y_pred']}, Label: {metadata['y_true']}\n" \
                       f"Test Loss: {metadata['loss']:.4f}"
        test_imgs_and_captions.append((test_img, test_caption))

        # Compute influence scores
        # NOTE: This may be expensive since re-computing gradients over entire
        #       training set. Consider caching
        inf_scores = module.influences(train_idxs=train_indices, test_idxs=[metadata["idx"]])

        # Store the index of the most helpful and harmful training set image
        helpful_train_idx, helpful_score = inf_scores.argmax(), inf_scores.max()
        harmful_train_idx, harmful_score = inf_scores.argmin(), inf_scores.min()

        # Predict on helpful training sample and store info
        helpful_ret = predict_single_by_index(helpful_train_idx, model, train_loader)
        helpful_img = get_image_from_index(helpful_train_idx, train_loader)
        helpful_caption = f"Pred: {helpful_ret['y_pred']}, Label: {metadata['label']}\n" \
                          f"Influence: {helpful_score:.4f}"
        helpful_imgs_and_captions.append((helpful_img, helpful_caption))

        # Predict on harmful training sample and store info
        harmful_ret = predict_single_by_index(harmful_train_idx, model, train_loader)
        harmful_img = get_image_from_index(harmful_train_idx, train_loader)
        harmful_caption = f"Pred: {harmful_ret['y_pred']}, Label: {metadata['label']}\n" \
                          f"Influence: {harmful_score:.4f}"
        harmful_imgs_and_captions.append((harmful_img, harmful_caption))

    # Create plot
    fig = plot_test_helpful_harmful(test_imgs_and_captions,
                                    helpful_imgs_and_captions,
                                    harmful_imgs_and_captions)

    return fig


def plot_test_helpful_harmful(test_imgs_and_captions,
                              helpful_imgs_and_captions,
                              harmful_imgs_and_captions):
    """
    Plot test sample and the most helpful and most harmful training examples
    for that test sample.

    Parameters
    ----------
    test_imgs_and_captions : list of tuples
        Each tuple contains the image and caption
    helpful_imgs_and_captions : list of tuples
        Each tuple contains the image and caption
    harmful_imgs_and_captions : list of tuples
        Each tuple contains the image and caption

    Returns
    -------
    plt.Figure
        Created figure
    """
    # Create figure
    fig, axes = plt.subplots(
          nrows=3, ncols=len(test_imgs_and_captions),
          sharex=True, sharey=True,
          figsize=(15, 8))
    plt.subplots_adjust(wspace=0.5, hspace=0.75)

    # Plot each image
    for row in zip([test_imgs_and_captions, helpful_imgs_and_captions, harmful_imgs_and_captions], axes):
        for ((image, caption), ax) in zip(*row):
            ax.set_title(caption, size="medium")
            ax.set(aspect="equal")
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()

    # Draw vertical lines to separate each test example
    def get_bbox(ax_):
        r = fig.canvas.get_renderer()
        return ax_.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)
    xmax = np.array(list(map(lambda b: b.x1, bboxes.flat))).reshape(axes.shape).max(axis=0)
    xmin = np.array(list(map(lambda b: b.x0, bboxes.flat))).reshape(axes.shape).min(axis=0)
    xs = np.c_[xmax[1:], xmin[:-1]].mean(axis=1)
    for x in xs:
        line = plt.Line2D([x, x], [0.03, 0.97], transform=fig.transFigure, color="black", linewidth=1.2)
        fig.add_artist(line)

    # Create row labels
    row_labels = ["Test Image", "Most Helpful", "Most Harmful"]
    for ax, label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(label, rotation=90, size="x-large", fontweight="bold")

    return fig


def predict_single_by_index(idx, model, dataloader):
    """
    Predict on single sample, based on their dataset index.

    Parameters
    ----------
    idx : int
        Sample/image index
    model : torch.nn.Module
        Arbitrary model
    dataloader : torch.utils.data.DataLoader
        Arbitrary DataLoader

    Returns
    -------
    dict
        Contains prediction and label for sample
    """
    # Get test point data
    data, metadata = dataloader.dataset[idx]
    metadata = metadata.copy()

    # Send data to the right device
    data = data.unsqueeze(0).to(DEVICE)
    metadata["label"] = torch.LongTensor([metadata["label"]]).to(DEVICE)

    # Pass to model
    ret = model.test_step((data, metadata), -1)

    # Flatten items
    ret["loss"] = float(ret["loss"].item())
    ret["y_true"] = ret["y_true"][0].item()
    ret["y_pred"] = ret["y_pred"][0].item()

    # Store index
    ret["idx"] = idx

    return ret


def get_image_from_index(idx, dataloader):
    """
    Given an image index, get their corresponding image.

    Parameters
    ----------
    idx : int
        Image index
    dataloader : torch.utils.data.DataLoader
        Arbitrary DataLoader

    Returns
    -------
    list of np.array
        Each item() in the list is an image corresponding to the indices
    """
    X_curr, _ = dataloader.dataset[idx]
    img = 255. * X_curr.cpu().numpy()
    return img
