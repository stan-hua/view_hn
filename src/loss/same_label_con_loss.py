"""
sup_ntx_ent_loss.py

Description: Implements a Same-Label Contrastive Loss.
             Given batch where each image is augmented twice, attempts to
             optimize for:
                (a) 1.0 similarity with the same-image, and
                (b) `sl_sim` similarity with the same-label samples
"""

# Non-standard libraries
import torch
from torch import nn
from torch.nn import functional as F


################################################################################
#                            SameLabelConLoss Class                            #
################################################################################
class SameLabelConLoss:
    """
    Implementation of an Contrastive MSE Loss with same-label.

    Attributes:
        sl_sim: float, optional
            Desired same-label similarity. Defaults to 0.8

    Examples:
        >>> loss_fn = SameLabelConLoss()

        >>> # Generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)

        >>> # Feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)

        >>> # Calculate loss, where positive samples are those of the same label
        >>> loss = loss_fn(output, labels)

    """

    def __init__(self, sl_sim=0.8):
        """
        Initialize SameLabelConLoss object.

        Parameters
        ----------
        sl_sim: float, optional
            Desired same-label similarity. Defaults to 0.8
        """
        self.sl_sim = sl_sim
        self.mse_loss = nn.MSELoss(reduction="mean")


    def __call__(self, *args, **kwargs):
        """
        Redirect to function for forward pass.
        """
        return self.forward(*args, **kwargs)


    def forward(self, out0, out1, labels):
        """
        Forward pass through Contrastive Cross-Entropy Loss.

        Args:
            out0: torch.Tensor
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1: torch.Tensor
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            labels : torch.LongTensor
                Labels for each image in the batch

        Returns:
            Contrastive Cross Entropy Loss value.
        """
        # Device
        device = out0.device
        # Batch size
        B = out0.size(dim=0)

        # Normalize the output to length 1
        # NOTE: This makes dot product on (out1, out2) -> cosine similarity
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        # Compute cosine similarity using einsum, where:
        #   n = m = batch_size,
        #   c = embedding_size
        logits = torch.einsum("nc, mc -> nm", out0, out1)

        # Create N x N `desired` cosine similarity matrix, where:
        #   1. Same-sample comparisons should yield 1.0, and
        #   2. Same-label comparisons should yield `sl_sim`
        sim_label = torch.eye(B, device=device)
        for label in labels.unique():
            # Create mask for samples with the same-label 
            label_mask = (labels == label)

            # Iterate over each row, where a sample w/ specified label is compared
            for idx in torch.argwhere(label_mask).flatten():
                # Set desired similarity for same-label comparisons
                sim_label[idx, label_mask] = self.sl_sim
        # Ensure desired same-sample similarity is 1.
        sim_label[torch.eye(B, dtype=torch.bool)] = 1.

        # Compute loss
        loss = self.mse_loss(logits, sim_label)

        return loss


if __name__ == "__main__":
    # B ::= batch size
    # C ::= embedding dimensionality
    # P ::= amount of perturbation to mimic 2nd batch
    # sl_sim ::= desired similarity of same-label comparisons
    B = 10
    C = 256
    P = 0.1
    sl_sim = 0.8

    # Simulate features for 1st augmented batch
    out0 = torch.rand(B, C, requires_grad=True)
    # Perform small perturbation to mimic 2nd augmented batch
    out1 = out0 - (P * torch.rand(B, C, requires_grad=True))

    # Example labels
    labels = torch.LongTensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])

    # Compute loss
    loss_fn = SameLabelConLoss(sl_sim=sl_sim)
    loss = loss_fn(out0, out1, labels)
