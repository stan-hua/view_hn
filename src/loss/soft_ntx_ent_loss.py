"""
soft_ntx_ent_loss.py

Description: Contains soft NT-Xent loss, requiring labels.
             For a given sample, it identifies those with the same label as
             positive samples
"""

# Non-standard libraries
import torch
from torch import nn


################################################################################
#                             SoftNTXentLoss Class                             #
################################################################################
class SoftNTXentLoss:
    """
    Implementation of the Contrastive Cross Entropy Loss with same-label
    positive sampling.

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:
        >>> loss_fn = SoftNTXentLoss()

        >>> # Generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)

        >>> # Feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)

        >>> # Calculate loss, where positive samples are those of the same label
        >>> loss = loss_fn(output, labels)

    """

    # TODO: Add flag to set soft vs. hard labeling
    #       Hard labels only sets 1 value as the label
    #       Soft labels sets all possible labels with an equal probability
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        if abs(self.temperature) < 1e-8:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

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
            labels : torch.Tensor
                Labels for each image in the batch

        Returns:
            Contrastive Cross Entropy Loss value.
        """
        device = out0.device

        # Normalize the output to length 1
        # NOTE: This makes dot product on (out1, out2) -> cosine similarity
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # Compute cosine similarity using einsum, where:
        #   n = batch_size,
        #   c = embedding_size
        logits = torch.einsum("nc, mc -> nm", out0, out1) / self.temperature

        # Create soft labels
        # NOTE: Positive samples are those with the same label
        #       and thus, have equally likely probability of being chosen
        soft_labels = []
        cache_soft_labels = {}
        for label in labels:
            # Check cache
            if label in cache_soft_labels:
                soft_labels.append(torch.clone(cache_soft_labels[label]))

            # Create mask for same-label samples
            same_label_mask = (labels == label)
            equal_prob = 1 / same_label_mask.sum()

            # Create soft labels
            soft_labels_i = torch.FloatTensor([0.] * len(same_label_mask),
                                              device=device)
            soft_labels_i[same_label_mask] = equal_prob

            # Store soft labels
            soft_labels.append(soft_labels_i)
            cache_soft_labels[label] = soft_labels_i
        soft_labels = torch.stack(soft_labels)

        # Compute loss
        loss = self.cross_entropy(logits, soft_labels)

        return loss


if __name__ == "__main__":
    # Example data
    out1 = torch.stack([torch.Tensor([i] * 32) for i in range(6)])
    out2 = torch.stack([torch.Tensor([i] * 32) for i in range(6)])
    out2[3:5, :] = 0.
    labels = torch.Tensor([0, 1, 2, 0, 0, 5])

    # Instantiate loss function
    loss_fn = SoftNTXentLoss()

    # Calculate soft label loss
    loss_fn.forward(out1, out2, labels)
