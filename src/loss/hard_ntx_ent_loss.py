"""
hard_ntx_ent_loss.py

Description: Contains hard same-label NT-Xent loss.
             For a given sample, it identifies the first pair with the same
             label as positive samples. Ignores remaining same-label samples
"""

# Non-standard libraries
import torch
from torch import nn


################################################################################
#                            HardSLNTXentLoss Class                            #
################################################################################
class HardSLNTXentLoss:
    """
    Implementation of the NT-Xent Loss, where same-label samples are ignored.

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:
        >>> loss_fn = HardSLNTXentLoss()

        >>> # Generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)

        >>> # Feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)

        >>> # Calculate loss, where positive samples are those of the same label
        >>> loss = loss_fn(output, labels)

    """

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        if abs(self.temperature) < 1e-8:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))


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
            labels : torch.Tensor
                Labels for each image in the batch

        Returns:
            Contrastive Cross Entropy Loss value.
        """
        device = out0.device
        # B := batch size
        B = out0.size(dim=0)

        # Normalize the output to length 1
        # NOTE: This makes dot product on (out1, out2) -> cosine similarity
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # Compute cosine similarity using einsum, where:
        #   n = batch_size,
        #   c = embedding_size
        logits = torch.einsum("nc, mc -> nm", out0, out1) / self.temperature

        # Create mask for same-label images, which are NOT the paired image
        same_label_mask = []
        cache_same_labels = {}
        for i, label in enumerate(labels):
            # Check cache
            if label in cache_same_labels:
                same_label_mask.append(torch.clone(cache_same_labels[label]))

            # Create mask for same-label samples
            same_label_mask_i = (labels == label)

            # Ignore paired image
            same_label_mask_i[i] = 0.

            # Store hard labels
            same_label_mask.append(same_label_mask_i)
        same_label_mask = torch.stack(same_label_mask)

        # Get labels (paired image)
        true_labels = torch.eye(B, device=device)

        # Mask out (ignore) other same-label similarities
        logits = logits[~same_label_mask]
        labels = true_labels[~same_label_mask]

        # Compute loss
        loss = self.cross_entropy(logits, labels)

        return loss


if __name__ == "__main__":
    # B ::= batch size
    # C ::= embedding dimensionality
    # P ::= amount of perturbation to mimic 2nd batch
    # sl_sim ::= desired similarity of same-label comparisons
    B = 10
    C = 256
    P = 0.1

    # Simulate features for 1st augmented batch
    out0 = torch.rand(B, C, requires_grad=True)
    # Perform small perturbation to mimic 2nd augmented batch
    out1 = out0 - (P * torch.rand(B, C, requires_grad=True))

    # Example labels
    labels = torch.LongTensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])

    # Compute loss
    loss_fn = HardSLNTXentLoss()
    loss = loss_fn(out0, out1, labels)
