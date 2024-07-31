"""
sup_moco_loss.py

Description: Supervised NT-Xent loss implementation, where memory bank does not
             pull negative samples containing examples from the positive class.
"""

# Non-standard libraries
import torch
from lightly.utils import dist
from torch import distributed as torch_dist


################################################################################
#                                   Classes                                    #
################################################################################
class SupMoCoLoss(torch.nn.Module):
    """
    SupMoCoLoss class.

    Note
    ----
    Implementation of the supervised MoCo loss; a supervised NT-Xent loss with
    a memory bank with embeddings and their labels. "NT-Xent" stands for
    normalized temperature-scaled cross entropy loss.
    """

    def __init__(
        self,
        num_classes=3,
        temperature: float = 0.5,
        memory_bank_size: int = 0,
        gather_distributed: bool = False,
    ):
        """
        Initialize NTXentLoss object.

        Parameters
        ----------
        num_classes : int, optional
            Number of classes, by default 3
        temperature : float, optional
            Temperature to scale logits, by default 0.5
        memory_bank_size : int, optional
            Size of desired memory bank, by default 0
        gather_distributed : bool, optional
            If True, training on multiple GPUs, by default False
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        # Create a memory bank that stores past batches and their labels
        self.memory_bank = LabeledMemoryBank(num_classes, memory_bank_size)

        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )


    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass through Supervised Contrastive Cross-Entropy Loss.

        Note
        ----
        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.

        Parameters
        ----------
        out0 : torch.FloatTensor
            Output projections of the first set of transformed images.
            Shape: (batch_size, embedding_size)
        out1 : torch.FloatTensor
            Output projections of the second set of transformed images.
            Shape: (batch_size, embedding_size)
        labels : torch.LongTensor
            Labels corresponding to both out0 and out2, which is assumed to be
            the same for both out0 and out1.

        Returns
        -------
        torch.Tensor
            Supervised MoCo Loss value.
        """
        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, mem_bank_samples, mem_bank_labels = self.memory_bank.forward(
            out1, labels, update=out0.requires_grad
        )

        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        ########################################################################
        #                      CASE 1: Using Memory Bank                       #
        ########################################################################
        # CASE 1: Memory bank provided examples
        if mem_bank_samples is not None:
            # use negatives from memory bank
            mem_bank_samples = mem_bank_samples.to(device)

            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum("nc,nc->n", out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum("nc,ck->nk", out0, mem_bank_samples)

            # For each sample, get negative memory bank samples of different labels
            # NOTE: Each could be of varying lengths
            same_label_mask = (labels.unsqueeze(1) == mem_bank_labels.unsqueeze(0))
            accum_loss = 0
            for idx in range(batch_size):
                curr_not_label_mask = ~same_label_mask[idx, :]
                curr_neg_samples = sim_neg[idx, :][curr_not_label_mask]
                # NOTE: First element is always the positive sample
                logits = torch.cat([sim_pos[idx], curr_neg_samples]).unsqueeze(0)

                # Compute loss for each sample, separately
                target = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
                accum_loss = accum_loss + self.cross_entropy(logits, target)

            # Compute loss as mean across samples
            loss = accum_loss / batch_size
            return loss

        ########################################################################
        #                    CASE 2: Not Using Memory Bank                     #
        ########################################################################
        #  CASE 2: Memory bank is not active or empty
        # Raise error, if doing multi-gpu training
        if self.gather_distributed and dist.world_size() > 1:
            raise NotImplementedError("Not implemented for  multi-GPU training!")

        # Calculate similiarities
        # NOTE: Each matrix is of shape (n, n)
        logits_00 = torch.einsum("nc,mc->nm", out0, out0) / self.temperature
        logits_01 = torch.einsum("nc,mc->nm", out0, out1) / self.temperature
        logits_10 = torch.einsum("nc,mc->nm", out1, out0) / self.temperature
        logits_11 = torch.einsum("nc,mc->nm", out1, out1) / self.temperature

        # Create diagonal mask
        diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

        # Create mask for same label images
        same_label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))

        # Compute loss on each sample
        accum_loss = 0
        for idx in range(batch_size):
            # Get mask for the current sample for DIFFERENT label images
            curr_not_label_mask = ~same_label_mask[idx, :]
            # Get mask for same image
            curr_diag_mask = diag_mask[idx, :]

            # Get similarities for the current sample
            # 1. Same view
            curr_00 = logits_00[idx, :]
            curr_11 = logits_11[idx, :]
            # 2. Different view
            curr_01 = logits_01[idx, :]
            curr_10 = logits_10[idx, :]

            # Remove same image/label comparisons in same view comparisons
            curr_00 = curr_00[curr_not_label_mask]
            curr_11 = curr_11[curr_not_label_mask]

            # Extract same image / diff view similarity
            sim_pos_01 = curr_01[curr_diag_mask]
            sim_pos_10 = curr_10[curr_diag_mask]

            # Remove same image/label comparisons in different view comparison
            curr_01 = curr_01[curr_not_label_mask]
            curr_10 = curr_10[curr_not_label_mask]

            # Create arrays where first element is the positive sample
            sim_all_01 = torch.cat([sim_pos_01, curr_01, curr_00]).unsqueeze(0)
            sim_all_10 = torch.cat([sim_pos_10, curr_10, curr_11]).unsqueeze(0)

            # Create targets
            target = torch.zeros(sim_all_01.shape[0], device=device, dtype=torch.long)

            # Compute loss two-way from 0->1 and 1->0
            loss_01 = self.cross_entropy(sim_all_01, target)
            loss_10 = self.cross_entropy(sim_all_10, target)
            accum_loss = accum_loss + loss_01 + loss_10

        # Compute loss as mean across samples
        loss = accum_loss / batch_size

        return loss


# TODO: Do ablation on `first_pass` logic
class LabeledMemoryBank(torch.nn.Module):
    """
    LabeledMemoryBank class.

    Note
    ----
    Stores labels additional to the embeddings
    """

    def __init__(self, num_classes=3, size: int = 2**16):
        super().__init__()

        if size < 0:
            msg = f"Illegal memory bank size {size}, must be non-negative."
            raise ValueError(msg)

        self.size = size
        self.num_classes = num_classes
        self.first_pass = True
        self.register_buffer(
            "bank", tensor=torch.empty(0, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "bank_ptr", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )
        # Store integer labels
        self.register_buffer(
            "labels", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )


    @torch.no_grad()
    def _init_memory_bank(self, dim: int):
        """
        Initialize the memory bank if it's empty

        Parameters
        ----------
        dim : int
            The dimension of the which are stored in the bank.
        """
        # create memory bank
        # we could use register buffers like in the moco repo
        # https://github.com/facebookresearch/moco but we don't
        # want to pollute our checkpoints
        self.bank = torch.randn(dim, self.size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=0)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

        # NOTE: Initialize uniformly random labels for memory bank examples
        self.labels = torch.randint(high=self.num_classes, size=(self.size,)).type_as(self.labels)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor, labels: torch.Tensor):
        """
        Dequeue the oldest batch and add the latest one

        Parameters
        ----------
        batch : torch.Tensor
            The latest batch of keys to add to the memory bank.
        labels : torch.Tensor
            Labels associated with each item in the batch
        """
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)

        if ptr + batch_size >= self.size:
            # Modify flag to say, the memory bank is completely full
            self.first_pass = False

            # Handle end of list
            self.bank[:, ptr:] = batch[: self.size - ptr].T.detach()
            self.labels[ptr:] = labels[: self.size - ptr].detach()
            self.bank_ptr[0] = 0

            # Handle start of list
            new_ptr = (ptr + batch_size) % self.size
            self.bank[:, :new_ptr] = batch[self.size - ptr:].T.detach()
            self.labels[:new_ptr] = labels[self.size - ptr:].detach()
            self.bank_ptr[0] = new_ptr
        else:
            self.bank[:, ptr : ptr + batch_size] = batch.T.detach()
            self.labels[ptr : ptr + batch_size] = labels.detach()
            self.bank_ptr[0] = ptr + batch_size


    def forward(self, output: torch.Tensor,
                labels: torch.Tensor = None,
                update: bool = False):
        """
        Query memory bank for additional negative samples

        Note
        ----
        The output if the memory bank is of size 0, otherwise the output
        and the entries from the memory bank.

        Parameters
        ----------
        output : torch.Tensor
            The output of the model.
        labels : torch.Tensor
            Batch labels

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.LongTensor)
            (i) Output provided
            (ii) Entire memory bank
            (iii) Stored labels for items in memory bank
        """

        # no memory bank, return the output
        if self.size == 0:
            return output, None, None

        _, dim = output.shape

        # initialize the memory bank if it is not already done
        if self.bank.nelement() == 0:
            self._init_memory_bank(dim)

        # query and update memory bank
        bank = self.bank.clone().detach()

        # get labels
        mem_bank_labels = self.labels.clone().detach()

        # Only return memory bank examples/labels that are stored
        if self.first_pass:
            ptr = int(self.bank_ptr)

            # CASE 1: First time running
            if ptr == 0:
                bank = mem_bank_labels = None
            # CASE 2: There is something in the memory bank, but not entirely
            else:
                bank = bank[:, :ptr]
                mem_bank_labels = mem_bank_labels[:ptr]

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output, labels)

        return output, bank, mem_bank_labels


################################################################################
#                                  Debugging                                   #
################################################################################
if __name__ == "__main__":
    # B ::= batch size
    # C ::= embedding dimensionality
    # P ::= amount of perturbation to mimic 2nd batch
    B = 10
    C = 256
    P = 0.1

    # Simulate features for 1st augmented batch
    out0 = torch.rand(B, C, requires_grad=True)
    # Perform small perturbation to mimic 2nd augmented batch
    out1 = out0 - (P * torch.rand(B, C, requires_grad=True))

    # Example labels
    labels = torch.LongTensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])

    # Set up loss function
    loss_fn = SupMoCoLoss(num_classes=3, memory_bank_size=100)

    # Compute loss (without memory bank)
    loss = loss_fn(out0, out1, labels)
    print(f"The loss is {loss}")

    # Compute loss (with memory bank)
    loss = loss_fn(out0, out1, labels)
    print(f"The loss is {loss}")
