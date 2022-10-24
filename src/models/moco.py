"""
moco.py

Description: Implementation of MoCo with an EfficientNet convolutional backbone,
             using Lightly.AI and PyTorch Lightning.
"""

# Standard libraries
import copy

# Non-standard libraries
import lightly
import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from torch.nn import functional as F

# Custom libraries
from src.data import constants


################################################################################
#                               MoCo Model Class                               #
################################################################################
# TODO: Consider same-patient memory banks to avoid conflict between
# `memory_bank_size` and `same_label` (in SSL data module)
class MoCo(pl.LightningModule):
    """
    MoCo for self-supervised learning.
    """
    def __init__(self, num_classes=5, img_size=(256, 256), adam=True, lr=0.0005,
                 momentum=0.9, weight_decay=0.0005,
                 memory_bank_size=4096, temperature=0.1,
                 extract_features=False, *args, **kwargs):
        """
        Initialize MoCo object.

        Parameters
        ----------
        num_classes : int, optional
            Number of classes to predict, by default 5
        img_size : tuple, optional
            Expected image's (height, width), by default (256, 256)
        adam : bool, optional
            If True, use Adam optimizer. Otherwise, use Stochastic Gradient
            Descent (SGD), by default True.
        lr : float, optional
            Optimizer learning rate, by default 0.0001
        momentum : float, optional
            If SGD optimizer, value to use for momentum during SGD, by
            default 0.9
        weight_decay : float, optional
            Weight decay value to slow gradient updates when performance
            worsens, by default 0.0005
        memory_bank_size : int, optional
            Number of items to keep in memory bank for calculating loss (from
            MoCo), by default 4096
        temperature : int, optional
            Temperature parameter for NT-Xent loss, by default 0.1.
        extract_features : bool, optional
            If True, forward pass returns model output before penultimate layer.
        """
        super().__init__()

        # Instantiate EfficientNet
        self.model_name = "efficientnet-b0"
        self.backbone = EfficientNet.from_name(
            self.model_name, image_size=img_size, include_top=False)
        self.feature_dim = 1280      # expected feature size from EfficientNetB0

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters()

        # Create MoCo model with EfficientNet backbone
        self.projection_head = MoCoProjectionHead(
            input_dim=self.feature_dim, hidden_dim=self.feature_dim,
            output_dim=128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Set all parameters to disable gradient computation for momentum
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Define loss (NT-Xent Loss with memory bank)
        self.loss = lightly.loss.NTXentLoss(
            temperature=temperature,
            memory_bank_size=memory_bank_size)


    def configure_optimizers(self):
        """
        Initialize and return optimizer (Adam/SGD) and learning rate scheduler.

        Returns
        -------
        tuple of (torch.optim.Optimizer, torch.optim.LRScheduler)
            Contains an initialized optimizer and learning rate scheduler. Each
            are wrapped in a list.
        """
        if self.hparams.adam:
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)

        return optimizer


    ############################################################################
    #                          Per-Batch Metrics                               #
    ############################################################################
    def training_step(self, train_batch, batch_idx):
        """
        Training step

        Parameters
        ----------
        train_batch : tuple
            Contains (img tensor, metadata dict)
        batch_idx : int
            Training batch index

        Returns
        -------
        torch.FloatTensor
            Loss for training batch
        """
        (x_q, x_k), _ = train_batch

        # Update momentum
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum,
                        m=0.99)

        # (For query), extract embedding 
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # Get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        # Calculate loss
        loss = self.loss(q, k)

        self.log("train_loss", loss)

        return loss


    def validation_step(self, val_batch, batch_idx):
        """
        Validation step

        Parameters
        ----------
        val_batch : tuple
            Contains (img tensor, metadata dict)
        batch_idx : int
            Validation batch index

        Returns
        -------
        torch.FloatTensor
            Loss for validation batch
        """
        (x_q, x_k), _ = val_batch

        # Update momentum
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum,
                        m=0.99)

        # (For query), extract embedding 
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # Get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        # Calculate loss
        loss = self.loss(q, k)

        self.log("val_loss", loss)

        return loss


    def test_step(self, test_batch, batch_idx):
        """
        Currently not implemented.
        """
        pass


    ############################################################################
    #                            Epoch Metrics                                 #
    ############################################################################
    def training_epoch_end(self, outputs):
        """
        Compute and log evaluation metrics for training epoch.

        Parameters
        ----------
        outputs: dict
            Dict of outputs of every training step in the epoch
        """
        loss = torch.stack([d['loss'] for d in outputs]).mean()
        self.log('epoch_train_loss', loss)

        # Log weights
        self.custom_histogram_weights()


    def validation_epoch_end(self, outputs):
        """
        Compute and log evaluation metrics for validation epoch.

        Parameters
        ----------
        outputs: dict
            Dict of outputs of every validation step in the epoch
        """
        loss = torch.stack(outputs).mean()
        self.log('epoch_val_loss', loss)


    def custom_histogram_weights(self):
        """
        Log histogram of weights. This is useful for debugging issues with
        dimension collapse, etc.
        """
        for name, params in self.named_parameters():
            self.logger.experiment[1].add_histogram(
                name, params, self.current_epoch)


    ############################################################################
    #                          Extract Embeddings                              #
    ############################################################################
    @torch.no_grad()
    def extract_embeds(self, inputs):
        """
        Extracts embeddings from input images.

        Parameters
        ----------
        inputs : torch.Tensor
            Ultrasound images. Expected size is (B, C, H, W)

        Returns
        -------
        numpy.array
            Deep embeddings before final linear layer
        """
        z = self.backbone(inputs)

        # Flatten
        z = z.view(inputs.size()[0], -1)

        return z.detach().cpu().numpy()
