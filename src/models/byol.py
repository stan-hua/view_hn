"""
byol.py

Description: Implementation of BYOL with an EfficientNet convolutional backbone,
             using Lightly.AI and PyTorch Lightning.
"""

# Standard libraries
import copy

# Non-standard libraries
import lightning as L
import torch
from efficientnet_pytorch import EfficientNet
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

# Custom libraries
from src.utils import efficientnet_pytorch_utils as effnet_utils


################################################################################
#                               BYOL Model Class                               #
################################################################################
class BYOL(L.LightningModule):
    """
    BYOL for self-supervised learning.
    """
    def __init__(self, img_size=(256, 256),
                 optimizer="adamw", lr=0.05,
                 momentum=0.9, weight_decay=0.0005,
                 effnet_name="efficientnet-b0",
                 *args, **kwargs):
        """
        Initialize BYOL object.

        Parameters
        ----------
        img_size : tuple, optional
            Expected image's (height, width), by default (256, 256)
        optimizer : str, optional
            Choice of optimizer, by default "adamw"
        lr : float, optional
            Optimizer learning rate, by default 0.0001
        momentum : float, optional
            If SGD optimizer, value to use for momentum during SGD, by
            default 0.9
        weight_decay : float, optional
            Weight decay value to slow gradient updates when performance
            worsens, by default 0.0005
        effnet_name : str, optional
            Name of EfficientNet backbone to use
        """
        super().__init__()

        # Instantiate EfficientNet
        self.model_name = effnet_name
        self.conv_backbone = EfficientNet.from_name(
            self.model_name, image_size=img_size, include_top=False)
        self.feature_dim = 1280      # expected feature size from EfficientNetB0

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters()

        # Create BYOL model with EfficientNet backbone
        self.projection_head = BYOLProjectionHead(
            input_dim=self.feature_dim, hidden_dim=2*self.feature_dim,
            output_dim=256)
        self.prediction_head = BYOLPredictionHead(256, 4096, 256)

        # Momentum Encoders
        self.conv_backbone_momentum = copy.deepcopy(self.conv_backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Set all parameters to disable gradient computation for momentum
        deactivate_requires_grad(self.conv_backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Define loss (NT-Xent Loss with memory bank)
        self.loss = NegativeCosineSimilarity()

        # Store outputs
        self.dset_to_outputs = {"train": [], "val": [], "test": []}


    def load_imagenet_weights(self):
        """
        Load imagenet weights for convolutional backbone.
        """
        # NOTE: Modified utility function to ignore missing keys
        effnet_utils.load_pretrained_weights(
            self.conv_backbone, self.model_name,
            load_fc=False,
            advprop=False)


    def configure_optimizers(self):
        """
        Initialize and return optimizer (AdamW or SGD).

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer.
        """
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer


    ############################################################################
    #                       Custom BYOL Forward Pass                           #
    ############################################################################
    def byol_forward(self, x):
        """
        Perform and return forward pass with online network

        Parameters
        ----------
        x : torch.Tensor
            Batch of images
        """
        # Embed and project image
        y = self.conv_backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)

        # Predict features of EMA target network
        p = self.prediction_head(z)
        return p


    def byol_forward_momentum(self, x):
        """
        Perform and return forward pass with target network

        Parameters
        ----------
        x : torch.Tensor
            Batch of images
        """
        # Embed and project image
        y = self.conv_backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)

        # Return detached features of EMA target network
        z = z.detach()
        return z


    ############################################################################
    #                          Per-Batch Metrics                               #
    ############################################################################
    def training_step(self, train_batch, batch_idx):
        """
        Training step

        Parameters
        ----------
        train_batch : tuple of ((torch.Tensor, torch.Tensor), dict)
            Contains paired (augmented) images and metadata dict.
            Each image tensor is of the shape:
            - (B, 3, H, W)
        batch_idx : int
            Training batch index

        Returns
        -------
        torch.FloatTensor
            Loss for training batch
        """
        (x_q, x_k), _ = train_batch

        # Get momentum
        momentum = cosine_schedule(self.current_epoch, self.hparams.get("stop_epoch", 600), 0.996, 1)

        # Update target network parameters
        update_momentum(self.conv_backbone, self.conv_backbone_momentum,
                        m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum,
                        m=momentum)

        # Pass through online network
        p0 = self.byol_forward(x_q)
        p1 = self.byol_forward(x_k)

        # Pass through target network
        z0 = self.byol_forward_momentum(x_q)
        z1 = self.byol_forward_momentum(x_k)

        # Compute loss
        # NOTE: Goal is to minimize the difference between online/target network
        #       representations for the same image
        loss = 0.5 * (self.loss(p0, z1) + self.loss(p1, z0))
        self.log("train_loss", loss)

        # Compute L2 norm of online embeddings
        with torch.no_grad():
            embeds = torch.concat([p0, p1])
            norm = torch.linalg.matrix_norm(embeds).item()
            norm = norm / len(embeds)
            self.log("proj_l2_norm", norm)

        # Prepare result
        self.dset_to_outputs["train"].append({"loss": loss})

        return loss


    ############################################################################
    #                            Epoch Metrics                                 #
    ############################################################################
    def on_train_epoch_end(self):
        """
        Compute and log evaluation metrics for training epoch.
        """
        outputs = self.dset_to_outputs["train"]
        loss = torch.stack([d['loss'] for d in outputs]).mean()
        self.log('epoch_train_loss', loss)


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
        z = self.conv_backbone(inputs)

        # Flatten
        z = z.view(inputs.size()[0], -1)

        return z.detach().cpu().numpy()
