"""
byol.py

Description: Implementation of BYOL with PyTorch Lightning.
"""

# Standard libraries
import copy

# Non-standard libraries
import lightning as L
import torch
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

# Custom libraries
from src.models.base import NAME_TO_FEATURE_SIZE, load_network, extract_features


################################################################################
#                                  Constants                                   #
################################################################################
# Default parameters
DEFAULT_PARAMS = {
    "model_provider": "timm",
    "model_name": "efficientnet_b0",
    "img_size": (256, 256),
    "optimizer": "adamw",
    "lr": 0.0005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
}


################################################################################
#                               BYOL Model Class                               #
################################################################################
class BYOL(L.LightningModule):
    """
    BYOL class.

    Note
    ----
    Used to perform BYOL pre-training
    """

    def __init__(self, hparams=None, **overwrite_params):
        """
        Initialize BYOL object.

        Parameters
        ----------
        hparams : dict, optional
            Model hyperparameters defined in `config/configspecs/model_training.ini`.
        """
        super().__init__()

        # Add default parameters
        hparams = hparams or {}
        hparams.update({k:v for k,v in DEFAULT_PARAMS.items() if k not in hparams})
        hparams.update(overwrite_params)

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters(hparams)

        # Instantiate backbone
        self.conv_backbone = load_network(hparams, remove_head=True)

        # Create BYOL model with backbone
        feature_dim = NAME_TO_FEATURE_SIZE[hparams["model_name"]]
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


    def configure_optimizers(self):
        """
        Initialize and return optimizer (AdamW or SGD).

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer.
        """
        optimizer = None
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        assert optimizer is not None, f"Optimizer specified is not supported! {self.hparams.optimizer}"
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
    def forward_features(self, inputs):
        """
        Extracts features from input images

        Parameters
        ----------
        inputs : torch.Tensor
            Ultrasound images. Expected size is (B, C, H, W)

        Returns
        -------
        numpy.array
            Deep embeddings
        """
        z = extract_features(self.hparams, self.network, inputs)
        z = z.view(inputs.size()[0], -1)
        return z


    @torch.no_grad()
    def extract_embeds(self, inputs):
        """
        Wrapper over `forward_features` but returns CPU numpy array

        Parameters
        ----------
        inputs : torch.Tensor
            Ultrasound images. Expected size is (B, C, H, W)

        Returns
        -------
        numpy.array
            Deep embeddings
        """
        return self.forward_features(inputs).detach().cpu().numpy()
