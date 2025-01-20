"""
moco.py

Description: Implementation of MoCo using PyTorch Lightning.
"""

# Standard libraries
import copy

# Non-standard libraries
import lightly
import lightning as L
import torch
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (batch_shuffle, batch_unshuffle,
                                  deactivate_requires_grad, update_momentum)

# Custom libraries
from src.loss import SupMoCoLoss
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
    "memory_bank_size": 4096,
    "temperature": 0.1,
    "exclude_momentum_encoder": False,
    "same_label": False,
    "custom_ssl_loss": None,
    "multi_objective": False
}


################################################################################
#                               MoCo Model Class                               #
################################################################################
# TODO: Consider same-patient memory banks to avoid conflict between
# `memory_bank_size` and `same_label` (in SSL data module)
class MoCo(L.LightningModule):
    """
    MoCo class.

    Note
    ----
    Used to perform MoCo pre-training
    """

    def __init__(self, hparams=None, **overwrite_params):
        """
        Initialize MoCo object.

        Parameters
        ----------
        hparams : dict, optional
            Model hyperparameters defined in `config/configspecs/model_training.ini`.
            memory_bank_size : int, optional
                Number of items to keep in memory bank for calculating loss (from
                MoCo), by default 4096
            temperature : int, optional
                Temperature parameter for NT-Xent loss, by default 0.1.
            exclude_momentum_encoder : bool, optional
                If True, uses primary (teacher) encoders for encoding keys. Defaults
                to False.
            same_label : bool, optional
                If True, uses labels to mark same-label samples as positives instead
                of supposed negatives. Defaults to False.
            custom_ssl_loss : str, optional
                One of (None, "same_label"). Specifies custom SSL loss to
                use. Defaults to None.
            multi_objective : bool, optional
                If True, optimizes for both supervised loss and SSL loss. Defaults
                to False.
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

        # Create MoCo model with backbone
        feature_dim = NAME_TO_FEATURE_SIZE[hparams["model_name"]]
        self.projection_head = MoCoProjectionHead(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=128,
        )

        # Momentum Encoders
        self.conv_backbone_momentum = copy.deepcopy(self.conv_backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Set all parameters to disable gradient computation for momentum
        deactivate_requires_grad(self.conv_backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Define loss (NT-Xent Loss with memory bank)
        # 1. NT-Xent loss
        if str(self.hparams.get("custom_ssl_loss")) == "None":
            self.loss = lightly.loss.NTXentLoss(
                temperature=self.hparams.get("temperature", 0.1),
                memory_bank_size=self.hparams.get("memory_bank_size", 4096),
            )
        # 2. Supervised MoCo
        elif self.hparams.get("custom_ssl_loss") == "same_label":
            self.loss = SupMoCoLoss(
                num_classes=self.hparams.get("num_classes", 5),
                temperature=self.hparams.get("temperature", 0.1),
                memory_bank_size=self.hparams.get("memory_bank_size", 4096),
            )
        elif self.hparams.get("custom_ssl_loss") == "soft":
            raise RuntimeError("Soft NT-Xent loss has been deprecated!")
        else:
            raise NotImplementedError(f"`custom_ssl_loss` must be one of (None, "
                                      "'same_label')!")

        # If multi-objective, add supervised loss + classification layer
        if self.hparams.get("multi_objective"):
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")
            self.fc_1 = torch.nn.Linear(feature_dim, self.hparams["num_classes"])

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
        (x_q, x_k), metadata = train_batch

        # Update momentum
        if not self.hparams.get("exclude_momentum_encoder"):
            update_momentum(self.conv_backbone, self.conv_backbone_momentum,
                            m=0.99)
            update_momentum(self.projection_head, self.projection_head_momentum,
                            m=0.99)

        # (For query), extract embedding 
        q = self.conv_backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # Get keys
        k, shuffle = batch_shuffle(x_k)
        if not self.hparams.get("exclude_momentum_encoder"):
            k = self.conv_backbone_momentum(k).flatten(start_dim=1)
            k = self.projection_head_momentum(k)
        else:
            k = self.conv_backbone(k).flatten(start_dim=1)
            k = self.projection_head(k)
        k = batch_unshuffle(k, shuffle)

        # Compute L2 norm of online embeddings
        with torch.no_grad():
            norm = torch.linalg.matrix_norm(q).item()
            norm = norm / len(q)
            self.log("proj_l2_norm", norm)

        # Calculate loss
        # 1. Regular NT-Xent loss
        if str(self.hparams.get("custom_ssl_loss")) == "None":
            loss = self.loss(q, k)
        # 2. Supervised MoCo Loss
        else:
            # Get label
            labels = metadata["label"]
            loss = self.loss(q, k, labels)

        # 2. Optionally, include supervised loss
        if self.hparams.get("multi_objective"):
            # First log SSL loss
            self.log("train_ssl_loss", loss)

            # NOTE: Supervised loss is calculated from both augmented batches
            x = torch.concatenate([q, k])
            x = self.fc_1(x)
            sup_labels = torch.concatenate([metadata["label"]] * 2)

            # Compute supervised loss
            sup_loss = self.cross_entropy(x, sup_labels)
            self.log("train_sup_loss", loss)

            # Add to total loss
            loss += sup_loss

        self.log("train_loss", loss)

        # Prepare result
        self.dset_to_outputs["train"].append({"loss": loss})

        return loss


    def validation_step(self, val_batch, batch_idx):
        """
        Validation step

        Parameters
        ----------
        val_batch : tuple of ((torch.Tensor, torch.Tensor), dict)
            Contains paired (augmented) images and metadata dict.
            Each image tensor is of the shape:
            - (B, 3, H, W)
        batch_idx : int
            Validation batch index

        Returns
        -------
        torch.FloatTensor
            Loss for validation batch
        """
        (x_q, x_k), metadata = val_batch

        # Update momentum
        if not self.hparams.get("exclude_momentum_encoder"):
            update_momentum(self.conv_backbone, self.conv_backbone_momentum,
                            m=0.99)
            update_momentum(self.projection_head, self.projection_head_momentum,
                            m=0.99)

        # (For query), extract embedding 
        q = self.conv_backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # Get keys
        k, shuffle = batch_shuffle(x_k)
        if not self.hparams.get("exclude_momentum_encoder"):
            k = self.conv_backbone_momentum(k).flatten(start_dim=1)
            k = self.projection_head_momentum(k)
        else:
            k = self.conv_backbone(k).flatten(start_dim=1)
            k = self.projection_head(k)
        k = batch_unshuffle(k, shuffle)

        # Calculate loss
        # 1. SSL Loss
        if not self.my_hparams.get("same_label") or self.hparams.get("custom_ssl_loss") is None:
            loss = self.loss(q, k)
        else:
            # Get label
            labels = metadata["label"]
            loss = self.loss(q, k, labels)

        # 2. Optionally, include supervised loss
        if self.hparams.get("multi_objective"):
            # First log SSL loss
            self.log("val_ssl_loss", loss)

            # NOTE: Supervised loss is calculated from both augmented batches
            x = torch.concatenate([q, k])
            x = self.fc_1(x)
            sup_labels = torch.concatenate([metadata["label"]] * 2)

            # Compute supervised loss
            sup_loss = self.cross_entropy(x, sup_labels)
            self.log("val_sup_loss", loss)

            # Add to total loss
            loss += sup_loss

        self.log("val_loss", loss)

        # Prepare result
        self.dset_to_outputs["val"].append({"loss": loss})

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

        # Clean stored output
        self.dset_to_outputs["train"].clear()


    def on_validation_epoch_end(self):
        """
        Compute and log evaluation metrics for validation epoch.
        """
        outputs = self.dset_to_outputs["val"]
        loss = torch.stack([d['loss'] for d in outputs]).mean()
        self.log('epoch_val_loss', loss)

        # Clean stored output
        self.dset_to_outputs["val"].clear()


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
        return self.conv_backbone(inputs).flatten(start_dim=1)


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
