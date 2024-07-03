"""
moco.py

Description: Implementation of MoCo with an EfficientNet convolutional backbone,
             using Lightly.AI and PyTorch Lightning.
"""

# Standard libraries
import copy

# Non-standard libraries
import lightly
import lightning as L
import torch
from efficientnet_pytorch import EfficientNet
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (batch_shuffle, batch_unshuffle, 
                                  deactivate_requires_grad, update_momentum)

# Custom libraries
from src.loss.soft_ntx_ent_loss import SoftNTXentLoss
from src.loss.same_label_con_loss import SameLabelConLoss
from src.utilities import efficientnet_pytorch_utils as effnet_utils


################################################################################
#                               MoCo Model Class                               #
################################################################################
# TODO: Consider same-patient memory banks to avoid conflict between
# `memory_bank_size` and `same_label` (in SSL data module)
class MoCo(L.LightningModule):
    """
    MoCo for self-supervised learning.
    """
    def __init__(self, img_size=(256, 256), optimizer="adamw", lr=0.0005,
                 momentum=0.9, weight_decay=0.0005,
                 memory_bank_size=4096, temperature=0.1,
                 exclude_momentum_encoder=False,
                 same_label=False,
                 custom_ssl_loss=None,
                 multi_objective=False,
                 *args, **kwargs):
        """
        Initialize MoCo object.

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
            One of (None, "soft", "same_label"). Specifies custom SSL loss to
            use. Defaults to None.
        multi_objective : bool, optional
            If True, optimizes for both supervised loss and SSL loss. Defaults
            to False.
        """
        super().__init__()

        # Flag, if same-label sampling
        self.same_label = same_label

        # Instantiate EfficientNet
        self.model_name = "efficientnet-b0"
        self.conv_backbone = EfficientNet.from_name(
            self.model_name, image_size=img_size, include_top=False)
        self.feature_dim = 1280      # expected feature size from EfficientNetB0

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters()

        # Create MoCo model with EfficientNet backbone
        self.projection_head = MoCoProjectionHead(
            input_dim=self.feature_dim, hidden_dim=self.feature_dim,
            output_dim=128)

        # Momentum Encoders
        self.conv_backbone_momentum = copy.deepcopy(self.conv_backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Set all parameters to disable gradient computation for momentum
        deactivate_requires_grad(self.conv_backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Define loss (NT-Xent Loss with memory bank)
        if not self.same_label or self.hparams.custom_ssl_loss is None:
            self.loss = lightly.loss.NTXentLoss(
                temperature=temperature,
                memory_bank_size=memory_bank_size)
        # Below are custom losses for same-label contrastive learning
        elif self.hparams.custom_ssl_loss == "soft":
            # NOTE: With same-label positive sampling, attempts to learn
            #       features, such that any sample of the same label are
            #       equally likely distinguished
            self.loss = SoftNTXentLoss(temperature=temperature)
        elif self.hparams.custom_ssl_loss == "same_label":
            self.loss = SameLabelConLoss()
        else:
            raise NotImplementedError(f"`custom_ssl_loss` must be one of (None, "
                                      "'soft', 'same_label')!")

        # If multi-objective, add supervised loss + classification layer
        if self.hparams.multi_objective:
            # Define supervisd loss
            self.cross_entropy_loss = torch.nn.CrossEntropy(reduction="mean")
            # Define classification layer
            num_classes = kwargs.get("num_classes", 5)
            self.fc_1 = torch.nn.Linear(self.feature_dim, num_classes)

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
        if not self.hparams.exclude_momentum_encoder:
            update_momentum(self.conv_backbone, self.conv_backbone_momentum,
                            m=0.99)
            update_momentum(self.projection_head, self.projection_head_momentum,
                            m=0.99)

        # (For query), extract embedding 
        q = self.conv_backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # Get keys
        k, shuffle = batch_shuffle(x_k)
        if not self.hparams.exclude_momentum_encoder:
            k = self.conv_backbone_momentum(k).flatten(start_dim=1)
            k = self.projection_head_momentum(k)
        else:
            k = self.conv_backbone(k).flatten(start_dim=1)
            k = self.projection_head(k)
        k = batch_unshuffle(k, shuffle)

        # Calculate loss
        # 1. SSL loss
        if not self.same_label or self.hparams.custom_ssl_loss is None:
            loss = self.loss(q, k)
        else:
            # Get label
            labels = metadata["label"]
            loss = self.loss(q, k, labels)

        # 2. Optionally, include supervised loss
        if self.hparams.multi_objective:
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
        if not self.hparams.exclude_momentum_encoder:
            update_momentum(self.conv_backbone, self.conv_backbone_momentum,
                            m=0.99)
            update_momentum(self.projection_head, self.projection_head_momentum,
                            m=0.99)

        # (For query), extract embedding 
        q = self.conv_backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # Get keys
        k, shuffle = batch_shuffle(x_k)
        if not self.hparams.exclude_momentum_encoder:
            k = self.conv_backbone_momentum(k).flatten(start_dim=1)
            k = self.projection_head_momentum(k)
        else:
            k = self.conv_backbone(k).flatten(start_dim=1)
            k = self.projection_head(k)
        k = batch_unshuffle(k, shuffle)

        # Calculate loss
        # 1. SSL Loss
        if not self.same_label or self.hparams.custom_ssl_loss is None:
            loss = self.loss(q, k)
        else:
            # Get label
            labels = metadata["label"]
            loss = self.loss(q, k, labels)

        # 2. Optionally, include supervised loss
        if self.hparams.multi_objective:
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

        # Log weights
        self.custom_histogram_weights()


    def on_validation_epoch_end(self):
        """
        Compute and log evaluation metrics for validation epoch.
        """
        outputs = self.dset_to_outputs["val"]
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
        z = self.conv_backbone(inputs)

        # Flatten
        z = z.view(inputs.size()[0], -1)

        return z.detach().cpu().numpy()
