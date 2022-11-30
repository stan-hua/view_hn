"""
tclr.py

Description: Implementation of TCLR (Temporal Contrastive Learning) with an
             EfficientNet convolutional backbone + LSTM for learning
             temporal information, using Lightly.AI and PyTorch Lightning.
"""

# Non-standard libraries
import numpy as np
import lightly
import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet
from lightly.models.modules.heads import SimCLRProjectionHead


################################################################################
#                               TCLR Model Class                               #
################################################################################
class TCLR(pl.LightningModule):
    """
    TCLR class for self-supervised pretraining
    """

    def __init__(self,
                 subclip_size=3, img_size=(256, 256),
                 n_lstm_layers=1, hidden_dim=512, bidirectional=False,
                 adam=True, lr=0.0005, momentum=0.9, weight_decay=0.0005,
                 temperature=0.1, extract_features=False, *args, **kwargs):
        """
        Initialize TCLR object

        Parameters
        ----------
        subclip_size : int, optional
            Number of image frames (in a clip) to form a sub-clip to use in the
            self-supervised tasks, by default 3.
        img_size : tuple, optional
            Expected image's (height, width), by default (256, 256)
        n_lstm_layers : int, optional
            Number of LSTM layers, by default 1
        hidden_dim : int, optional
            Dimension/size of hidden layers, by default 512
        bidirectional : bool, optional
            If True, trains a bidirectional LSTM, by default False
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
        temperature : int, optional
            Temperature parameter for NT-Xent loss, by default 0.1.
        extract_features : bool, optional
            If True, forward pass returns model output before penultimate layer.
        """
        super().__init__()

        self.model_name = "efficientnet-b0"
        self.feature_dim = 1280      # expected feature size from EfficientNetB0

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters()

        # Instantiate convolutional backbone
        # TODO: Consider converting batch norm. into layer norm.?
        self.conv_backbone = EfficientNet.from_name(
            self.model_name, image_size=img_size, include_top=False)

        # Instantiate temporal backbone
        self.temporal_backbone = torch.nn.LSTM(
            self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Instantiate projection head
        self.projection_head = SimCLRProjectionHead(
            input_dim=lstm_output_dim,
            hidden_dim=self.feature_dim,
            output_dim=128)

        # Define loss (NT-Xent Loss)
        # NOTE: If memory bank size > 0, may lead to negative sample spillage
        # NOTE: This would interfere with the TCLR losses
        self.loss = lightly.loss.NTXentLoss(temperature=temperature)


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

        Notes
        -----
        Groups clips to compute the 3 contrastive losses from TCLR:
            (1) Instance-based,
            (2) Local-Local, and
            (3) Global-Local

        Parameters
        ----------
        train_batch : tuple of ((torch.Tensor, torch.Tensor), dict)
            Contains paired (augmented) image clips and metadata dict.
            Each tensor of image clips is of the shape:
            - (B, clip_size, 3, H, W)
        batch_idx : int
            Training batch index

        Returns
        -------
        torch.FloatTensor
            Loss for training batch
        """
        (x_q, x_k), _ = train_batch

        # Get input dimensions
        B, clip_size, _, _, _ = x_q.size()

        # Extract convolutional features for each clip
        # NOTE: Output size: (B, `seq_length`, 1280, 1, 1)
        x_q_features = torch.stack([self.conv_backbone(clip) for clip in x_q])
        x_k_features = torch.stack([self.conv_backbone(clip) for clip in x_k])

        # Remove extra dimensions
        x_q_features = x_q_features.view(B, clip_size, -1)
        x_k_features = x_k_features.view(B, clip_size, -1)

        # (1) Instance-based contrastive loss
        # NOTE: Disabled instance-based contrastive loss
        instance_loss = 0
        # instance_loss = self.compute_instance_loss(x_q_features)

        # (2) Local-Local contrastive loss
        local_loss = self.compute_local_local_loss(x_q_features, x_k_features)

        # (3) Global-Local contrastive loss
        global_loss = self.compute_global_local_loss(x_q_features)

        # Accumulate losses
        loss = instance_loss + local_loss + global_loss

        self.log("train_loss", loss, batch_size=B)

        # Calculate mean of SD along each dimension for spatiel embeddings
        # NOTE: To monitor potential collapse of learned representations
        x_q_flattened = x_q_features.view(B, -1)
        collapse_metric = lightly.utils.debug.std_of_l2_normalized(
            x_q_flattened)
        self.log("train_collapse_metric", collapse_metric, batch_size=B)
        self.log("ideal_collapse_metric", 1 / np.sqrt(x_q_flattened.size()[-1]),
                 batch_size=B)

        return loss


    def validation_step(self, val_batch, batch_idx):
        """
        Validation step

        Parameters
        ----------
        val_batch : tuple of ((torch.Tensor, torch.Tensor), dict)
            Contains paired (augmented) image clips and metadata dict.
            Each tensor of image clips is of the shape:
            - (B, clip_size, 3, H, W)
        batch_idx : int
            Validation batch index

        Returns
        -------
        torch.FloatTensor
            Loss for validation batch
        """
        (x_q, x_k), _ = val_batch

        # Get input dimensions
        B, clip_size, _, _, _ = x_q.size()

        # Extract convolutional features for each clip
        # NOTE: Output size: (B, `seq_length`, 1280, 1, 1)
        x_q_features = torch.stack([self.conv_backbone(clip) for clip in x_q])
        x_k_features = torch.stack([self.conv_backbone(clip) for clip in x_k])

        # Remove extra dimensions
        x_q_features = x_q_features.view(B, clip_size, -1)
        x_k_features = x_k_features.view(B, clip_size, -1)

        # (1) Instance-based contrastive loss
        # NOTE: Disabled instance-based contrastive loss
        instance_loss = 0
        # instance_loss = self.compute_instance_loss(x_q_features)

        # (2) Local-Local contrastive loss
        local_loss = self.compute_local_local_loss(x_q_features, x_k_features)

        # (3) Global-Local contrastive loss
        global_loss = self.compute_global_local_loss(x_q_features)

        # Accumulate losses
        loss = instance_loss + local_loss + global_loss

        self.log("val_loss", loss, batch_size=B)

        return loss


    def compute_instance_loss(self, x_q):
        """
        Calculate (1) instance-based loss, incorporating temporal information
        over latent spatial representations.

        Note
        ----
        Attempts to match a subclip to a subclip from the same clip.

        In this case, it isn't important to have two augmentations of the same
        clip, since we compare different different non-overlapping subclips.

        Parameters
        ----------
        x_q : torch.Tensor
            Spatial embeddings for query subclips

        Returns
        -------
        torch.Float
            Instance-based contrastive loss
        """
        # Get input dimensions
        B, clip_size, _ = x_q.size()

        # V: How many subclips per clip
        subclip_size = self.hparams.subclip_size
        V = clip_size // subclip_size

        # Choose non-overlapping subclips to match
        first_idx, second_idx = np.random.choice(range(V), size=2,
                                                 replace=False)
        first_idx *= subclip_size
        second_idx *= subclip_size

        # Get different non-overlapping subclips from the same clips
        # NOTE: Not important whether taken from x_q or x_k
        instance_x_q = x_q[:, first_idx: first_idx+subclip_size, :]
        instance_x_k = x_q[:, second_idx: second_idx+subclip_size, :]

        # (For query), extract temporally-aware embeddings
        # NOTE: Average over frames in subclip to create subclip representation
        q = self.temporal_backbone(instance_x_q)[0]
        q = torch.mean(q, dim=1)
        q = self.projection_head(q)

        # (For keys), extract temporally-aware embeddings
        # NOTE: Average over frames in subclip to create subclip representation
        k = self.temporal_backbone(instance_x_k)[0]
        k = torch.mean(k, dim=1)
        k = self.projection_head(k)

        # Calculate loss
        loss = self.loss(q, k)

        # Normalize by number of comparisons (batch size)
        loss /= B

        return loss


    def compute_local_local_loss(self, x_q, x_k):
        """
        Calculate (2) local-local loss, incorporating temporal information
        over latent spatial representations.

        Note
        ----
        Attempts to match a subclip to the same subclip (augmented differently)
        against other subclips from the same clip.

        Preconditions
        -------------
            a) There are at least (2 * `subclip_size`) frames in a clip.
            b) The clip size is perfectly divisible by `subclip_size`.

        Parameters
        ----------
        x_q : torch.Tensor
            Spatial embeddings for query subclips
        x_k : torch.Tensor
            Spatial embeddings for key subclips

        Returns
        -------
        torch.Float
            Local-local loss
        """
        # Get input dimensions
        B, clip_size, _ = x_q.size()

        # V: How many subclips per clip
        subclip_size = self.hparams.subclip_size
        V = clip_size // subclip_size

        # Divide clip equally into V subclips of size `subclip_size`
        # NOTE: Each item is of shape: (B, `subclip_size`, 1280)
        x_q__batch_subclips = torch.split(x_q, subclip_size, dim=1)
        x_k__batch_subclips = torch.split(x_k, subclip_size, dim=1)

        # Concatenate temporarily to parallelize temporal feature extraction
        x_q__all_subclips = torch.cat(x_q__batch_subclips, dim=0)
        x_k__all_subclips = torch.cat(x_k__batch_subclips, dim=0)

        # (For query), extract temporally-aware embeddings
        # NOTE: Average over frames in subclip to create subclip representation
        q = self.temporal_backbone(x_q__all_subclips)[0]
        q = torch.mean(q, dim=1)
        q = self.projection_head(q)

        # (For keys), extract temporally-aware embeddings
        # NOTE: Average over frames in subclip to create subclip representation
        k = self.temporal_backbone(x_k__all_subclips)[0]
        k = torch.mean(k, dim=1)
        k = self.projection_head(k)

        # Split once again, to retrieve batched subclips
        q__batch_subclips = torch.split(q, V, dim=0)
        k__batch_subclips = torch.split(k, V, dim=0)
        # Recombine to get subclips (representations) for each clip
        # NOTE: Output size: (B, V, 128)
        q__clip_subclips = torch.stack(q__batch_subclips, dim=0)
        k__clip_subclips = torch.stack(k__batch_subclips, dim=0)

        # Calculate loss w.r.t subclips from the same clip
        loss = 0
        for clip_idx in range(len(q__clip_subclips)):
            # Get all subclips from the same clip
            q__subclips = q__clip_subclips[clip_idx]
            k__subclips = k__clip_subclips[clip_idx]
            loss += self.loss(q__subclips, k__subclips)

        # Normalize by number of comparisons (batch size * num. of subclips)
        loss /= B * V

        return loss


    def compute_global_local_loss(self, x_q):
        """
        Calculate (3) global-local loss, incorporating temporal information
        over latent spatial representations.

        Note
        ----
        Attempts to match a local subclip (not knowing other subclips) to the
        corresponding global subclip (knowing other subclips), against other
        local and global subclips.

        In this case, it isn't important to have two augmentations of the same
        clip, since we compare different different non-overlapping subclips.

        Preconditions
        -------------
        Assumes:
            a) there are at least (2 * `subclip_size`) frames in a clip
            b) the clip size is perfectly divisible by `subclip_size`

        Parameters
        ----------
        x_q : torch.Tensor
            Spatial embeddings for query subclips

        Returns
        -------
        torch.Float
            Global-local loss
        """
        # Get input dimensions
        B, clip_size, _ = x_q.size()

        # V: How many subclips per clip
        subclip_size = self.hparams.subclip_size
        V = clip_size // subclip_size

        # A. Create local representations
        # Divide clip equally into V subclips of size `subclip_size`
        x_q__batch_subclips = torch.split(x_q, subclip_size, dim=1)
        # Concatenate temporarily to parallelize temporal feature extraction
        x_q__all_subclips = torch.cat(x_q__batch_subclips, dim=0)

        # (For query), extract (local) temporally-aware embeddings
        # NOTE: Average over frames in subclip to create local subclip
        #       representation
        q = self.temporal_backbone(x_q__all_subclips)[0]
        q = torch.mean(q, dim=1)
        q = self.projection_head(q)

        # Split once again, to retrieve batched subclips
        q__batch_subclips = torch.split(q, V, dim=0)
        # Recombine to get subclips (representations) for each clip
        # NOTE: Output size: (B, V, 128)
        q__clip_local_subclips = torch.stack(q__batch_subclips, dim=0)

        # B. Create global representations
        # (For keys), extract (global) temporally-aware embeddings
        k = self.temporal_backbone(x_q)[0]

        # Average over subclips
        k__batch_subclips = torch.split(k, subclip_size, dim=1)
        k__batch_subclips = torch.stack(k__batch_subclips, dim=1)
        k = torch.mean(k__batch_subclips, dim=2)

        # Extract MLP embeddings
        k__clip_global_subclips = self.projection_head(k)

        # Calculate loss w.r.t subclips from the same clip
        loss = 0
        for clip_idx in range(len(q__clip_local_subclips)):
            # Get all subclips from the same clip
            q__local_subclips = q__clip_local_subclips[clip_idx]
            k__global_subclips = k__clip_global_subclips[clip_idx]
            loss += self.loss(q__local_subclips, k__global_subclips)

        # Normalize by number of comparisons (batch size * num. of subclips)
        loss /= B * V

        return loss


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
            Ultrasound images from one patient. Expected size is (B, C, H, W)

        Returns
        -------
        numpy.array
            Embeddings after CNN+LSTM
        """
        # Get dimensions
        dims = inputs.size()

        # Extract convolutional features
        z = self.conv_backbone(inputs)

        # Flatten
        z = z.view(1, dims[0], -1)

        # Extract temporal features
        c = self.temporal_backbone(z)[0]

        return c.detach().cpu().numpy()
