"""
tcl.py

Description: Twin Contrastive Learning for Noisy Labels implementation

Source Code: https://github.com/Hzzone/TCL/
"""

# Standard libraries
import copy
import logging

# Non-standard libraries
import numpy as np
import lightly
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from efficientnet_pytorch import EfficientNet
from lightly.models.utils import (batch_shuffle, batch_unshuffle,
                                  deactivate_requires_grad, update_momentum)
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Custom libraries
from src.utils import efficientnet_pytorch_utils as effnet_utils
from src.data import constants


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)


################################################################################
#                                TCL Model Class                               #
################################################################################
class TCL(L.LightningModule):
    """
    Twin Contrastive Learning for Noisy Labels.
    """

    def __init__(self, **hparams):
        """
        Initialize MoCo object.

        Parameters
        ----------
        **hparams : Any
            Keyword arguments including:
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
                Temperature parameter for losses, by default 0.1.
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
            effnet_name : str, optional
                Name of EfficientNet backbone to use
            warmup_epochs : int
                Number of warmup epochs
        """
        super().__init__()

        # Instantiate EfficientNet
        self.model_name = hparams.get("effnet_name", "efficientnet-b0")
        self.conv_backbone = EfficientNet.from_name(
            self.model_name,
            image_size=hparams.get("img_size", constants.IMG_SIZE),
            include_top=False)
        self.feature_dim = 1280      # expected feature size from EfficientNetB0
        self.head_hidden_dim = 128

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters()

        # A. Projection Head
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.BatchNorm1d(self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, self.head_hidden_dim),
        )

        # B. Prediction Head
        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.head_hidden_dim),
            torch.nn.BatchNorm1d(self.head_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.head_hidden_dim, self.hparams["num_classes"]),
        )

        # Momentum Encoders
        self.conv_backbone_momentum = copy.deepcopy(self.conv_backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Set all parameters to disable gradient computation for momentum
        deactivate_requires_grad(self.conv_backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Define losses
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.contrastive_loss = lightly.loss.NTXentLoss(
            temperature=self.hparams.temperature,
            memory_bank_size=(self.hparams.memory_bank_size, self.head_hidden_dim),
        )

        # Store metrics, just for validation set
        # NOTE: Split validation set into clean and noisy labels based on GMM
        self.dset_to_acc = torch.nn.ModuleDict({
            dset: torchmetrics.Accuracy(
                num_classes=self.hparams.num_classes,
                task='multiclass')
            for dset in ("val", "val_clean", "val_noisy")
        })

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
        dict
            Contains optimizer and LR scheduler
        """
        # Create optimizer
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)

        # Prepare return
        ret = {
            "optimizer": optimizer,
        }

        # Set up LR Scheduler
        if self.hparams.get("lr_scheduler") == "cosine_annealing":
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            ret["lr_scheduler"] = lr_scheduler

        return ret


    def init_buffers(self, train_set_size):
        """
        Initialize buffers.

        Parameters
        ----------
        train_set_size : int
            Number of samples in training set
        """
        # Stored attributes, updated at the start of every epoch
        # 1. Cluster means
        self.register_buffer(
            "cluster_means",
            torch.randn((self.hparams["num_classes"], self.head_hidden_dim),
                        device=self.device),
        )
        # 2. Clean label probabilities
        self.register_buffer(
            "is_clean_prob",
            torch.rand((train_set_size,), device=self.device),
        )


    ############################################################################
    #                             Epoch Hooks                                  #
    ############################################################################
    @torch.no_grad()
    def on_train_epoch_start(self):
        """
        On train epoch start, perform Expectation Step:
            (i) Assign pseudo-label to each item in the dataset using GMM, and
            (ii) Use 2nd GMM to determine if each sample has a clean/noisy label
        """
        train_dataloader = self.trainer.train_dataloader
        train_set_size = len(train_dataloader.dataset)

        # If first epoch, initialize buffer
        if self.current_epoch == 0:
            self.init_buffers(train_set_size)

        # For each of the weakly augmented images,
        # 1. Use classifier head to get cluster predictions
        # 2. Use projector head to get normalized features
        # NOTE: In the original paper, they use train images w/o augmentation
        # NOTE: Below tensors are on CPU only
        features = torch.zeros((train_set_size, self.head_hidden_dim))
        labels = torch.zeros((train_set_size, )).to(int)
        cluster_labels = torch.zeros((train_set_size, self.hparams["num_classes"])).to(int)
        for (x_weak, _, _), metadata in train_dataloader:
            x_weak = x_weak.to(self.device)
            dataset_idx = metadata["dataset_idx"].to(int)
            labels[dataset_idx] = metadata["label"].to(int)
            with torch.no_grad():
                out = self.conv_backbone(x_weak).flatten(start_dim=1)
                # Get cluster label
                cluster_labels[dataset_idx] = F.softmax(self.prediction_head(out), dim=1).detach().cpu()
                # Get normalized features
                features[dataset_idx] = F.normalize(self.projection_head(out)).detach().cpu()

        # Use GMM to detect if label is clean/noisy
        ret = self.gmm_get_noisy_labels(features, labels, cluster_labels)

        # Modify buffers
        self.cluster_means[:] = ret["cluster_means"]
        self.is_clean_prob[:] = ret["is_clean_prob"]


    ############################################################################
    #                           Loss Computation                               #
    ############################################################################
    # TODO: Consider MixUp on contrastive loss
    def compute_contrastive_loss(self, train_batch):
        """
        Compute contrastive loss using projection head.

        Parameters
        ----------
        train_batch : tuple of ((torch.Tensor, torch.Tensor, torch.Tensor), dict)
            (i) (weak, strong, strong) augmented views of the same image
            (ii) unused metadata dictionary 

        Returns
        -------
        torch.FloatTensor
            Contrastive InfoNCE (NT-Xent) loss with MoCo memory bank trick
        """
        (_, x_q, x_k), _ = train_batch

        # Update EMA models
        update_momentum(self.conv_backbone, self.conv_backbone_momentum,
                        m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum,
                        m=0.99)

        # (For query), extract embedding
        q = self.conv_backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # Get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.conv_backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        # Compute L2 norm of online embeddings
        with torch.no_grad():
            norm = torch.linalg.matrix_norm(q).item()
            norm = norm / len(q)
            self.log("proj_l2_norm", norm)

        # Calculate contrastive loss
        con_loss = self.contrastive_loss(q, k)
        return con_loss


    def compute_classifier_alignment_and_entropy_loss(self, train_batch):
        """
        Compute (i) classifier loss, (ii) alignment loss and (iii) entropy
        regularization loss.

        Parameters
        ----------
        train_batch : tuple of ((torch.Tensor, torch.Tensor, torch.Tensor), dict)
            (i) (weak, strong, strong) augmented views of the same image
            (ii) metadata dictionary

        Returns
        -------
        torch.FloatTensor
            Loss for training batch
        """
        # _weak = Weakly augmented view. Later _m = MixUp view
        # _1, _2 = Strongly augmented views
        (x_weak, x_1, x_2), metadata = train_batch
        labels = metadata["label"].to(int)
        dataset_idx = metadata["dataset_idx"].to(int)

        # Get label weight (how clean is each label?)
        is_clean_prob = self.is_clean_prob[dataset_idx]
        # Get cluster centres/prototypes
        cluster_means = self.cluster_means

        # Make noisy labels one-hot encoded
        one_hot_labels = F.one_hot(labels, num_classes=self.hparams["num_classes"])

        # Perform mix-up on all views
        x_mixup, mixup_idx, mixup_coef = mixup(
            torch.cat([x_weak, x_1, x_2]),
            alpha=self.hparams.get("mixup_alpha", 1.0),
        )

        # Get number of samples for each of (x_weak, x_1, x_2, x_m)
        x_lst = [x_weak, x_1, x_2, x_mixup]
        lengths = [x_i.size(0) for x_i in x_lst]

        # Extract convolutional features
        # out = torch.cat([self.conv_backbone(x).flatten(start_dim=1) for x in x_lst])
        out = self.conv_backbone(torch.cat(x_lst)).flatten(start_dim=1)

        # 1. Predict cluster/class (logits)
        logit_weak, logit_1, logit_2, logit_mixup = self.prediction_head(out).split(lengths)
        logits = torch.cat([logit_weak, logit_1, logit_2, logit_mixup])
        _, y_1, y_2, y_mixup = F.softmax(logits, dim=1).split(lengths)

        # 2. Project features for MixUp images
        out_mixup = out.split(lengths)[3]
        projected_mixup = self.projection_head(out_mixup)

        # Compute cluster assignment (logits) for MixUp images
        # Shape: (3*N, K)
        mixup_cluster_logits = projected_mixup.mm(cluster_means.T) / self.hparams["temperature"]

        # Create targets for cross-supervision and alignment loss
        # NOTE: Stop gradient here to prevent prediction collapse
        with torch.no_grad():
            # CASE 1: Warm-Up Epochs; use noisy label
            if self.current_epoch < self.hparams["warmup_epochs"]:
                # [Cross-Supervision Loss]
                t_1 = t_2 = labels

                # [Alignment Loss]
                t_mixup = one_hot_labels.repeat((3, 1))
                t_mixup = interpolate(t_mixup, t_mixup[mixup_idx], mixup_coef)
            # CASE 2: Past warm-up; use estimated labels
            else:
                # [Cross-Supervision Loss]
                # Create targets; convex combination between noisy and predicted label
                t_1 = interpolate(one_hot_labels, y_1, is_clean_prob)
                t_2 = interpolate(one_hot_labels, y_2, is_clean_prob)

                # [Alignment Loss]
                # Create targets for alignment loss (for MixUp images only)
                # Labels := avg. estimated labels for mixed up image pairs
                t_bar = (t_1 + t_2).mean()
                # Repeat 3 time since `len(x_mixup) = 3 * len(x_weak)`
                t_bar = t_bar.repeat((3, 1))
                # Interpolate with corresponding mixed up images
                t_mixup = interpolate(t_bar, t_bar[mixup_idx], mixup_coef)

        # 1. Cross-Supervision Loss (Continued)
        cross_sup_loss = self.ce_loss(logit_1, t_1) + self.ce_loss(logit_2, t_2)

        # 2. Alignment Loss (Continued)
        align_loss = self.ce_loss(mixup_cluster_logits, t_mixup)

        # 3. Entropy Regularization Loss
        entropy_loss = self.compute_entropy_loss(torch.cat([y_1, y_2, y_mixup]))

        return cross_sup_loss, align_loss, entropy_loss


    def compute_entropy_loss(self, y_pred):
        """
        Compute entropy regularization loss

        Note
        -----
        Original implementation may be wrong. See
        https://github.com/Hzzone/TCL/blob/master/models/tcl/tcl_wrapper.py#L80

        Parameters
        ----------
        y_pred : torch.Tensor
            Each row contains normalized class probabilities

        Returns
        -------
        torch.FloatTensor
            Entropy regularization loss
        """
        # Maximize entropy across predictions; to prevent prediction collapse
        entropy_loss_1 = -compute_entropy(y_pred.mean(dim=0))

        # Minimize entropy for each prediction; to have more confident examples
        entropy_loss_2 = compute_entropy(y_pred, dim=1).mean()

        return entropy_loss_1 + entropy_loss_2


    ############################################################################
    #                         Per-Batch Operations                             #
    ############################################################################
    def training_step(self, train_batch, batch_idx):
        """
        Training step

        Parameters
        ----------
        train_batch : tuple of ((torch.Tensor, torch.Tensor, torch.Tensor), dict)
            (i) (weak, strong, strong) augmented views of the same image
            (ii) metadata dictionary
        batch_idx : int
            Training batch index

        Returns
        -------
        torch.FloatTensor
            Loss for training batch
        """
        # Compute
        # 1. Cross-supervision loss
        # 2. Alignment loss, and
        # 3. Entropy regularization loss
        cross_sup_loss, align_loss, entropy_loss = \
            self.compute_classifier_alignment_and_entropy_loss(train_batch)
        # 4. Contrastive Loss
        con_loss = self.compute_contrastive_loss(train_batch)

        # Compute loss altogether
        loss = cross_sup_loss + entropy_loss + con_loss + align_loss

        # Prepare result
        self.dset_to_outputs["train"].append({"loss": loss})

        return loss


    def validation_step(self, val_batch, batch_idx):
        """
        Validation step

        Parameters
        ----------
        val_batch : tuple of ((torch.Tensor, torch.Tensor, torch.Tensor), dict)
            (i) (weak, strong, strong) augmented views of the same image
            (ii) metadata dictionary
        batch_idx : int
            Training batch index

        Returns
        -------
        torch.FloatTensor
            Loss for training batch
        """
        # Compute validation loss
        losses = []
        losses.append(self.compute_contrastive_loss(val_batch))
        losses.extend(self.compute_classifier_alignment_and_entropy_loss(val_batch))
        loss = sum(losses)

        # Compute validation accuracies on weakly augmented images
        (x_weak, _, _), metadata = val_batch
        labels = metadata["label"].to(int)

        # Get cluster labels and extract features
        with torch.no_grad():
            out = self.conv_backbone(x_weak).flatten(start_dim=1)
            # Get cluster label
            cluster_labels = F.softmax(self.prediction_head(out), dim=1).detach()
            # Get normalized features
            features = F.normalize(self.projection_head(out)).detach()

        # Use GMM to detect if label is clean/noisy
        ret = self.gmm_get_noisy_labels(features, labels, cluster_labels)
        is_clean_mask = (ret["is_clean_prob"] >= 0.5)

        # Compute validation accuracy
        self.dset_to_acc["val"].update(cluster_labels, labels)

        # Stratify by clean vs. noisy labels
        if is_clean_mask.sum():
            self.dset_to_acc["val_clean"].update(cluster_labels[is_clean_mask], labels[is_clean_mask])
        if (~is_clean_mask).sum():
            self.dset_to_acc["val_noisy"].update(cluster_labels[~is_clean_mask], labels[~is_clean_mask])

        # Prepare result
        ret = {
            "loss": loss.detach().cpu(),
            "y_pred": cluster_labels.detach().cpu(),
            "y_true": labels.detach().cpu(),
        }
        self.dset_to_outputs["val"].append(ret)

        return loss


    ############################################################################
    #                          Per-Epoch Metrics                               #
    ############################################################################
    def on_train_epoch_end(self):
        """
        Compute and log evaluation metrics for training epoch.
        """
        outputs = self.dset_to_outputs["train"]
        loss = torch.stack([d['loss'] for d in outputs]).mean()
        self.log('epoch_train_loss', loss, batch_size=self.hparams.get("batch_size"))

        # Clean stored output
        self.dset_to_outputs["train"].clear()


    def on_validation_epoch_end(self):
        """
        Compute and log evaluation metrics for validation epoch.
        """
        outputs = self.dset_to_outputs["val"]
        # Stre loss
        loss = torch.tensor([o["loss"] for o in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True)

        # Compute accuracies
        for acc_key in ("val", "val_clean", "val_noisy"):
            acc = self.dset_to_acc[acc_key].compute()
            self.log(f'{acc_key}_acc', acc, prog_bar=True)
            self.dset_to_acc[acc_key].reset()

        # Create confusion matrix
        if self.hparams.get("use_comet_logger"):
            self.logger.experiment.log_confusion_matrix(
                y_true=torch.cat([o["y_true"] for o in outputs]),
                y_predicted=torch.cat([o["y_pred"] for o in outputs]),
                labels=constants.LABEL_PART_TO_CLASSES[self.hparams.label_part]["classes"],
                title="Validation Confusion Matrix",
                file_name="val_confusion-matrix.json",
                overwrite=False,
            )

        # Clean stored output
        self.dset_to_outputs["val"].clear()


    ############################################################################
    #                            Helper Methods                                #
    ############################################################################
    def gmm_get_noisy_labels(self, features, labels, cluster_labels):
        """
        Using a 2-component GMM, identify noisy labels.

        Parameters
        ----------
        features : torch.Tensor
            Contains normalized extracted features of shape (N, D)
        labels : torch.Tensor
            Contains ground-truth noisy labels of shape (N,)
        cluster_labels : torch.Tensor
            Contains soft cluster labels from classifier head of shape (N, K)

        Returns
        -------
        dict
            Contains cluster centers, soft cluster assignment and probability
            that sample has clean label (based on GMM)
        """
        # N := # of samples
        # K := # of labels/clusters
        # D := # of feature dimensions
        N = labels.shape[0]

        # Normalize features
        features = F.normalize(features)

        # Get (normalized) cluster centers
        # Shape: (K, D)
        cluster_means = cluster_labels.T.mm(features)
        cluster_means = F.normalize(cluster_means, dim=1)

        # Compute cluster assignment based on distance
        # NOTE: Scale similarities with temperature
        # Shape: (N, K)
        cluster_assignments_logits = features.mm(cluster_means.T) / self.hparams["temperature"]
        cluster_assignments = F.softmax(cluster_assignments_logits, dim=1)

        # Compute (normalized) negative log-likelihood of "correct" cluster
        # assignment according to noisy label
        nll_loss = -cluster_assignments[torch.arange(N), labels].to(torch.float32)
        nll_loss = nll_loss.cpu().numpy()[:, np.newaxis]
        epsilon = 1e-10
        nll_loss = (nll_loss - nll_loss.min()) / (epsilon + nll_loss.max() - nll_loss.min())

        # TODO: Consider separate GMM for each label
        # Fit (OOD) GMM to detect clean/noisy labels
        gm = GaussianMixture(n_components=2, random_state=0).fit(nll_loss)
        # Predict probability of being in cluster 1/2
        gm_probs = gm.predict_proba(nll_loss)
        # Get probability that label belongs in clean cluster
        # NOTE: Cluster with lower avg. negative log-likelihood is the clean cluster
        is_clean_prob = gm_probs[:, np.argmin(gm.means_)]
        is_clean_prob = torch.tensor(is_clean_prob, dtype=float)

        # Store values
        ret_dict = {
            "cluster_means": cluster_means,
            # Soft cluster assignments
            "cluster_assignments": cluster_assignments,
            # Prob. that label is in clean cluster
            "is_clean_prob": is_clean_prob.to(self.device),
        }

        return ret_dict


    def get_cluster_logits(self, x):
        """
        Given an image, get cluster logits from classifier head

        Parameters
        ----------
        x : torch.Tensor
            Input image

        Returns
        -------
        torch.Tensor
            Cluster logits from prediction head
        """
        out = self.conv_backbone(x).flatten(start_dim=1)
        logits = self.prediction_head(out)
        return logits


    def extract_norm_features(self, x):
        """
        Given an image, extract normalized features from the projection head.

        Parameters
        ----------
        x : torch.Tensor
            Input image

        Returns
        -------
        torch.Tensor
            Model intermediate features from the projection head
        """
        out = self.conv_backbone(x).flatten(start_dim=1)
        out = self.projection_head(out)
        # Ensure features are unit-norm
        out = F.normalize(out)
        return out


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


################################################################################
#                               Helper Functions                               #
################################################################################
def compute_entropy(probs, dim=None):
    """
    Compute entropy

    Parameters
    ----------
    probs : torch.Tensor
        Probability vector of size (K,), where K is the number of classes
        or probability tensor of size (N, K), where N is the number of samples
    dim : int
        Dimension to sum across (rows). If None, sum across all rows

    Returns
    -------
    torch.Tensor
        Entropy of probabilities
    """
    return -torch.sum(probs * torch.log(probs), dim=dim)


def mixup(x, alpha=1.0):
    """
    Perform MixUp on image

    Parameters
    ----------
    x : torch.Tensor
        Batch of images of shape (B, C, H, W)
    alpha : float, optional
        Alpha parameter to describe Beta distribution to sample mixing
        coefficient, by default 1.0

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
        (i) MixUp image,
        (ii) Index of images mixed up with the current image,
        (iii) Mixing parameter for each index
    """
    bs = x.size(0)

    # Get random indices to mix with
    other_idx = torch.randperm(bs).to(x.device)

    # Get mixing coefficient
    lam = np.random.beta(alpha, alpha)
    lam = torch.ones_like(other_idx).float() * lam
    lam = torch.max(lam, 1. - lam)
    lam_expanded = lam.view([-1] + [1] * (x.dim() - 1))

    # Perform MixUp on image
    x = lam_expanded * x + (1. - lam_expanded) * x[other_idx]

    return x, other_idx, lam.unsqueeze(1)


def interpolate(x_1, x_2, lam):
    """
    Linearly interpolate between two variables

    Parameters
    ----------
    x_1 : Any
        Arbitrary variable
    x_2 : Any
        Arbitrary variable
    lam : float
        Interpolation parameter between 0 and 1

    Returns
    -------
    Any
        Interpolated variable
    """
    return (lam * x_1) + ((1-lam) * x_2)
