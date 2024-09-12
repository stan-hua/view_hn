"""
efficientnet_pl.py

Description: PyTorch Lightning wrapper over efficientnet-pytorch library.
"""

# Non-standard libraries
import lightning as L
import torch
import torchmetrics
from efficientnet_pytorch import EfficientNet, get_model_params
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.transforms import v2

# Custom libraries
from src.data import constants
from src.loss.gradcam_loss import ViewGradCAMLoss
from src.utils import efficientnet_pytorch_utils as effnet_utils
from src.utils.grokfast import gradfilter_ema


class EfficientNetPL(EfficientNet, L.LightningModule):
    """
    PyTorch Lightning wrapper module over EfficientNet.
    """
    def __init__(self, num_classes=5, img_size=(256, 256),
                 optimizer="adamw", lr=0.0005, momentum=0.9, weight_decay=0.0005,
                 use_gradcam_loss=False,
                 use_mixup_aug=False,
                 freeze_weights=False, effnet_name="efficientnet-b0",
                 *args, **kwargs):
        """
        Initialize EfficientNetPL object.

        Parameters
        ----------
        num_classes : int, optional
            Number of classes to predict, by default 5
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
        use_gradcam_loss : bool, optional
            If True, add auxiliary segmentation-attention GradCAM loss, by
            default False.
        use_mixup_aug : bool, optional
            If True, use Mixup augmentation during training, by default False
        freeze_weights : bool, optional
            If True, freeze convolutional weights, by default False.
        effnet_name : str, optional
            Name of EfficientNet backbone to use
        """
        # Instantiate EfficientNet
        self.model_name = effnet_name
        blocks_args, global_params = get_model_params(
            self.model_name, {"num_classes": num_classes,
                              "image_size": img_size})
        super().__init__(blocks_args=blocks_args, global_params=global_params)
        self._change_in_channels(kwargs.get("mode", 3))

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters()

        # If specified, store training-specific augmentations
        # NOTE: These augmentations require batches of input
        self.train_aug = None
        if use_mixup_aug:
            self.train_aug = v2.MixUp(num_classes=num_classes)

        # Define loss
        self.loss = torch.nn.CrossEntropyLoss()
        # If specified, include auxiliary GradCAM loss
        self.gradcam_loss = None
        if use_gradcam_loss:
            self.gradcam_loss = ViewGradCAMLoss(self, self.hparams)

        # Evaluation metrics
        dsets = ['train', 'val', 'test']
        for dset in dsets:
            exec(f"self.{dset}_acc = torchmetrics.Accuracy("
                 f"num_classes={self.hparams.num_classes}, task='multiclass')")

        # Store outputs
        self.dset_to_outputs = {"train": [], "val": [], "test": []}

        ########################################################################
        #                          Post-Setup                                  #
        ########################################################################
        # Freeze convolutional weights, if specified
        self.prep_conv_weights()


    def prep_conv_weights(self):
        """
        If specified by internal attribute, freeze all convolutional weights.
        """
        # Skip, if not freezing weights
        if not self.hparams.freeze_weights:
            return

        # SPECIAL CASE: If using GradCAM loss, not compatible with below setup
        if self.hparams.use_gradcam_loss:
            raise NotImplementedError("GradCAM loss is not compatible with freezing weights!")

        conv_requires_grad = not self.hparams.freeze_weights
        blacklist = ["fc", "_fc"]
        for parameter in self.parameters():
            if parameter.name and any(
                parameter.name.startswith(name) for name in blacklist):
                continue
            parameter.requires_grad = conv_requires_grad


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


    def load_imagenet_weights(self):
        """
        Load imagenet weights for convolutional backbone.
        """
        # NOTE: Modified utility function to ignore missing keys
        effnet_utils.load_pretrained_weights(
            self, self.model_name,
            load_fc=False,
            advprop=False)


    ############################################################################
    #                             Optimization                                 #
    ############################################################################
    def on_before_optimizer_step(self, optimizer):
        """
        After `loss.backward()` and before `optimizer.step()`, perform any
        specified operations (e.g., gradient filtering).
        """
        # If specified, use Grokfast-EMA algorithm to filter for slow gradients
        if self.hparams.get("use_grokfast"):
            gradfilter_ema(self)


    def on_train_epoch_start(self):
        """
        Deal with Stochastic Weight Averaging (SWA) Issue in Lightning<=2.3.2
        """
        if self.hparams.get("swa") and self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True


    ############################################################################
    #                          Per-Batch Metrics                               #
    ############################################################################
    def training_step(self, train_batch, batch_idx):
        """
        Batch training step

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
        data, metadata = train_batch
        B = len(data)

        # Get label (and modify for loss if using MixUp)
        y_true = metadata["label"]

        # If specified, apply MixUp augmentation on images
        y_true_aug = y_true
        if self.hparams.get("use_mixup_aug"):
            data, y_true_aug = self.train_aug(data, y_true)

        # Get prediction
        out = self.forward(data)
        y_pred = torch.argmax(out, dim=1)

        losses = []

        # CASE 1: Cross-entropy loss on labeled + Penalize on unlabeled
        if self.hparams.get("penalize_other_loss"):
            # Assume last class (not predicted) is unlabeled idx
            unlabeled_idx = self.hparams["num_classes"]
            # Assert that second half is only "Other" labeled images
            mid_idx = B//2
            assert (y_true[mid_idx:] == unlabeled_idx).all(), \
                "More than 1 label detected in 'Others' samples! Issue with sampler..."

            # 1. Cross-entropy loss on labeled data (first half)
            ce_loss = self.loss(out[:mid_idx], y_true_aug[:mid_idx])
            # 2. Negative entropy loss; to lower confidence on "Other" data predictions
            weight = self.hparams.get("other_penalty_weight", .5)
            other_neg_entropy_loss = (weight * compute_entropy_loss(out[mid_idx:]))
            losses.extend([ce_loss, other_neg_entropy_loss])

            # Filter predictions and labels for correct accuracy computation
            out = out[:mid_idx]
            y_pred = y_pred[:mid_idx]
            y_true = y_true[:mid_idx]
        # CASE 2: Cross-entropy + GradCAM loss
        elif self.hparams.get("use_gradcam_loss"):
            # 1. Cross-entropy loss
            ce_loss = self.loss(out, y_true_aug)
            # 2. GradCAM loss
            weight = self.hparams.get("gradcam_loss_weight", 1.)
            gradcam_loss = weight * self.gradcam_loss(*train_batch)
            losses.extend([ce_loss, gradcam_loss])
        # CASE 3: [Standard] Cross-entropy loss
        else:
            losses.append(self.loss(out, y_true_aug))

        # 4. If specified, add entropy loss to increase prediction confidence
        if self.hparams.get("use_entropy_loss"):
            entropy_loss = -compute_entropy_loss(out)
            losses.append(entropy_loss)

        # Compute loss
        losses = sum(losses)

        # Log training metrics
        self.train_acc.update(y_pred, y_true)

        # Prepare result
        ret = {
            "loss": losses.detach().cpu(),
        }
        self.dset_to_outputs["train"].append(ret)

        return losses


    def validation_step(self, val_batch, batch_idx):
        """
        Batch validation step

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
        data, metadata = val_batch

        # Get prediction
        out = self.forward(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # Get loss
        loss = self.loss(out, y_true)

        # Log validation metrics
        self.val_acc.update(y_pred, y_true)

        # Prepare result
        ret = {
            "loss": loss.detach().cpu(),
            "y_pred": y_pred.detach().cpu(),
            "y_true": y_true.detach().cpu(),
        }
        self.dset_to_outputs["val"].append(ret)

        return ret


    def test_step(self, test_batch, batch_idx):
        """
        Batch test step

        Parameters
        ----------
        test_batch : tuple
            Contains (img tensor, metadata dict)
        batch_idx : int
            Test batch index

        Returns
        -------
        torch.FloatTensor
            Loss for test batch
        """
        data, metadata = test_batch

        # Get prediction
        out = self.forward(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # Get loss
        loss = self.loss(out, y_true)

        # Log test metrics
        self.test_acc.update(y_pred, y_true)

        # Prepare result
        ret = {
            "loss": loss.detach().cpu(),
            "y_pred": y_pred.detach().cpu(),
            "y_true": y_true.detach().cpu(),
        }
        self.dset_to_outputs["test"].append(ret)

        return ret


    ############################################################################
    #                            Epoch Metrics                                 #
    ############################################################################
    def on_train_epoch_end(self):
        """
        Compute and log evaluation metrics for training epoch.
        """
        outputs = self.dset_to_outputs["train"]
        loss = torch.stack([d['loss'] for d in outputs]).mean()
        acc = self.train_acc.compute()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        self.train_acc.reset()

        # Clean stored output
        self.dset_to_outputs["train"].clear()


    def on_validation_epoch_end(self):
        """
        Compute and log evaluation metrics for validation epoch.
        """
        outputs = self.dset_to_outputs["val"]
        loss = torch.tensor([o["loss"] for o in outputs]).mean()
        acc = self.val_acc.compute()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self.val_acc.reset()

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


    def on_test_epoch_end(self):
        """
        Compute and log evaluation metrics for test epoch.
        """
        outputs = self.dset_to_outputs["test"]
        dset = f'test'

        loss = torch.tensor([o["loss"] for o in outputs]).mean()
        acc = eval(f'self.{dset}_acc.compute()')

        self.log(f'{dset}_loss', loss)
        self.log(f'{dset}_acc', acc)

        exec(f'self.{dset}_acc.reset()')

        # Create confusion matrix
        if self.hparams.get("use_comet_logger"):
            self.logger.experiment.log_confusion_matrix(
                y_true=torch.cat([o["y_true"].cpu() for o in outputs]),
                y_predicted=torch.cat([o["y_pred"].cpu() for o in outputs]),
                labels=constants.LABEL_PART_TO_CLASSES[self.hparams.label_part]["classes"],
                title="Test Confusion Matrix",
                file_name="test_confusion-matrix.json",
                overwrite=False,
            )

        # Clean stored output
        self.dset_to_outputs["test"].clear()


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
        # 1. CNN Encoder
        z = self.extract_features(inputs)

        # 2. Average Pooling
        z = self._avg_pooling(z)

        # Flatten
        z = z.view(inputs.size()[0], -1)

        return z.detach().cpu().numpy()


def compute_entropy_loss(out):
    """
    Compute entropy regularization loss

    Note
    ----
    Simply negative entropy for each prediction. Minimizing this means
    increasing entropy (uncertainty) in each prediction.

    Parameters
    ----------
    out : torch.Tensor
        Each row contains model logits

    Returns
    -------
    torch.FloatTensor
        Entropy regularization loss
    """
    # Convert to probabilities
    y_probs = torch.nn.functional.softmax(out, dim=1)

    # Maximize entropy of each prediction (less confident predictions)
    H_pred = torch.mean(torch.sum(y_probs * torch.log(y_probs + 1e-9), dim=1))

    return H_pred