"""
efficientnet_pl.py

Description: PyTorch Lightning wrapper over efficientnet-pytorch library.
"""

# Non-standard libraries
import lightning as L
import torch
import torchmetrics
from efficientnet_pytorch import EfficientNet, get_model_params
from torch.nn import functional as F
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
                 use_cutmix_aug=False, use_mixup_aug=False,
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
        use_cutmix_aug : bool, optional
            If True, use CutMix augmentation during training, by default False.
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
        if use_cutmix_aug:
            assert not use_mixup_aug
            self.train_aug = v2.CutMix(num_classes=num_classes)
        if use_mixup_aug:
            assert not use_cutmix_aug
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

            # Metrics for binary classification
            if self.hparams.num_classes == 2:
                exec(f"""self.{dset}_auroc = torchmetrics.AUROC(
                    task='multiclass')""")
                exec(f"""self.{dset}_auprc = torchmetrics.AveragePrecision(
                    task='multiclass')""")

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
        torch.optim.Optimizer
            Initialized optimizer.
        """
        # Instantiate optimizer
        params = self.parameters()
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params,
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(params,
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer


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

        # Get label (and modify for loss if using cutmix)
        y_true = metadata["label"]

        # If specified, apply CutMix augmentation on images
        y_true_aug = y_true
        if self.hparams.get("use_cutmix_aug") or self.hparams.get("use_mixup_aug"):
            data, y_true_aug = self.train_aug(data, y_true)

        # Get prediction
        out = self.forward(data)
        y_pred = torch.argmax(out, dim=1)

        # Get loss
        ce_loss = self.loss(out, y_true_aug)
        # If specified, compute GradCAM loss
        if self.hparams.use_gradcam_loss:
            gradcam_loss = self.gradcam_loss(*train_batch)
            gradcam_loss_weight = self.hparams.get("gradcam_loss_weight", 1.)
            loss = ce_loss + (gradcam_loss_weight * gradcam_loss)
        else:
            loss = ce_loss

        # Log training metrics
        self.train_acc.update(y_pred, y_true)

        if self.hparams.num_classes == 2:
            self.train_auroc.update(out[:, 1], y_true)
            self.train_auprc.update(out[:, 1], y_true)

        # Prepare result
        ret = {
            "loss": loss.detach().cpu(),
        }
        self.dset_to_outputs["train"].append(ret)

        return loss


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

        if self.hparams.num_classes == 2:
            self.val_auroc.update(out[:, 1], y_true)
            self.val_auprc.update(out[:, 1], y_true)

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

        if self.hparams.num_classes == 2:
            self.test_auroc.update(out[:, 1], y_true)
            self.test_auprc.update(out[:, 1], y_true)

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

        if self.hparams.num_classes == 2:
            auroc = self.train_auroc.compute()
            auprc = self.train_auprc.compute()

            self.log('train_auroc', auroc)
            self.log('train_auprc', auprc, prog_bar=True)

            self.train_auroc.reset()
            self.train_auprc.reset()

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

        if self.hparams.num_classes == 2:
            auroc = self.val_auroc.compute()
            auprc = self.val_auprc.compute()
            self.log('val_auroc', auroc)
            self.log('val_auprc', auprc, prog_bar=True)
            self.val_auroc.reset()
            self.val_auprc.reset()

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

        if self.hparams.num_classes == 2:
            auroc = eval(f'self.{dset}_auroc.compute()')
            auprc = eval(f'self.{dset}_auprc.compute()')
            self.log(f'{dset}_auroc', auroc)
            self.log(f'{dset}_auprc', auprc, prog_bar=True)
            exec(f'self.{dset}_auroc.reset()')
            exec(f'self.{dset}_auprc.reset()')

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
