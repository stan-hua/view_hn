"""
lstm_linear_eval.py

Description: Used to provide a lstm + linear classification evaluation over
             pretrained convolutional backbones.
"""

# Non-standard libraries
import lightning as L
import torch
import torchmetrics
from lightly.models.utils import deactivate_requires_grad
from torch.nn import functional as F

# Custom libraries
from config import constants


# TODO: Consider adding Grokfast
# TODO: Handle last epoch issue with SWA
class LSTMLinearEval(L.LightningModule):
    """
    LSTMLinearEval object, wrapping over convolutional backbone.
    """
    def __init__(self, conv_backbone,
                 temporal_backbone=None,
                 freeze_weights=True,
                 conv_backbone_output_dim=1280,
                 num_classes=5,
                 img_size=(256, 256),
                 optimizer="adamw", lr=0.0005, momentum=0.9, weight_decay=0.0005,
                 n_lstm_layers=1, hidden_dim=512, bidirectional=True,
                 *args, **kwargs):
        """
        Initialize LSTMLinearEval object.

        Parameters
        ----------
        conv_backbone : torch.nn.Module
            Pretrained convolutional backbone
        temporal_backbone : torch.nn.Module, optional
            Pretrained temporal backbone. Instantiates and trains one, by
            default None.
        freeze_weights : bool, optional
            If True, freeze weights of provided pretrained backbone/s, by
            default True.
        conv_backbone_output_dim : int, optional
            Size of output of pretrained convolutional backbone, by default 1280
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
        n_lstm_layers : int, optional
            Number of LSTM layers, by default 1
        hidden_dim : int, optional
            Dimension/size of hidden layers, by default 512
        bidirectional : bool, optional
            If True, trains a bidirectional LSTM, by default True
        """
        raise RuntimeError("LSTMLinear is now deprecated!")
        super().__init__()

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters(ignore=["conv_backbone", "temporal_backbone"])

        # Store convolutional backbone, and freeze its weights
        self.conv_backbone = conv_backbone
        if self.hparams.freeze_weights:
            deactivate_requires_grad(conv_backbone)

        # If provided, store temporal backbone
        if temporal_backbone:
            self.temporal_backbone = temporal_backbone
            if self.hparams.freeze_weights:
                deactivate_requires_grad(temporal_backbone)
        else:
            # Define LSTM layers
            self.temporal_backbone = torch.nn.LSTM(
                self.hparams.conv_backbone_output_dim,
                self.hparams.hidden_dim,
                batch_first=True,
                num_layers=self.hparams.n_lstm_layers,
                bidirectional=self.hparams.bidirectional)

        # Define classification layer
        multiplier = 2 if self.hparams.bidirectional else 1
        size_after_lstm = self.hparams.hidden_dim * multiplier
        self.fc = torch.nn.Linear(size_after_lstm, num_classes)

        # Define loss
        self.loss = torch.nn.NLLLoss()

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


    def forward(self, inputs):
        """
        Forward pass

        Parameters
        ----------
        inputs : torch.Tensor
            Sequential images for an ultrasound sequence for 1 patient. Expected
            size is (T, C, H, W).

        Returns
        -------
        torch.Tensor
            Model output after final linear layer
        """
        # Extract convolutional features
        x = self.conv_backbone(inputs)

        # LSTM layers
        seq_len = x.size()[0]
        x = x.view(1, seq_len, -1)
        x, _ = self.temporal_backbone(x)

        # Linear layer
        x = self.fc(x)

        # Remove extra dimension added for LSTM
        x = x.squeeze(dim=0)

        return x


    ############################################################################
    #                          Per-Batch Metrics                               #
    ############################################################################
    def training_step(self, train_batch, batch_idx):
        """
        Training step, where a batch represents all images from the same
        ultrasound sequence.

        Note
        ----
        Image tensors are of the shape:
            - (sequence_length, num_channels, img_height, img_width)

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

        # If shape is (1, seq_len, C, H, W), flatten first dimension
        if len(data.size()) == 5:
            data = data.squeeze(dim=0)
################################################################################
################################################################################
        # Get prediction
        out = self.forward(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # If shape of labels is (1, seq_len), flatten first dimension
        if len(y_true.size()) > 1:
            y_true = y_true.flatten()

        # Get loss
        loss = self.loss(F.log_softmax(out, dim=1), y_true)

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

        return ret


    def validation_step(self, val_batch, batch_idx):
        """
        Validation step, where a batch represents all images from the same
        ultrasound sequence.

        Note
        ----
        Image tensors are of the shape:
            - (sequence_length, num_channels, img_height, img_width)

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

        # If shape is (1, seq_len, C, H, W), flatten first dimension
        if len(data.size()) == 5:
            data = data.squeeze(dim=0)

        # Get prediction
        out = self.forward(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # If shape of labels is (1, seq_len), flatten first dimension
        if len(y_true.size()) > 1:
            y_true = y_true.flatten()

        # Get loss
        loss = self.loss(F.log_softmax(out, dim=1), y_true)

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
        Test step, where a batch represents all images from the same
        ultrasound sequence.

        Note
        ----
        Image tensors are of the shape:
            - (sequence_length, num_channels, img_height, img_width)

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

        # If shape is (1, seq_len, C, H, W), flatten first dimension
        if len(data.size()) == 5:
            data = data.squeeze(dim=0)

        # Get prediction
        out = self.forward(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # If shape of labels is (1, seq_len), flatten first dimension
        if len(y_true.size()) > 1:
            y_true = y_true.flatten()

        # Get loss
        loss = self.loss(F.log_softmax(out, dim=1), y_true)

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

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        self.train_acc.reset()

        if self.hparams.num_classes == 2:
            auroc = self.train_auroc.compute()
            auprc = self.train_auprc.compute()

            self.log('train_auroc', auroc)
            self.log('train_auprc', auprc, prog_bar=True)

            self.train_auroc.reset()
            self.train_auprc.reset()


    def on_validation_epoch_end(self):
        """
        Compute and log evaluation metrics for validation epoch.
        """
        outputs = self.dset_to_outputs["val"]
        loss = torch.tensor([o["loss"] for o in outputs]).mean()
        acc = self.val_acc.compute()

        self.log('val_loss', loss)
        self.log('val_acc', acc)

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
                y_true=torch.cat([o["y_true"].cpu() for o in outputs]),
                y_predicted=torch.cat([o["y_pred"].cpu() for o in outputs]),
                labels=constants.LABEL_PART_TO_CLASSES[self.hparams.label_part]["classes"],
                title="Validation Confusion Matrix",
                file_name="val_confusion-matrix.json",
                overwrite=False,
            )


    def on_test_epoch_end(self):
        """
        Compute and log evaluation metrics for test epoch.
        """
        dset = f'test'
        outputs = self.dset_to_outputs[dset]
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
            Ultrasound images from one ultrasound image sequence. Expected size
            is (T, C, H, W)

        Returns
        -------
        numpy.array
            Embeddings after CNN+LSTM
        """
        # Get dimensions
        T, _, _, _ = inputs.size()

        # Extract convolutional features
        z = self.conv_backbone(inputs)

        # # Flatten
        # z = z.view(1, T, -1)

        # # Extract temporal features
        # c = self.temporal_backbone(z)[0]

        # return c

        # TODO: To use LSTM features, uncomment above and remove below
        return z.detach().cpu().numpy()
