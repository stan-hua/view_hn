"""
efficientnet_lstm_pl.py

Description: CNN-LSTM using an EfficientNet convolutional backbone. PyTorch
             Lightning is used to wrap over the efficientnet-pytorch library.
"""

# Non-standard libraries
import pytorch_lightning as pl
import torch
import torchmetrics
from efficientnet_pytorch import EfficientNet, get_model_params
from torch.nn import functional as F


class EfficientNetLSTM(EfficientNet, pl.LightningModule):
    """
    EfficientNet + LSTM model for sequence-based classification.
    """
    def __init__(self, num_classes=5, img_size=(256, 256),
                 adam=True, lr=0.0005, momentum=0.9, weight_decay=0.0005,
                 n_lstm_layers=1, hidden_dim=512, bidirectional=True,
                 extract_features=False, *args, **kwargs):
        """
        Initialize EfficientNetLSTM object.

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
        n_lstm_layers : int, optional
            Number of LSTM layers, by default 1
        hidden_dim : int, optional
            Dimension/size of hidden layers, by default 512
        bidirectional : bool, optional
            If True, trains a bidirectional LSTM, by default True
        extract_features : bool, optional
            If True, forward pass returns model output after LSTM.
        """
        # Instantiate EfficientNet
        self.model_name = "efficientnet-b0"
        feature_dim = 1280      # expected feature size from EfficientNetB0
        blocks_args, global_params = get_model_params(
            self.model_name, {"image_size": img_size,
                              "include_top": False})
        super().__init__(blocks_args=blocks_args, global_params=global_params)

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters()

        # Define LSTM layers
        self._lstm = torch.nn.LSTM(
            feature_dim, self.hparams.hidden_dim, batch_first=True,
            num_layers=self.hparams.n_lstm_layers,
            bidirectional=self.hparams.bidirectional)

        # Define classification layer
        multiplier = 2 if self.hparams.bidirectional else 1
        size_after_lstm = self.hparams.hidden_dim * multiplier
        self._fc = torch.nn.Linear(size_after_lstm, self.hparams.num_classes)

        # Define loss
        self.loss = torch.nn.NLLLoss()

        # Evaluation metrics
        dsets = ['train', 'val', 'test']
        for dset in dsets:
            exec(f"self.{dset}_acc = torchmetrics.Accuracy()")

            # Metrics for binary classification
            if self.hparams.num_classes == 2:
                exec(f"self.{dset}_auroc = torchmetrics.AUROC()")
                exec(f"self.{dset}_auprc = torchmetrics.AveragePrecision()")


    def forward(self, inputs):
        """
        Modified EfficientNet + LSTM forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Sequential images for an ultrasound sequence for 1 patient

        Returns
        -------
        torch.Tensor
            Model output after final linear layer
        """
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling
        x = self._avg_pooling(x)

        # LSTM layers
        seq_len = x.size()[0]
        x = x.view(1, seq_len, -1)
        x, _ = self._lstm(x)

        if not self.hparams.extract_features:
            x = self._fc(x)

        # Remove extra dimension added for LSTM
        x = x.squeeze(dim=0)

        return x


    def configure_optimizers(self):
        """
        Initialize and return optimizer (Adam or SGD).

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer.
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

        return loss


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

        return loss


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


    def validation_epoch_end(self, validation_step_outputs):
        """
        Compute and log evaluation metrics for validation epoch.

        Parameters
        ----------
        outputs: dict
            Dict of outputs of every validation step in the epoch
        """
        loss = torch.tensor(validation_step_outputs).mean()
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


    def test_epoch_end(self, test_step_outputs):
        """
        Compute and log evaluation metrics for test epoch.

        Parameters
        ----------
        outputs: dict
            Dict of outputs of every test step in the epoch
        """
        dset = f'test'

        loss = torch.tensor(test_step_outputs).mean()
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