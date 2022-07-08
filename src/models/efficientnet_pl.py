"""
efficientnet_pl.py

Description: PyTorch Lightning wrapper over efficientnet-pytorch library.
"""

# Non-standard libraries
import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet


class EfficientNetPL(pl.LightningModule):
    """
    PyTorch Lightning wrapper module over EfficientNet.
    """
    def __init__(self, num_classes=5, img_size=(256, 256), *args, **kwargs):
        """
        Initialize EfficientNetPL object.

        Parameters
        ----------
        num_classes : int, optional
            Number of classes to predict, by default 5
        img_size : tuple, optional
            Expected image's (height, width)
        """
        # Instantiate EfficientNetB0
        self.model = EfficientNet.from_name("efficientnet-b0",
                                            num_classes=num_classes,
                                            image_size=img_size)

        # Define loss
        self.loss = torch.nn.NLLLoss()

        # Evaluation metrics
        dsets = ['train', 'val', 'test']
        for dset in dsets:
            exec(f"self.{dset}_acc = torchmetrics.Accuracy()")
            exec(f"self.{dset}_auroc = torchmetrics.AUROC("
                 "num_classes={num_classes}, average='micro')")
            exec(f"self.{dset}_auprc= torchmetrics.AveragePrecision("
                 "num_classes={num_classes})")


    def forward(self, data):
        """
        Forward pass through network

        Parameters
        ----------
        data : tuple
            Contains (img tensor, metadata dict)
        
        Returns
        -------
        torch.Tensor
            Output of forward pass (logits). Of size (_, num_classes)
        """
        img, _ = data

        return super(EfficientNet)(img)


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

        # Get prediction
        out = self.forward(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # Get loss
        loss = self.loss(out, y_true)

        # Log training metrics
        self.train_acc.update(y_pred, y_true)
        self.train_auroc.update(out[:, 1], y_true)
        self.train_auprc.update(out[:, 1], y_true)

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
        self.val_auroc.update(out[:, 1], y_true)
        self.val_auprc.update(out[:, 1], y_true)

        return loss


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
        auroc = self.train_auroc.compute()
        auprc = self.train_auprc.compute()

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_auroc', auroc)
        self.log('train_auprc', auprc, prog_bar=True)

        self.train_acc.reset()
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
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_auroc', auroc)
        self.log('val_auprc', auprc, prog_bar=True)

        self.val_acc.reset()
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
        auroc = eval(f'self.{dset}_auroc.compute()')
        auprc = eval(f'self.{dset}_auprc.compute()')

        print(acc, auroc, auprc)

        self.log(f'{dset}_loss', loss)
        self.log(f'{dset}_acc', acc)
        self.log(f'{dset}_auroc', auroc)
        self.log(f'{dset}_auprc', auprc, prog_bar=True)

        exec(f'self.{dset}_acc.reset()')
        exec(f'self.{dset}_auroc.reset()')
        exec(f'self.{dset}_auprc.reset()')
