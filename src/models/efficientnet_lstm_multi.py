"""
efficientnet_lstm_multi.py

Description: Multi-output CNN-LSTM using an EfficientNet convolutional backbone.
             PyTorch Lightning is used to wrap over the efficientnet-pytorch
             library.
"""

# Non-standard libraries
import lightning as L
import torch
import torchmetrics
from efficientnet_pytorch import EfficientNet, get_model_params
from torch.nn import functional as F


################################################################################
#                                  Constants                                   #
################################################################################
DEFAULT_LABEL_TO_NUM_CLASSES = {
    "side": 3,
    "plane": 3
}


################################################################################
#                         EfficientNetLSTMMulti Class                          #
################################################################################
# TODO: Update (training/validation/test)_step for PL integration
# TODO: Update on_(train/val/test)_epoch_end for PL integration
class EfficientNetLSTMMulti(EfficientNet, L.LightningModule):
    """
    EfficientNet + LSTM model for sequence-based classification.
    """
    def __init__(self, label_to_num_classes=None, img_size=(256, 256),
                 optimizer="adamw", lr=0.0005, momentum=0.9, weight_decay=0.0005,
                 n_lstm_layers=1, hidden_dim=512, bidirectional=False,
                 *args, **kwargs):
        """
        Initialize EfficientNetLSTMMulti object.

        Parameters
        ----------
        label_to_num_classes : int, optional
            Mapping of metadata label to number of classes to predict.
            Determines number of model output, by default
            DEFAULT_LABEL_TO_NUM_CLASSES
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
        raise NotImplementedError("Fix TODOs before using!")

        # INPUT: Default multi-output prediction
        if not label_to_num_classes:
            label_to_num_classes = DEFAULT_LABEL_TO_NUM_CLASSES

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
        self.temporal_backbone = torch.nn.LSTM(
            feature_dim, self.hparams.hidden_dim, batch_first=True,
            num_layers=self.hparams.n_lstm_layers,
            bidirectional=self.hparams.bidirectional)

        # Define classification layer
        multiplier = 2 if self.hparams.bidirectional else 1
        size_after_lstm = self.hparams.hidden_dim * multiplier
        for label, num_classes in self.hparams.label_to_num_classes.items():
            exec(f"self.fc_{label} = torch.nn.Linear({size_after_lstm}, "
                 f"{num_classes})")

        # Define loss
        self.loss = torch.nn.NLLLoss()

        # Evaluation metrics
        dsets = ['train', 'val', 'test']
        for dset in dsets:
            for label in self.hparams.label_to_num_classes:
                exec(f"""self.{dset}_acc_{label} = torchmetrics.Accuracy(
                    task='multiclass')""")


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
        x, _ = self.temporal_backbone(x)

        # Remove extra dimension added
        x = x.squeeze(dim=0)

        # Create prediction for each label
        pred_dict = {}
        for label in self.hparams.label_to_num_classes:
            exec(f"pred_dict['{label}'] = self.fc_{label}(x)")

        return pred_dict


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

        # Get multi-output prediction dictionary
        pred_dict = self.forward(data)

        # Accumulate loss for each label
        loss = 0
        for label in self.hparams.label_to_num_classes:
            # Get prediction
            out = pred_dict[label]
            y_pred = torch.argmax(out, dim=1)

            # Get label
            y_true = metadata[label]

            # If shape of labels is (1, seq_len), flatten first dimension
            if len(y_true.size()) > 1:
                y_true = y_true.flatten()

            # Add to loss
            loss += self.loss(F.log_softmax(out, dim=1), y_true)

            # Log training metrics
            exec(f"self.train_acc_{label}.update(y_pred, y_true)")

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

        # Get multi-output prediction dictionary
        pred_dict = self.forward(data)

        # Accumulate loss for each label
        loss = 0
        for label in self.hparams.label_to_num_classes:
            # Get prediction
            out = pred_dict[label]
            y_pred = torch.argmax(out, dim=1)

            # Get label
            y_true = metadata[label]

            # If shape of labels is (1, seq_len), flatten first dimension
            if len(y_true.size()) > 1:
                y_true = y_true.flatten()

            # Add to loss
            loss += self.loss(F.log_softmax(out, dim=1), y_true)

            # Log validation metrics
            exec(f"self.val_acc_{label}.update(y_pred, y_true)")

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

        # Get multi-output prediction dictionary
        pred_dict = self.forward(data)

        # Accumulate loss for each label
        loss = 0
        for label in self.hparams.label_to_num_classes:
            # Get prediction
            out = pred_dict[label]
            y_pred = torch.argmax(out, dim=1)

            # Get label
            y_true = metadata[label]

            # If shape of labels is (1, seq_len), flatten first dimension
            if len(y_true.size()) > 1:
                y_true = y_true.flatten()

            # Add to loss
            loss += self.loss(F.log_softmax(out, dim=1), y_true)

            # Log test metrics
            exec(f"self.test_acc_{label}.update(y_pred, y_true)")

        return loss


    ############################################################################
    #                            Epoch Metrics                                 #
    ############################################################################
    def on_train_epoch_end(self, outputs):
        """
        Compute and log evaluation metrics for training epoch.

        Parameters
        ----------
        outputs: dict
            Dict of outputs of every training step in the epoch
        """
        loss = torch.stack([d['loss'] for d in outputs]).mean()
        self.log('train_loss', loss)

        for label in self.hparams.label_to_num_classes:
            exec(f"acc = self.train_acc_{label}.compute()")
            exec(f"self.log('train_acc_{label}', acc)")
            exec(f"self.train_acc_{label}.reset()")


    def on_validation_epoch_end(self, validation_step_outputs):
        """
        Compute and log evaluation metrics for validation epoch.

        Parameters
        ----------
        outputs: dict
            Dict of outputs of every validation step in the epoch
        """
        loss = torch.tensor(validation_step_outputs).mean()
        self.log('val_loss', loss)

        for label in self.hparams.label_to_num_classes:
            exec(f"acc = self.val_acc_{label}.compute()")
            exec(f"self.log('val_acc_{label}', acc)")
            exec(f"self.val_acc_{label}.reset()")


    def on_test_epoch_end(self, test_step_outputs):
        """
        Compute and log evaluation metrics for test epoch.

        Parameters
        ----------
        outputs: dict
            Dict of outputs of every test step in the epoch
        """
        dset = f'test'

        loss = torch.tensor(test_step_outputs).mean()
        self.log(f'{dset}_loss', loss)
        for label in self.hparams.label_to_num_classes:
            exec(f"acc = self.{dset}_acc_{label}.compute()")
            exec(f"self.log('{dset}_acc_{label}', acc)")
            exec(f"self.{dset}_acc_{label}.reset()")


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
        z = self.extract_features(inputs)

        # 2. Average Pooling
        z = self._avg_pooling(z)

        # Flatten
        z = z.view(1, T, -1)

        # Extract temporal features
        c = self.temporal_backbone(z)[0]
        c = c.view(T, -1)

        return c.detach().cpu().numpy()
