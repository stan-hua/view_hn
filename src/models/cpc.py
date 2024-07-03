"""
cpc.py

Description: Implementation of Contrastive Predictive Coding (CPC) via CNN-LSTM
             with an EfficientNet convolutional backbone. PyTorch Lightning is
             used to wrap over the efficientnet-pytorch library.
"""

# Non-standard libraries
import numpy as np
import lightning as L
import torch
import torchmetrics
from efficientnet_pytorch import EfficientNet, get_model_params
from torch.nn import functional as F


class CPC(EfficientNet, L.LightningModule):
    """
    EfficientNet + LSTM model, implementation of Contrastive Predictive Coding
    (CPC).
    """
    def __init__(self, num_classes=5, img_size=(256, 256),
                 optimizer="adamw", lr=0.0005, momentum=0.9, weight_decay=0.0005,
                 n_lstm_layers=1, hidden_dim=256, bidirectional=True,
                 timesteps=5, extract_features=False, *args, **kwargs):
        """
        Initialize CPC object.

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
        n_lstm_layers : int, optional
            Number of LSTM layers, by default 1
        hidden_dim : int, optional
            Dimension/size of hidden layers, by default 256
        bidirectional : bool, optional
            If True, trains a bidirectional LSTM, by default True
        timesteps : int, optional
            Max number of timesteps ahead to predict, by default 5.
        extract_features : bool, optional
            If True, forward pass returns model output after LSTM.
        """
        # Instantiate EfficientNet
        self.model_name = "efficientnet-b0"
        self.feature_dim = 1280      # expected feature size from EfficientNetB0
        blocks_args, global_params = get_model_params(
            self.model_name, {"image_size": img_size,
                              "include_top": False})
        super().__init__(blocks_args=blocks_args, global_params=global_params)

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters()

        # Define LSTM layers
        self._lstm = torch.nn.LSTM(
            self.feature_dim, self.hparams.hidden_dim, batch_first=True,
            num_layers=self.hparams.n_lstm_layers,
            bidirectional=self.hparams.bidirectional)

        # Define linear layers for each k in (1, <timestep>)
        multiplier = 2 if self.hparams.bidirectional else 1
        size_after_lstm = self.hparams.hidden_dim * multiplier

        for k in range(1, self.hparams.timesteps+1):
            exec(f"self.fc_{k} = torch.nn.Linear(size_after_lstm, "
                 "self.feature_dim)")

        # Define loss
        self.loss = torch.nn.NLLLoss()


    def forward(self, inputs):
        """
        Modified EfficientNet + LSTM forward pass with CPC loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Sequential images for an ultrasound sequence for 1 patient. Expected
            size is (seq_len, C, H, W)

        Returns
        -------
        torch.Tensor
            Model output after final linear layer
        """
        # Expected input size is (seq_len, C, H, W)
        seq_len = inputs.size()[0]

        # Randomly chosen starting time point (t_i)
        t_i = np.random.randint(seq_len - self.hparams.timesteps)

        # 1. CNN Encoder
        z = self.extract_features(inputs)
        z = self._avg_pooling(z)
        z = z.view(seq_len, -1)     # Flatten to (<seq_len>, 512)

        # Get embedded vectors (z_i) for future [t_(i+1), ..., t_(i+timestep)]
        z_future = z[t_i+1:t_i+self.hparams.timesteps+1, :]

        # Get embedded vectors (z_i) for past [t_(i-timestep), ... , t_i]
        z_prev = z[:t_i+1, :]

        # 2. LSTM
        # Get context vector (c_i) from past latent vectors [t_(i-timestep), ... , t_i]
        z_prev = z_prev.unsqueeze(0)    # reshaped for LSTM
        output, _ = self._lstm(z_prev)
        c_i = output[:, t_i, :]

        # Predict [t_(i+1), ..., t_(i+timestep)] from context vector at (t_i)
        z_future_pred = torch.empty((self.hparams.timesteps, self.feature_dim),
                                     device=self.device)
        for k in range(1, self.hparams.timesteps+1):
            linear_k = eval(f"self.fc_{k}")
            z_future_pred[k-1] = linear_k(c_i)

        # Transpose matrix, so matrix multiplication produces dot products
        z_future_pred_T = torch.transpose(z_future_pred, 0, 1)

        # z_(t+k) * W_k * c_t
        all_pairwise_sims = torch.mm(z_future, z_future_pred_T)

        # Return log probability across each row in the
        # <timesteps> x <timesteps> matrix
        return F.log_softmax(all_pairwise_sims, dim=1)


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

        # Get predicted positions of randomly sampled <timesteps>
        preds = self.forward(data)
        # True labels are on the diagonal
        labels = torch.arange(self.hparams.timesteps, device=self.device)

        # Calculate NLL loss
        loss = self.loss(preds, labels)
        # Avg. NCE Loss over num. time steps predicted)
        loss /= self.hparams.timesteps

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

        # Get predicted positions of randomly sampled <timesteps>
        preds = self.forward(data)
        # True labels are on the diagonal
        labels = torch.arange(self.hparams.timesteps, device=self.device)

        # Calculate NLL loss
        loss = self.loss(preds, labels)
        # Avg. NCE Loss over num. time steps predicted)
        loss /= self.hparams.timesteps

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

        # Get predicted positions of randomly sampled <timesteps>
        preds = self.forward(data)
        # True labels are on the diagonal
        labels = torch.arange(self.hparams.timesteps, device=self.device)

        # Calculate NLL loss
        loss = self.loss(preds, labels)
        # Avg. NCE Loss over num. time steps predicted)
        loss /= self.hparams.timesteps

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
        self.log('train_loss', loss)


    def validation_epoch_end(self, validation_step_outputs):
        """
        Compute and log evaluation metrics for validation epoch.

        Parameters
        ----------
        outputs: dict
            Dict of outputs of every validation step in the epoch
        """
        loss = torch.tensor(validation_step_outputs).mean()
        self.log('val_loss', loss)


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
        self.log(f'{dset}_loss', loss)

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
