"""
base.py

Description: PyTorch Lightning wrapper over timm and torchvision libraries
"""

# Standard libraries
import logging

# Non-standard libraries
import lightning as L
import timm
import torch
import torchmetrics
import torchvision
# from hocuspocus.data.augmentations import mix_background
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2

# Custom libraries
from config import constants
from src.loss.gradcam_loss import ViewGradCAMLoss
from src.utils.grokfast import gradfilter_ema


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Default parameters
DEFAULT_PARAMS = {
    "model_provider": "timm",
    "model_name": "efficientnet_b0",
    "num_classes": 5,
    "img_size": (256, 256),
    "optimizer": "adamw",
    "lr": 0.0005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "use_gradcam_loss": False,
    "use_mixup_aug": False,
    "freeze_weights": False,
}


# Mapping of model name to feature size
NAME_TO_FEATURE_SIZE = {
    "efficientnet_b0": 1280,
}


################################################################################
#                                   Classes                                    #
################################################################################
class ModelWrapper(L.LightningModule):
    """
    ModelWrapper class.

    Note
    ----
    Used to load any torchvision/timm model
    """

    def __init__(self, hparams=None, **overwrite_params):
        """
        Initialize ModelWrapper object.

        Parameters
        ----------
        hparams : dict, optional
            Model hyperparameters. To view exhaustive list, see `config/configspecs/model_training.ini`.
            Parameters includes:
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
            model_name : str, optional
                Name of backbone to use
            mode : int, optional
                Image mode to load in (1=grayscale, 3=rgb)
        **overwrite_params : Any
            Keyword arguments to overwrite hyperparameters
        """
        super().__init__()

        # Add default parameters
        hparams = hparams or {}
        hparams.update({k:v for k,v in DEFAULT_PARAMS.items() if k not in hparams})
        hparams.update(overwrite_params)

        # Instantiate model
        self.network = load_network(hparams)

        # Save hyperparameters (now in self.hparams)
        self.save_hyperparameters(hparams)

        # If specified, store training-specific augmentations
        # NOTE: These augmentations require batches of input
        self.train_aug = None
        if self.hparams.use_mixup_aug:
            self.train_aug = v2.MixUp(num_classes=self.hparams.num_classes)

        # Define loss
        self.loss = torch.nn.CrossEntropyLoss()
        # If specified, include auxiliary GradCAM loss
        self.gradcam_loss = None
        if self.hparams.use_gradcam_loss:
            self.gradcam_loss = ViewGradCAMLoss(self, self.hparams)

        # Evaluation metrics
        dsets = ['train', 'val', 'test']
        self.dset_metrics = torch.nn.ModuleDict({
            f"{dset}_acc": torchmetrics.Accuracy(num_classes=self.hparams.num_classes, task='multiclass')
            for dset in dsets
        })

        # Store outputs
        self.dset_to_outputs = {"train": [], "val": [], "test": []}

        # Store class means & inverse covariance matrices
        self.class_means = None
        self.class_inv_covs = None

        ########################################################################
        #                          Post-Setup                                  #
        ########################################################################
        # Placeholder for SAFT parameters whose gradient to mask
        self.saft_param_mask = None

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
        # Get filtered or all parameters
        params = self.parameters()
        
        # Create optimizer
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params,
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(params,
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise RuntimeError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Prepare return
        ret = {
            "optimizer": optimizer,
        }

        # Set up LR Scheduler
        if self.hparams.get("lr_scheduler") == "cosine_annealing":
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            ret["lr_scheduler"] = lr_scheduler

        return ret


    def create_saft_param_mask(self, train_dataloader):
        """
        Create mask for SAFT, to filter for parameters with strong gradients
        for the task

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Training dataset to compute gradients over
        """
        num_samples = 0
        gradients = []
        for (data, metadata) in train_dataloader:
            num_samples += len(data)

            y_true = metadata["label"]
            data, y_true = data.to(self.device), y_true.to(self.device)

            # Perform forward and backward pass
            out = self.network(data)
            loss = self.loss(out, y_true)
            self.zero_grad()
            loss.backward()

            # NOTE: Don't include FC weights in mask
            gradients.append([p.grad.clone() for name, p in self.named_parameters() if p.requires_grad and not name.startswith("_fc")])

        avg_gradients = [torch.mean(torch.stack([g[i] for g in gradients]), dim=0) for i in range(len(gradients[0]))]
        flat_grads = torch.cat([g.view(-1) for g in avg_gradients])
        threshold = torch.topk(flat_grads.abs(), int(self.hparams.get("saft_sparsity", 0.1) * flat_grads.numel()), largest=True)[0][-1]
        self.saft_param_mask = [torch.abs(g) >= threshold for g in avg_gradients]


    ############################################################################
    #                             Optimization                                 #
    ############################################################################
    def on_before_optimizer_step(self, optimizer):
        """
        After `loss.backward()` and before `optimizer.step()`, perform any
        specified operations (e.g., gradient filtering).
        """
        # If Grokfast specified, use Grokfast-EMA algorithm to filter for slow gradients
        if self.hparams.get("use_grokfast"):
            gradfilter_ema(self)

        # If SAFT specified, only keep gradients for chosen parameters
        if self.hparams.get("saft"):
            for (name, p), m in zip(self.named_parameters(), self.saft_param_mask):
                if str(name).startswith("_fc"):
                    continue
                p.grad *= m


    def on_train_epoch_start(self):
        """
        Deal with Stochastic Weight Averaging (SWA) Issue in Lightning<=2.3.2
        """
        # CASE 1: If performing Sparse Adaptive Fine-Tuning (SAFT), filter for
        #         strong parameters on the first epoch
        if self.hparams.get("saft") and self.current_epoch == 0:
            print("Performing Sparse Adaptive Fine-Tuning (SAFT)! "
                  "Getting parameters with strong gradients...")
            self.eval()
            self.create_saft_param_mask(self.trainer.train_dataloader)
            self.train()

        # Pre-compute training set feature means/cov, if doing Mahalanobis OOD
        if self.hparams.get("ood_method") == "maha_distance":
            self.precompute_train_statistics()

        # HACK: Fix issue with Stochastic Weight Optimization on the last epoch
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
        data, metadata = standardize_batch(train_batch["id"])
        B = len(data)
        y_true = metadata["label"]

        # If specified, apply MixUp augmentation on images & labels
        y_true_aug = y_true
        if self.hparams.get("use_mixup_aug"):
            data, y_true_aug = self.train_aug(data, y_true)

        # Get prediction
        out = self.network(data)
        y_pred = torch.argmax(out, dim=1)

        # Accumulate losses
        losses = []
        # 1. Cross-Entropy Loss
        losses.append(self.loss(out, y_true_aug))

        # 2. GradCAM loss
        if self.hparams.get("use_gradcam_loss"):
            weight = self.hparams.get("gradcam_loss_weight", 1.)
            gradcam_loss = weight * self.gradcam_loss(*train_batch)
            losses.append(gradcam_loss)

        # Outlier Exposure
        if self.hparams.get("outlier_exposure"):
            # Compute OOD loss on in-distribution dataset
            curr_ood_loss = -self.ood_step(train_batch["id"])
            self.log("train-id-ood_score", curr_ood_loss, on_step=False, on_epoch=True, batch_size=B)
            losses.append(curr_ood_loss)

            # Compute OOD loss on each OOD dataset
            dataset_names = train_batch.keys()
            for name in dataset_names:
                if name.startswith("ood_"):
                    curr_ood_loss = self.ood_step(train_batch[name])
                    self.log(f"train-{name}-ood_score", curr_ood_loss, on_step=False, on_epoch=True, batch_size=B)
                    losses.append(curr_ood_loss)

        # Aggregate loss
        loss = sum(losses)

        # Log training metrics
        self.dset_metrics["train_acc"](y_pred, y_true)
        self.log("train_acc", self.dset_metrics["train_acc"], on_step=False, on_epoch=True, batch_size=B)

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
        return self.eval_step("val", val_batch)


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
        return self.eval_step("test", test_batch)


    def eval_step(self, split, eval_batch):
        """
        Batch eval step

        Parameters
        ----------
        eval_batch : tuple
            Contains (img tensor, metadata dict)

        Returns
        -------
        torch.FloatTensor
            Loss for eval batch
        """
        assert split in ("val", "test")
        data, metadata = standardize_batch(eval_batch["id"])
        B = len(data)

        # Get prediction
        out = self.network(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # Compute cross-entropy loss
        losses = []
        losses.append(self.loss(out, y_true))

        # Compute OOD loss on in-distribution dataset
        curr_ood_loss = -self.ood_step(eval_batch["id"])
        self.log(f"{split}-id-ood_score", curr_ood_loss, on_step=False, on_epoch=True, batch_size=B)
        losses.append(curr_ood_loss)

        # Compute OOD loss on each OOD dataset
        dataset_names = eval_batch.keys()
        for name in dataset_names:
            if name.startswith("ood_"):
                curr_ood_loss = self.ood_step(eval_batch[name])
                self.log(f"{split}-{name}-ood_score", curr_ood_loss, on_step=False, on_epoch=True, batch_size=B)
                losses.append(curr_ood_loss)
        loss = sum(losses)

        # Log test metrics
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, batch_size=B)
        self.dset_metrics[f"{split}_acc"](y_pred, y_true)
        self.log(f"{split}_acc", self.dset_metrics[f"{split}_acc"], on_step=False, on_epoch=True, batch_size=B)

        # Prepare result
        ret = {
            "y_pred": y_pred.detach().cpu(),
            "y_true": y_true.detach().cpu(),
        }
        self.dset_to_outputs[split].append(ret)

        return loss


    def ood_step(self, ood_batch):
        """
        Perform OOD step. Compute OOD loss to differentiate

        Note
        ----
        For a training batch, negate the loss output by this function.

        Parameters
        ----------
        ood_batch : tuple of (torch.Tensor, dict)
            Contains (img tensor, metadata dict) from OOD data batch
        """
        X, metadata = standardize_batch(ood_batch)
        ood_method = self.hparams.get("ood_method", "msp")
        oe_weight = self.hparams.get("oe_weight", .1)

        # Consider background overlay augmentation
        # if self.hparams.get("ood_overlay_background") and "background_img" in metadata:
        #     shuffle = self.hparams.get("ood_mix_background", False)
        #     background = metadata["background_img"]
        #     X = mix_background(X, background, shuffle=shuffle)

        # Compute OOD loss
        # NOTE: Optimize to increase entropy/energy in each prediction
        loss = -self.ood_score(X)
        return oe_weight * loss


    ############################################################################
    #                            Epoch Metrics                                 #
    ##############################X##########################################
    def on_validation_epoch_end(self):
        """
        Compute and log evaluation metrics for validation epoch.
        """
        self.on_eval_epoch_end("val")


    def on_test_epoch_end(self):
        """
        Compute and log evaluation metrics for test epoch.
        """
        self.on_eval_epoch_end("test")


    def on_eval_epoch_end(self, split):
        """
        Compute and log evaluation metrics for the specified epoch split.
        
        Parameters
        ----------
        split : str
            The dataset split to evaluate. Must be one of ('val', 'test').

        Notes
        -----
        - This function calculates and logs a confusion matrix using the Comet logger
        if enabled in the hyperparameters.
        - It clears the stored outputs for the specified split after logging.
        """
        assert split in ("val", "test")

        # Create confusion matrix
        outputs = self.dset_to_outputs[split]
        if self.hparams.get("use_comet_logger"):
            self.logger.experiment.log_confusion_matrix(
                y_true=torch.cat([o["y_true"] for o in outputs]),
                y_predicted=torch.cat([o["y_pred"] for o in outputs]),
                labels=constants.LABEL_PART_TO_CLASSES[self.hparams.label_part]["classes"],
                title=f"{split.capitalize()} Confusion Matrix",
                file_name=f"{split}_confusion-matrix.json",
                overwrite=False,
            )

        # Clean stored output
        self.dset_to_outputs[split].clear()


    ############################################################################
    #                          Extract Embeddings                              #
    ############################################################################
    @torch.no_grad()
    def precompute_train_statistics(self):
        """
        Computes the mean and covariance matrix for each class in the training set.

        Uses the trainer's train dataloader to extract features for every image in the
        training set. Then, computes the mean and covariance matrix for each class
        using the corresponding feature embeddings.

        Stores the computed class means and covariance matrices in the object.
        """
        train_dataloader = self.trainer.train_dataloader

        # Initialize lists to store feature embeddings for each class
        class_embeddings = {idx: list() for idx in range(self.hparams["num_classes"])}

        # Extract features for every image
        for train_batch in train_dataloader:
            X, metadata = train_batch["id"]
            X = X.to(self.device)
            labels = metadata["label"].to(self.device)

            # Extract features
            features = self.forward_features(X)
            for idx in range(len(features)):
                label = labels[idx].item()
                class_embeddings[label].append(features[idx])

        # Compute mean and covariance matrix for each class
        class_means = [None] * self.hparams["num_classes"]
        class_inv_cov = [None] * self.hparams["num_classes"]
        for class_label, embeddings in class_embeddings.items():
            embeddings = torch.tensor(embeddings)
            # Compute mean
            mean = torch.mean(embeddings, dim=0)
            class_means[class_label] = mean
            # Compute inverse covariance matrix
            cov_matrix = torch.cov(embeddings.T)
            inv_cov_matrix = torch.inverse(cov_matrix)
            class_inv_cov[class_label] = inv_cov_matrix

        # Store class means/covariance matrices
        self.class_means = torch.stack(class_means)
        self.class_inv_covs = torch.stack(class_inv_cov)


    def forward(self, x):
        """
        Implement forward pass function.

        Parameters
        ----------
        x : torch.Tensor
            Input image
        """
        return self.network(x)


    def forward_features(self, inputs):
        """
        Extracts features from input images

        Parameters
        ----------
        inputs : torch.Tensor
            Ultrasound images. Expected size is (B, C, H, W)

        Returns
        -------
        numpy.array
            Deep embeddings
        """
        z = extract_features(self.hparams, self.network, inputs)
        z = z.view(inputs.size()[0], -1)
        return z


    @torch.no_grad()
    def extract_embeds(self, inputs):
        """
        Wrapper over `forward_features` but returns CPU numpy array

        Parameters
        ----------
        inputs : torch.Tensor
            Ultrasound images. Expected size is (B, C, H, W)

        Returns
        -------
        numpy.array
            Deep embeddings
        """
        return self.forward_features(inputs).detach().cpu().numpy()


    def ood_score(self, imgs, ood_method=None):
        """
        Compute OOD score to optimize, based on method

        Parameters
        ----------
        imgs : torch.Tensor
            Images
        ood_method : str, optional
            OOD Method

        Returns
        -------
        torch.Tensor
            OOD score (to optimize)
        """
        ood_method = ood_method or self.hparams.get("ood_method", "msp")
        assert ood_method in ["msp", "energy", "maha_distance"], "Invalid `ood_method` provided!"

        # Compute OOD score (higher = OOD)
        # CASE 1: Maximum Softmax Probability (MSP)
        if ood_method == "msp":
            logits = self.network(imgs)
            score = compute_entropy(logits)
        # CASE 2: Energy
        elif ood_method == "energy":
            logits = self.network(imgs)
            score = compute_energy(logits).mean()
        # CASE 3: Mahalanobis Distance
        elif ood_method == "maha_distance":
            features = self.forward_features(imgs)
            score = compute_mahalanobis_distance(features, self.class_means, self.class_inv_covs)
        return score



################################################################################
#                               Model Functions                                #
################################################################################
def load_network(hparams, remove_head=False):
    """
    Load model in PyTorch

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters. Can contain any of the following:
        model_provider : str
            One of "torchvision" or "timm"
        model_name : str
            Name of model
        num_classes : int
            Number of classes
        pretrained : bool
            Whether to use ImageNet pretrained weights
        img_size : int
            Image size
        mode : int
            Number of input channels
    remove_head : bool, optional
        Whether to remove the classification head, by default False

    Returns
    ------
    torch.nn.Module
        Loaded model
    """
    # Load model backbone using torchvision or timm
    model_provider = hparams.get("model_provider", "timm")
    # CASE 1: Torchvision
    if model_provider == "torchvision":
        # Raise error for not supported arguments
        model_cls = getattr(torchvision.models, hparams["model_name"])
        model = model_cls(
            num_classes=hparams["num_classes"],
            weights="IMAGENET1K_V1" if hparams.get("pretrained") else None,
        )
        # Change number of input channels
        if hparams.get("mode", 3) != 3:
            # Compute the average of the weights across the input channels
            original_weights = model.conv1.weight.data
            new_weights = original_weights.mean(dim=1, keepdim=True)

            # Modify the first convolutional layer to accept 1 input channel instead of 3
            model.conv1 = torch.nn.Conv2d(
                1, model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=model.conv1.bias,
            )
            # Assign the new weights to the modified convolutional layer
            model.conv1.weight.data = new_weights

        # Remove head, if specified
        if remove_head:
            if hasattr(model, "classifier"):
                model.classifier = torch.nn.Identity()
            else:
                raise NotImplementedError(f"Head removal not implemented for `{model_provider}/{hparams['model_name']}`")
    # CASE 2: timm
    elif model_provider == "timm":
        model = timm.create_model(
            model_name=hparams["model_name"],
            num_classes=hparams["num_classes"],
            in_chans=hparams.get("mode", 3),
            pretrained=hparams.get("pretrained", False),
        )

        # Remove head, if specified
        if remove_head:
            model = model.reset_classifier(0, "")
    else:
        raise NotImplementedError(f"Invalid model_provider specified! `{model_provider}`")

    return model


def extract_features(hparams, model, inputs):
    """
    Extract features from model.

    Parameters
    ----------
    hparams : dict
        Hyperparameters
    model : torch.nn.Module
        Neural network (not wrapper)
    inputs : torch.Tensor
        Model input

    Returns
    -------
    torch.Tensor
        Extracted features
    """
    model_name = hparams["model_name"]
    model_provider = hparams["model_provider"]
    extractor = None
    # CASE 1: Torchvision model
    if model_provider == "torchvision":
        if model_name == "resnet50":
            return_nodes = {"layer4": "layer4"}
            extractor = create_feature_extractor(model, return_nodes)
    # CASE 2: Timm model
    elif model_provider == "timm":
        extractor = model.forward_features

    # Raise error, if not implemented
    if extractor is None:
        raise NotImplementedError(
            "Feature extraction not implemented for "
            f"`{model_provider}/{model_name}`"
        )

    return extractor(inputs)


def standardize_batch(batch):
    """
    Standardize batch to contain X and metadata.

    Parameters
    ----------
    batch : tuple or dict
        Batch of data to standardize

    Returns
    -------
    X : torch.Tensor
        Model input
    metadata : dict
        Any additional metadata

    Raises
    ------
    RuntimeError
        If standardization fails
    """
    try:
        if isinstance(batch, (tuple, list)):
            X, metadata = batch
            return X, metadata
        elif isinstance(batch, dict):
            key = "image" if "image" in batch else "img"
            X = batch.pop(key)
            metadata = batch
            return X, metadata
    except:
        raise RuntimeError(f"[Standardize Batch] Failed to standardize batch with type `{type(batch)}`")


################################################################################
#                                OOD Functions                                 #
################################################################################
def compute_entropy(logits):
    """
    Compute entropy regularization loss

    Note
    ----
    Simply entropy for each prediction. Minimizing this means
    decreasing entropy (uncertainty) for each prediction.

    Parameters
    ----------
    logits : torch.Tensor
        Each row contains model logits

    Returns
    -------
    torch.FloatTensor
        Entropy regularization loss
    """
    # Convert to probabilities
    y_probs = torch.nn.functional.softmax(logits, dim=1)

    # Compute entropy
    H_pred = -torch.mean(torch.sum(y_probs * torch.log(y_probs + 1e-9), dim=1))

    return H_pred


def compute_energy(logits):
    """
    Compute OOD energy scores

    Parameters
    ----------
    logits : torch.Tensor
        Each row contains model logits

    Returns
    -------
    torch.FloatTensor
        OOD energy scores
    """
    energy_score = -torch.logsumexp(logits, dim=1)
    return energy_score


def compute_mahalanobis_distance(x, class_means, class_inv_cov):
    """
    Compute the Mahalanobis distance for a given sample.

    Parameters
    ----------
    x : torch.Tensor
        A tensor representing the sample for which the distance is computed.
    class_means : torch.Tensor
        A tensor representing the means of the classes.
    class_inv_cov : torch.Tensor
        A tensor representing the inverse covariance matrix for each class.

    Returns
    -------
    torch.Tensor
        The Mahalanobis distance of the sample from the class means.
    """
    # Compute the Mahalanobis distance
    diff = x - class_means
    distance = torch.sqrt(torch.mm(torch.mm(diff.unsqueeze(0), class_inv_cov), diff.unsqueeze(1)))
    return distance
