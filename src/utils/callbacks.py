"""
callbacks.py

Description: Implements custom callbacks for PyTorch Lightning Trainer to use.
"""

# Standard libraries
from copy import deepcopy

# Non-standard libraries
import torch
import lightning as L


################################################################################
#                                   Classes                                    #
################################################################################
class EMACallback(L.Callback):
    """
    Exponential Moving Average (EMA) Callback for PyTorch Lightning.

    This callback maintains a shadow copy of the model's weights and updates them
    using an exponential moving average of the current model weights after each
    optimizer step. It also handles the following:

    -   Correctly updates both model parameters and buffers (like BatchNorm running stats).
    -   Manages EMA weight synchronization in a Distributed Data Parallel (DDP) setting.
    -   Saves and loads EMA weights correctly with model checkpoints.
    -   Swaps model weights with EMA weights during validation and testing to
        evaluate the averaged model.

    Parameters
    ----------
    decay : float, optional
        The decay rate for the EMA. A value closer to 1.0 results in a stronger
        smoothing effect. Typical values are 0.999 or 0.9999.
        Defaults to 0.9999.
    apply_ema_every_n_steps : int, optional
        The number of training steps to wait before applying the EMA update.
        Defaults to 1, meaning EMA is updated after every optimizer step.
    """

    def __init__(self, decay: float = 0.9999, apply_ema_every_n_steps: int = 1):
        super().__init__()
        self.decay = decay
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.ema_model = None
        self.original_model_state_dict = None
        self.num_updates = 0


    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Initializes the EMA model at the start of training.

        A deepcopy of the `LightningModule` is created to serve as the shadow model.
        It is immediately placed on the correct device.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The `LightningModule` being trained.
        """
        # Create a deepcopy of the model to hold the EMA weights
        self.ema_model = deepcopy(pl_module)
        self.ema_model.eval()

        # Place EMA model on the correct device(s)
        self.ema_model.to(pl_module.device)


    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Any,
        batch: torch.Any,
        batch_idx: int,
    ):
        """
        Updates the EMA weights after each optimizer step.

        The update formula `theta_ema = decay * theta_ema + (1 - decay) * theta_model`
        is applied to all parameters and buffers. Handles gradient accumulation and
        DDP synchronization.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The `LightningModule` being trained.
        outputs : torch.Any
            The output of the model's training step.
        batch : torch.Any
            The current training batch.
        batch_idx : int
            The index of the current training batch.
        """
        # Check if an optimizer step was performed (handles gradient accumulation)
        if (self.num_updates + 1) % trainer.accumulate_grad_batches == 0:
            # Apply EMA update based on the specified step frequency
            if self.num_updates % self.apply_ema_every_n_steps == 0:
                with torch.no_grad():
                    # Update parameters
                    for param_main, param_ema in zip(pl_module.parameters(), self.ema_model.parameters()):
                        # We only update trainable parameters on the correct device
                        if param_main.requires_grad and param_main.device == param_ema.device:
                            param_ema.data.mul_(self.decay).add_(param_main.data, alpha=1 - self.decay)
                            
                    # Update model buffers (e.g., BatchNorm running stats)
                    # Buffers are not optimized, so we simply copy them.
                    for buffer_main, buffer_ema in zip(pl_module.buffers(), self.ema_model.buffers()):
                        buffer_ema.data.copy_(buffer_main.data)

                    # In a DDP setting, synchronize the EMA weights across all GPUs
                    if trainer.is_global_zero and trainer.world_size > 1:
                        for param_ema in self.ema_model.parameters():
                            torch.distributed.all_reduce(param_ema.data, op=torch.distributed.ReduceOp.SUM)
                            param_ema.data /= trainer.world_size

            self.num_updates += 1


    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Swaps the model's weights with the EMA weights before validation.

        This ensures that the validation loop is run on the more stable,
        averaged EMA model.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The `LightningModule` being trained.
        """
        # Store original weights and use EMA weights for validation
        self.original_model_state_dict = deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self.ema_model.state_dict())


    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Restores the original model weights after validation.

        This is crucial to ensure that training continues from where it left off
        with the correct, non-averaged weights.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The `LightningModule` being trained.
        """
        # Restore original weights after validation
        pl_module.load_state_dict(self.original_model_state_dict)


    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Swaps the model's weights with the EMA weights before testing.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The `LightningModule` being trained.
        """
        self.on_validation_start(trainer, pl_module)


    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Restores the original model weights after testing.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The `LightningModule` being trained.
        """
        self.on_validation_end(trainer, pl_module)


    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict):
        """
        Saves the state of the EMA model along with the regular checkpoint.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The `LightningModule` being trained.
        checkpoint : dict
            The checkpoint dictionary to be saved.
        """
        checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        checkpoint['ema_num_updates'] = self.num_updates


    def on_load_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict):
        """
        Loads the EMA model state from a checkpoint.

        Initializes the EMA model if it doesn't exist yet and loads its state
        from the checkpoint dictionary.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The `LightningModule` being trained.
        checkpoint : dict
            The checkpoint dictionary loaded from a file.
        """
        if 'ema_state_dict' in checkpoint:
            # The EMA model needs to be initialized before loading the state dict
            if self.ema_model is None:
                self.ema_model = deepcopy(pl_module)
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.num_updates = checkpoint.get('ema_num_updates', 0)
