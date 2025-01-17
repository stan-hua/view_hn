"""
gradcam_loss.py

Description: Implementation of segmentation-guided GradCAM loss. This loss
             penalizes gradients that fall outside of given segmentations.
             Code adapted from original repository linked below.

Warning: GradCAMLoss currently is not compatible with the following:
    1. Gradient accumulation; gradient zero-ing is needed,
    2. Freezing weights; the gradient need to be computed across weights,
    3. Video models; 

Source Repo: https://github.com/MeriDK/segmentation-guided-attention/tree/master
"""

# Non-standard libraries
import torch
from torchvision.transforms.v2.functional import resize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Custom libraries
from src.scripts import load_model

# TODO: Consider modifying ActivationsAndGradients to use last conv. layer
#       output instead of after AveragePool
################################################################################
#                                   Classes                                    #
################################################################################
class ViewGradCAMLoss(torch.nn.Module):
    """
    View label - specific GradCAM loss
    """

    def __init__(self, model, hparams):
        """
        Instantiate ViewGradCAMLoss object.

        Parameters
        ----------
        model : torch.nn.Module
            Arbitrary model
        hparams : dict
            Contains model hyperparameters
        """
        super().__init__()

        # Store hyperparameters
        self.hparams = hparams

        # Flag to penalize ALL classes for having gradients OUTSIDE the positive class mask
        self.use_all_class_gradcam_loss = hparams.get("use_all_class_gradcam_loss", True)

        # Flag to penalize negative class for having gradients in true class segmentation masks
        self.add_neg_class_gradcam_loss = hparams.get("add_neg_class_gradcam_loss", False)
        # NOTE: The two flags conflict
        if self.use_all_class_gradcam_loss:
            self.add_neg_class_gradcam_loss = False

        # Flag to use original implementation
        # NOTE: Original implementation was only for binary. Penalize all classes
        self.use_orig_implement = hparams.get("use_orig_implement", False)
        if self.use_orig_implement:
            self.use_all_class_gradcam_loss = True

        # Create GradCAM
        self.cam = create_cam(model)


    def forward(self, X, metadata):
        """
        Compute GradCAM loss.

        Parameters
        ----------
        X : torch.Tensor
            Input image/s
        metadata : dict
            Contains labels and masks

        Returns
        -------
        torch.Tensor
            GradCAM loss
        """
        N, _, H, W = X.shape

        # Early return, if no segmentation masks in batch
        has_seg_mask = metadata["has_seg_mask"]
        if not has_seg_mask.any():
            device = metadata["label"].device
            return torch.tensor(0., dtype=float, device=device,
                                requires_grad=True)

        # Filter out items without segmentation masks
        X = X[has_seg_mask]
        seg_masks = metadata["seg_mask"][has_seg_mask]
        labels = metadata["label"][has_seg_mask]

        # Add channel dimension to segmentation mask for broadcasting
        seg_masks = seg_masks.unsqueeze(1)

        # Prepare targets from labels
        # NOTE: Used to index class-specific gradients
        target_categories = labels.cpu().data.numpy()
        target_cls = ClassifierOutputAllTargets if self.use_all_class_gradcam_loss else ClassifierOutputTarget
        pos_targets = [target_cls(category) for category in target_categories]

        # Create GradCAM for positive class
        pos_cam = self.cam(X, pos_targets)

        # [ORIGINAL IMPLEMENTATION]
        # 1. Compute MSE between segmentation mask and GradCAM
        if self.use_orig_implement:
            assert (seg_masks.float() > 1.0).sum() == 0, f"Max seg mask exceeds 1! Max: `{seg_masks.float().max()}`"
            loss = ((pos_cam - seg_masks.float()) ** 2).sum()
            loss = loss / (N * H * W)
            return loss

        # Compute mean-squared error loss
        # NOTE: Penalize activations outside of segmentation mask
        pos_loss = (pos_cam[~seg_masks] ** 2).sum()
        # Normalize to pixel-level loss (across channels)
        pos_loss = pos_loss / (N * (len(seg_masks[~seg_masks])))

        # NEW: Scale loss by inverse proportion of samples with mask
        inverse_prop = len(has_seg_mask) / sum(has_seg_mask)
        pos_loss = inverse_prop * pos_loss

        # Early exit, if not penalizing negative classes
        # NOTE: Don't penalize negative class if already penalizing all classes
        if not self.add_neg_class_gradcam_loss or self.use_all_class_gradcam_loss:
            return pos_loss

        # NEW: Below can be used if classes strictly do not overlap in the image
        # Index negative class specific gradients
        neg_targets = [ClassifierOutputNegativeTarget(category) for category in target_categories]
        # Create GradCAMs for negative (all other) classes
        neg_cam = self.cam(X, neg_targets)

        # Compute mean-squared error loss
        # NOTE: Penalize activations inside of segmentation mask
        neg_loss = (neg_cam[seg_masks] ** 2).sum()
        # Normalize to pixel-level loss (across channels)
        neg_loss = neg_loss / (N * (len(seg_masks[seg_masks])))

        # Compute GradCAM loss as their weighted sum
        loss = (0.5 * pos_loss) + (0.5 * neg_loss)

        return loss


# Adapted from `pytorch_grad_cam`
# Inspired by https://github.com/MeriDK/segmentation-guided-attention/
class ClassifierOutputAllTargets(ClassifierOutputTarget):
    """
    A sub-class of ClassifierOutputTarget to extract gradients related to
    all the classes.

    Note
    ----
    This means that the GradCAM will be for all classes.
    """

    def __call__(self, model_output):
        # NOTE: Need to sum up activations across classes
        return model_output.sum(dim=-1)


class ClassifierOutputNegativeTarget(ClassifierOutputTarget):
    """
    A sub-class of ClassifierOutputTarget to extract gradients related to
    all the negative classes.

    Note
    ----
    This may be useful if the positive and negative classes shouldn't have
    gradients that overlap their segmented regions.
    """

    def __call__(self, model_output):
        # Filter for output for all of the negative classes
        if len(model_output.shape) == 1:
            mask = torch.ones(model_output.shape[0], dtype=bool)
            mask[self.category] = False
            return model_output[mask]

        assert len(model_output.shape) == 2
        mask = torch.ones(model_output.shape[1], dtype=bool)
        mask[self.category] = False
        return model_output[:, mask]


class DifferentiableActivationsAndGradients(ActivationsAndGradients):
    """
    Sub-class of ActivationsAndGradients that can be back-propagated.
    """

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation)


    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad] + self.gradients

        output.register_hook(_store_grad)


class DifferentiableGradCAM(GradCAM):
    """
    Sub-class of GradCAM that can be back-propagated.
    """

    def __init__(self, model, target_layers, reshape_transform=None):
        """
        Overwrite `__init__` to avoid setting `model.eval()`
        """
        self.target_layers = target_layers

        # NOTE: Ignore model call to `eval()` in BaseCAM
        self.model = model
        self.device = next(self.model.parameters()).device

        # Default parameters
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = False
        self.uses_gradients = True
        # Following is not used by GradCAM specifically
        self.tta_transforms = None

        # NOTE: Replaced ActivationsAndGradients with differentiable version
        self.activations_and_grads = DifferentiableActivationsAndGradients(
            model, target_layers, reshape_transform
        )


    def get_cam_weights(self, input_tensor, target_layer, target_category,
                        activations, grads):
        # 2D image
        if len(grads.shape) == 4:
            return torch.mean(grads, dim=(2, 3))

        # 3D image
        elif len(grads.shape) == 5:
            return torch.mean(grads, dim=(2, 3, 4))

        else:
            raise ValueError("Invalid grads shape." 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")


    def compute_cam_per_layer(self, input_tensor, targets, eigen_smooth):
        activations_list = self.activations_and_grads.activations
        grads_list = self.activations_and_grads.gradients
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i, target_layer in enumerate(self.target_layers):
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets,
                                     layer_activations, layer_grads,
                                     eigen_smooth)
            cam = torch.maximum(cam, torch.tensor(0))
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer


    def aggregate_multi_layers(self, cam_per_target_layer):
        # Early return, if there's only one
        if len(cam_per_target_layer) == 1:
            return cam_per_target_layer[0]

        cam_per_target_layer = torch.cat(cam_per_target_layer, dim=1)
        cam_per_target_layer = torch.maximum(cam_per_target_layer, torch.tensor(0))
        result = torch.mean(cam_per_target_layer, dim=1)

        # Normalize CAM
        return scale_cam_image(result)


    # HACK: Handle case where model device changed since this object was made.
    # HACK: Ensure gradients from computing GradCAM aren't kept
    def forward(self, *args, **kwargs):
        # Ensure device is correct (in case model device changed)
        self.device = next(self.model.parameters()).device

        # Compute GradCAM
        out = super().forward(*args, **kwargs)

        # Zero out gradients from computing GradCAM
        self.model.zero_grad()

        return out


################################################################################
#                               Helper Functions                               #
################################################################################
def create_cam(model, target_layers=None):
    """
    Create GradCAM instance

    Parameters
    ----------
    model : torch.nn.Module
        Arbitrary model
    target_layers : list of torch.nn.Module, optional
        Specifies layers in model to use, by default None

    Returns
    -------
    pytorch_grad_cam.GradCAM
        GradCAM instance
    """
    # CASE 1: Target layer is provided
    if target_layers is not None:
        return DifferentiableGradCAM(model, target_layers=target_layers)
    # CASE 2: Attempt to extract from the model
    try:
        target_layer = load_model.get_last_conv_layer(model)
        return DifferentiableGradCAM(model, target_layers=[target_layer])
    except Exception as error_msg:
        raise NotImplementedError(f"Model `{type(model)}` does not have GradCAM logic implemented! \n\tError: {error_msg}")


def scale_cam_image(cam, target_size=None):
    """
    Differentiable re-implementation of scaling CAM

    Parameters
    ----------
    cam : torch.Tensor
        Computed GradCAM
    target_size : tuple, optional
        Desired size of GradCAM, by default None

    Returns
    -------
    torch.Tensor
        Resized CAM
    """
    # Normalize between 1-0
    cam = cam - cam.min()
    cam = cam / (1e-7 + cam.max())
    if target_size is not None:
        # NOTE: Perform bilinear interpolation to resize CAM
        cam = resize(cam, target_size, antialias=True)
    return cam
