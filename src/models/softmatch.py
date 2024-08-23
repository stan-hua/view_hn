"""
softmatch.py

Description: Stores implementation for semi-supervised learning method (SoftMatch)
"""

# Non-standard libraries
import torch
import torch.nn.functional as F


################################################################################
#                               Helper Functions                               #
################################################################################
def create_pseudo_labels(logits, use_hard_label=True, T=1.0,
                         softmax=True, label_smoothing=0.0):
    """
    Generate pseudo-labels from logits/probs

    Parameters
    ----------
    logits : torch.Tensor
        Logits (or probs, need to set softmax to False)
    use_hard_label : bool, optional
        If True, create hard label instead of soft labels
    T : float, optional
        Temperature parameter for prob. simplex smoothening/sharpening
    softmax : bool, optional
        If True, apply softmax on logits
    label_smoothing : float
        Label_smoothing parameter

    Returns
    -------
    torch.Tensor
        Contains pseudo-labels to compute loss on
    """
    logits = logits.detach()
    if use_hard_label:
        # return hard label directly
        pseudo_label = torch.argmax(logits, dim=-1)
        if label_smoothing:
            pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
        return pseudo_label
    
    # If not performing softmax, return logits
    if not softmax:
        return logits
    return torch.softmax(logits / T, dim=-1)


def smooth_targets(logits, targets, smoothing=0.1):
    """
    Apply label smoothening on logits

    Parameters
    ----------
    logits : torch.Tensor
        Logits
    targets : torch.Tensor
        Hard pseudo-label (from logits)
    smoothing : float
        Label smoothening value

    Returns
    -------
    torch.Tensor
        Contains smoothened targets
    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist


def consistency_loss(logits, targets, name='ce', mask=None):
    """
    Compute consistency regularization loss

    Parameters
    ----------
    logits : torch.Tensor
        Logits to calculate loss on, usually the strongly-augmented unlabeled
        samples
    targets : torch.Tensor
        Pseudo-labels (either hard label or soft label)
    name : str
        Name of loss to compute. "ce"=cross-entropy, "mse"=mean-squared error,
        "kl"=KL divergence
    mask : torch.Tensor
        Masks to remove samples when calculating the loss. Confidence can be
        used to compute mask, for example.
    """
    assert name in ['ce', 'mse', 'kl']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name == 'kl':
        loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
        loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
    else:
        loss = F.cross_entropy(logits, targets, reduction='none')

    if mask is not None and name != 'kl':
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()