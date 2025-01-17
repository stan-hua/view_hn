"""
efficientnet_pytorch_utils.py

Description: Used to make utility functions in certain cases.
"""

# Non-standard libraries
import torch
from efficientnet_pytorch import utils
from torch.utils import model_zoo


################################################################################
#                               Helper Functions                               #
################################################################################
# NOTE: Following function is taken from `efficientnet_pytorch.utils`
def load_pretrained_weights(
        model, model_name,
        weights_path=None,
        load_fc=True,
        advprop=False,
        verbose=True):
    """
    Loads pretrained weights from weights path or download using url.

    Note:
        Changed to not be sensitive to extra model layers.

    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        advprop (bool): Whether to load pretrained weights
                        trained with advprop (valid when weights_path is None).
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = utils.url_map_advprop if advprop else utils.url_map
        state_dict = model_zoo.load_url(url_map_[model_name])

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        # NOTE: Ignore case when extra layers are present
        # assert set(ret.missing_keys) == set(
        #     ['_fc.weight', '_fc.bias']), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    assert not ret.unexpected_keys, \
        'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)

    if verbose:
        print('Loaded pretrained weights for {}'.format(model_name))
