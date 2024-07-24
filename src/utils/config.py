"""
config.py

Description: Used to load configuration file with specification
"""

# Standard libaries
import os

# Non-standard libraries
from configobj import ConfigObj
from configobj.validate import Validator

# Custom libraries
from src.data import constants


################################################################################
#                                  Functions                                   #
################################################################################
def load_config(calling_file, config_fname=None, copy=True):
    """
    Load configuration file (located in specific config. subdirectory). Load
    config specification file as well to validate configuration file.

    Note
    ----
    If not provided, only load defaults from config. specs.

    Parameters
    ----------
    calling_file : str
        Path to calling file whose configs to load
    config_fname : str, optional
        Name of configuration file
    copy : bool, optional
        If True, copy default (unspecified) parameters to config.

    Returns
    -------
    ConfigDict
        Loaded configurations dictionary
    """
    # Ensure configurations directories exists
    assert os.path.exists(constants.DIR_CONFIG), "Configs directory doesn't exist!"
    assert os.path.exists(constants.DIR_CONFIG_SPECS), "Config Specs directory doesn't exist!"

    # Check if config dir. (specific for this file) exists
    curr_fname = os.path.basename(calling_file).split(".")[0]
    DIR_CONFIG_CURR_FILE = os.path.join(constants.DIR_CONFIG, curr_fname)
    assert os.path.exists(DIR_CONFIG_CURR_FILE), f"Configs directory for `{curr_fname}` doesn't exist!"

    # Check that provided file exists
    config_path = None
    if config_fname:
        config_path = os.path.join(DIR_CONFIG_CURR_FILE, config_fname)
        assert os.path.exists(config_path), f"Configuration doesn't exist at `{config_path}`!"

    # Load configuration spec
    config_spec_path = os.path.join(constants.DIR_CONFIG_SPECS, f"{curr_fname}.ini")
    assert os.path.exists(config_spec_path), f"Config spec. file doesn't exist at `{config_spec_path}`!"
    config_spec = ConfigObj(config_spec_path, interpolation=False,
                            list_values=False, _inspec=True)

    # If config not provided, simply provide defaults
    if not config_fname:
        return config_spec

    # Load configuration
    print(f"Loading configuration file at: `{config_path}`")
    conf = ConfigObj(config_path, configspec=config_spec)
    conf.validate(Validator(), copy=copy)
    return conf


def flatten_nested_dict(dictionary):
    """
    Flatten nested dictionary recursively

    Parameters
    ----------
    dictionary : dict
        Arbitrarily nested dictionary

    Returns
    -------
    dict
        Flattened dictionay
    """
    items = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            items.extend(flatten_nested_dict(value).items())
        else:
            items.append((key, value))
    return dict(items)
