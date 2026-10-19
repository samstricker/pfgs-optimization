"""Module for loading configuration from a TOML file."""

from argparse import ArgumentParser
from typing import Dict, Tuple

import toml


def load_config() -> Tuple[Dict, Dict]:
    """Load configuration from a TOML file.
    File is expected to have sections [parameters] and [files] and the filepath to the config as a flag --config.

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple containing the configuration dictionary and files dictionary.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the TOML configuration file."
    )
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as config_file:
        data = toml.load(config_file)
    config = data.get("parameters", {})
    files = data.get("files", {})
    return config, files
