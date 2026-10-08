"""
Small helper for loading the YAML configuration file.

Kept separate from main.py so that any module (camera, detection,
analytics, ...) can load config the same way later on.
"""

import yaml


def load_config(path="config/traffic_config.yaml"):
    """Load the YAML config file and return it as a dict."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
