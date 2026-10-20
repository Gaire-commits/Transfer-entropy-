import hashlib
from pathlib import Path

import yaml


def load(path):
    return yaml.safe_load(Path(path).read_text())


def digest(path):
    """sha256 of the config file; tests/test_frozen.py pins it so a registered rule can't drift."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
