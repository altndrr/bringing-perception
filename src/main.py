"""
src

Usage:
    src download <dataset>
    src -h | --help
    src --version

Options:
    -h --help                   Show the help screen
    --version                   Print the current version
"""

import random

import numpy as np
import torch
from docopt import docopt

from src import __version__
from src.commands import download_datasets

COMMANDS = {"download": download_datasets}


def main():
    """Main execution."""
    options = docopt(__doc__, version=__version__)

    # Set determinism.
    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    for key in options:
        command = COMMANDS.get(key)
        if options[key] is True and command is not None:
            command(**options)
