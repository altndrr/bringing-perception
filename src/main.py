"""
src

Usage:
    src download <dataset>
    src prepare <dataset_dir> [--num VAL]
    src test <model> [<query_dir>] [<gallery_dir>] [--submit]
    src -h | --help
    src --version

Options:
    --submit                    Submit the results or not.
    -n VAL --num=VAL            Num of triplets to generate
    -h --help                   Show the help screen
    --version                   Print the current version
"""

import random
import sys
from warnings import warn

import numpy as np
import torch
from docopt import docopt

from src import __version__
from src.commands import download_datasets, prepare_data, test_gallery

COMMANDS = {"download": download_datasets, "prepare": prepare_data, "test": test_gallery}


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
            sys.exit()

    warn("command not recognised")
