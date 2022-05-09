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

from docopt import docopt

from src import __version__
from src.commands import download_datasets

COMMANDS = {
    "download": download_datasets
}


def main():
    """Main execution."""
    options = docopt(__doc__, version=__version__)

    for key in options:
        command = COMMANDS.get(key)
        if options[key] is True and command is not None:
            command(**options)
