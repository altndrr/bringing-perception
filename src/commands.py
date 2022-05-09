"""Commands of the cli."""

import os
import shutil
import zipfile

import gdown


def download_datasets(*args, **kwargs):
    """Download the datasets."""
    dataset = kwargs.get("<dataset>", None)

    assert dataset is not None

    if dataset == "lfw":
        output_dir = "./datasets/lfw/"
        os.makedirs(output_dir, exist_ok=True)

        drive_ids = {"train": "1PY8rxEGxNruwNajEEkJgq3cSXIUg_R7R"}
        for split, drive_id in drive_ids.items():
            download_path = f"./datasets/lfw_{split}.zip"
            output_path = f"./datasets/lfw/{split}"

            gdown.download(id=drive_id, output=download_path)

            with zipfile.ZipFile(download_path) as file:
                file.extractall(output_dir)

            shutil.move(os.path.join(output_dir, "lfw"), output_path)
    else:
        raise ValueError(f"{dataset} is not a known dataset")
