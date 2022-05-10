"""Commands of the cli."""

import os
import random
import shutil
import zipfile

import gdown
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def download_datasets(*args, **kwargs):
    """Download the datasets."""
    dataset = kwargs.get("<dataset>", None)

    assert dataset is not None

    if dataset == "lfw":
        output_dir = "./datasets/lfw/"
        os.makedirs(output_dir, exist_ok=True)

        drive_ids = {
            "train": "1PY8rxEGxNruwNajEEkJgq3cSXIUg_R7R",
            "test": "1VaahGkQc36dQy7q08AfEgiS4tqitSY4K",
        }
        for split, drive_id in drive_ids.items():
            download_path = f"./datasets/lfw_{split}.zip"
            output_path = f"./datasets/lfw/{split}"

            gdown.download(id=drive_id, output=download_path)
            with zipfile.ZipFile(download_path) as file:
                file.extractall(output_dir)

            if split == "train":
                shutil.move(os.path.join(output_dir, "lfw"), output_path)
            elif split == "test":
                for elem in ["query", "gallery"]:
                    shutil.move(
                        os.path.join(output_dir, elem), os.path.join(output_path, elem),
                    )
    else:
        raise ValueError(f"{dataset} is not a known dataset")


def prepare_data(*args, **kwargs):
    """Prepare the dataset to generate the query, the gallery and the triplets."""
    dataset_dir = kwargs.get("<dataset_dir>")
    parent_dir = os.path.dirname(dataset_dir)
    num_triplets = int(kwargs.get("--num", 1000))

    assert dataset_dir is not None

    ds = ImageFolder(dataset_dir)
    sizes = (int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8))
    train_set, val_set = torch.utils.data.random_split(ds, sizes)

    val_targets = np.array(ds.imgs)[:, 0][val_set.indices].tolist()
    random.shuffle(val_targets)
    val_query = val_targets[: int(len(val_targets) * 0.2)]
    val_gallery = val_targets[int(len(val_targets) * 0.2) :]
    with open(os.path.join(parent_dir, "query.txt"), "w", encoding="utf-8") as f:
        f.writelines("\n".join(val_query))
    with open(os.path.join(parent_dir, "gallery.txt"), "w", encoding="utf-8") as f:
        f.writelines("\n".join(val_gallery))

    train_targets = np.array(ds.targets)[train_set.indices]
    class_idx, num_samples = np.unique(train_targets, return_counts=True)
    positive_classes = class_idx[num_samples > 1]

    triplets = []
    for _ in tqdm(range(num_triplets)):
        class_idx = random.choice(positive_classes)
        positives = [
            ds.imgs[i][0] for i in train_set.indices if ds.targets[i] == class_idx
        ]
        negatives = [
            ds.imgs[i][0] for i in train_set.indices if ds.targets[i] != class_idx
        ]

        assert len(positives) > 1

        anchor = random.choice(positives)
        positive = None
        negative = None
        while positive == anchor or positive is None:
            positive = random.choice(positives)

        while negative is None or negative == anchor or negative == positive:
            negative = random.choice(negatives)

        triplet = ", ".join([anchor, positive, negative]) + "\n"
        triplets.append(triplet)

    with open(os.path.join(parent_dir, "triplets.txt"), "w", encoding="utf-8") as f:
        f.writelines(triplets)
