"""Commands of the cli."""

import json
import os
import random
import shutil
import zipfile
from glob import glob

import gdown
import numpy as np
import requests as r
import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import resnet50
from tqdm import tqdm

GROUP_NAME = "group_3"
SUBMISSION_URL = "http://coruscant.disi.unitn.it:3001/results/"

BATCH_SIZE = 64


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
    num_triplets = int(kwargs.get("--num") or 1000)

    assert dataset_dir is not None

    ds = ImageFolder(dataset_dir)

    targets = np.array(ds.targets)
    class_idx, num_samples = np.unique(targets, return_counts=True)
    positive_classes = class_idx[num_samples > 1]

    triplets = []
    for _ in tqdm(range(num_triplets)):
        class_idx = random.choice(positive_classes)
        positives = [path for path, target in ds.imgs if target == class_idx]
        negatives = [path for path, target in ds.imgs if target != class_idx]

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


def test_gallery(*args, **kwargs):
    """Test a model on a query-gallery system."""
    checkpoint = kwargs.get("<model>")
    query_dir = kwargs.get("<query_dir>") or "./datasets/lfw/test/query/"
    gallery_dir = kwargs.get("<gallery_dir>") or "./datasets/lfw/test/gallery/"

    assert checkpoint is not None

    model = resnet50(num_classes=8631)
    weights = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(weights, strict=True)
    model.fc = torch.nn.Identity()

    query_images = glob(os.path.join(query_dir, "*"))
    gallery_images = glob(os.path.join(gallery_dir, "*"))

    query_embeddings = _forward_step(model, query_images)
    gallery_embeddings = _forward_step(model, gallery_images)

    results = {}
    gallery_names = np.array([os.path.basename(image) for image in gallery_images])
    for i, embedding in enumerate(query_embeddings):
        query_name = os.path.basename(query_images[i])
        distances = (gallery_embeddings - embedding).norm(dim=1)
        top_10 = gallery_names[distances.topk(10, largest=False)[1].to(int)]

        results[query_name] = top_10.tolist()

    if kwargs.get("--submit"):
        _submit(results)


def _forward_step(model, images):
    model.eval()
    model.to("cuda")

    num_batches = int(len(images) / BATCH_SIZE) + 1
    batch_start = lambda x: x * BATCH_SIZE
    batch_end = lambda x: (x * BATCH_SIZE) + BATCH_SIZE
    total_embeddings = torch.Tensor()
    for i in tqdm(range(num_batches), desc="forwarding...", total=num_batches):
        batch = _read_images(images[batch_start(i) : batch_end(i)]).to("cuda")
        embeddings = model(batch).detach().cpu()
        total_embeddings = torch.cat((total_embeddings, embeddings), dim=0)

    return total_embeddings


def _fixed_image_standardization(image):
    processed_tensor = (image - 127.5) / 128.0
    return processed_tensor


def _read_images(images):
    tensors = []

    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        selection_method="center_weighted_size",
        device="cuda",
    )

    transformations = T.Compose([np.float32, T.ToTensor(), T.Resize((160, 160))])

    for image in images:
        if isinstance(image, str):
            if not os.path.exists(image) or not os.path.isfile(image):
                raise FileNotFoundError(f"{image} is an invalid image path")

            image = Image.open(image)

        # Convert to RGB to fix the number of channels to three.
        image = image.convert("RGB")

        face = mtcnn(image)
        extract = face
        if extract is None:
            extract = _fixed_image_standardization(transformations(image))
        tensors.append(extract)

    tensors = torch.stack(tensors, dim=0)

    return tensors


def _submit(results):
    """
    Submit a json file to the system for evaluation.
    """
    assert isinstance(results, dict) and results is not None

    data = {}
    data["groupname"] = GROUP_NAME
    data["images"] = results

    res = json.dumps(data)

    response = r.post(SUBMISSION_URL, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
