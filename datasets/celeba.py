import torchvision.datasets.celeba as CELEBA
import datasets.dataset as DS
import gdown
import os

class CelebA(DS.ClassificationDataset):

    def __init__(
            self,
            name = 'celeba',
            root = None,
            transform = None,
            base=0,
            window=1,
            dataset = CELEBA.CelebA,
            target_type = 'identity',
            download=False,
            **kwargs
    ):

        download_path = os.path.join(root, 'celeba/img_align_celeba.zip')
        if not os.path.isfile(download_path):
            drive_ids = {
                'celeba': '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
            }
            split, drive_id = tuple(*drive_ids.items())


            gdown.download(id=drive_id, output=download_path)

        super(CelebA, self).__init__(
            name=name,
            root=root,
            transform=transform,
            base=base,
            window=window,
            dataset=dataset,
            target_type=target_type,
            download=download,
            **kwargs
        )

