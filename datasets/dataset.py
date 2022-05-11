from torchvision.datasets import vision

class Dataset(vision.VisionDataset):

    def __init__(
            self,
            name,
            root = None,
            transform = None,
            base=0,
            window=1,
            dataset = None,
            download = True,
            **kwargs
    ):

        self.name, self.root, self.transform, self.base, self.window, self.dataset = \
            name, root, transform, base, window, dataset
        assert self.dataset is not None, f'Dataset attribute must not be none. Found {self.dataset}'


class ClassificationDataset(Dataset):

    def __init__(
            self,
            name,
            root = None,
            transform = None,
            base=0,
            window=1,
            dataset = None,
            download = True,
            **kwargs
    ):
        super(ClassificationDataset,self).__init__(
            name=name,
            root=root,
            transform=transform,
            base=base,
            window=window,
            dataset=dataset,
            download=download,
            **kwargs
        )

        splits = list()

        for split in ('train', 'valid', 'test'):
            splits.append(self.dataset(
                root=self.root,
                transform=self.transform,
                split=split,
                download=download,
                **kwargs
            ))

        self.train, self.val, self.test = splits

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

class DetectionDataset(Dataset):

    def __init__(
            self,
            name,
            root = None,
            transform = None,
            base=0,
            window=1,
            dataset = None,
            download = True,
            **kwargs
    ):
        super(DetectionDataset,self).__init__(
            name=name,
            root=root,
            transform=transform,
            base=base,
            window=window,
            dataset=dataset,
            **kwargs
        )

        self.dataset(
            root=self.root,
            transform=self.transform,
            **kwargs
        )
