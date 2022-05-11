import torchvision.datasets.imagenet as INET
import datasets.dataset as DS

class ImageNet(DS.ClassificationDataset):

    def __init__(
            self,
            name = 'imagenet',
            root = None,
            transform = None,
            base=0,
            window=1,
            dataset = INET.ImageNet,
            **kwargs
    ):

        super(ImageNet, self).__init__(
            name=name,
            root=root,
            transform=transform,
            base=base,
            window=window,
            dataset=dataset,
            **kwargs
        )

