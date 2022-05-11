import torchvision.datasets.coco as coco
import datasets.dataset as DS

class COCO(DS.DetectionDataset):

    def __init__(
            self,
            name = 'coco',
            root = '.',
            annFile = '.',
            transform = None,
            base=0,
            window=1,
            dataset = coco.CocoDetection,
            **kwargs
    ):

        super(COCO, self).__init__(
            name=name,
            root=root,
            annFile=annFile,
            transform=transform,
            base=base,
            window=window,
            dataset=dataset,
            **kwargs
        )