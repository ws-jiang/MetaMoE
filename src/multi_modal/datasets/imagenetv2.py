import os
from typing import Callable, Optional

from src.multi_modal.datasets.imagenet import ImageNet, listdir_nohidden, Datum
from src.multi_modal.datasets.imagenet_a import ImageNet_A


class ImageNetV2(ImageNet_A):

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            args=None
    ) -> None:
        self.args = args
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_dir = os.path.join(root, "imagenetv2")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        self.dataset = self.read_data(classnames, "imgs")
        self.classes = self.read_classes(classnames)

    def read_data(self, classnames, split):
        image_dir = os.path.join(self.dataset_dir, split)
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            img_names = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in img_names:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def read_classes(self, classnames):
        folders = list(classnames.keys())
        classes = []

        for label in range(1000):
            folder = folders[label]
            classname = classnames[folder]
            classes.append(classname)

        return classes