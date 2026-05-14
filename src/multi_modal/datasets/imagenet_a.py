import os
from typing import Callable, Optional

import PIL

from src.multi_modal.datasets.imagenet import ImageNet, listdir_nohidden, Datum


class ImageNet_A:

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
        self.dataset_dir = os.path.join(root, "imagenet-adversarial")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        self.dataset = self.read_data(classnames, "imagenet-a")
        self.classes = self.read_classes_name("imagenet-a")

    def __len__(self) -> int:
        return len(self.dataset)

    def read_classes_name(self, split_dir):
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)
        split_dir = os.path.join(self.dataset_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())

        classes = []
        for label, folder in enumerate(folders):
            classname = classnames[folder]
            classes.append(classname)
        return classes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_file = item.impath
        label = item.label
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.dataset_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items