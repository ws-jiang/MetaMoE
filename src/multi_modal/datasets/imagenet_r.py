import os
from typing import Callable, Optional

import PIL

from src.multi_modal.datasets.imagenet import ImageNet, listdir_nohidden, Datum
from src.multi_modal.datasets.imagenet_a import ImageNet_A


class ImageNet_R(ImageNet_A):

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
        self.dataset_dir = os.path.join(root, "imagenet-rendition")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        self.dataset = self.read_data(classnames, "imagenet-r")
        self.classes = self.read_classes_name("imagenet-r")
