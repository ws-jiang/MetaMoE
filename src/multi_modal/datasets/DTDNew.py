from scipy.io import loadmat
from torchvision.datasets import Flowers102, DTD
from typing import Any, Tuple, Callable, Optional
import json
import os
from src.utils.constants import DEFAULT_DATA_DIR


class DTD(DTD):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(DTD, self).__init__(root=root, split=split, transform=transform,
                                            target_transform=target_transform, download=download)

        self._labels = []
        self._image_files = []
        split_dict = self.read_split()
        for item in split_dict[split]:
            self._labels.append(item[1])
            self._image_files.append(self._images_folder / f"{item[0]}")

    def read_split(self):
        with open(os.path.join(DEFAULT_DATA_DIR, "zhou_data_splits/split_zhou_DescribableTextures.json")) as f:
            return json.load(f)