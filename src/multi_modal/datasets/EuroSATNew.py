import PIL
from scipy.io import loadmat
from torchvision.datasets import Flowers102, DTD, EuroSAT
from typing import Any, Tuple, Callable, Optional
import json
import os
from src.utils.constants import DEFAULT_DATA_DIR


class EuroSAT(EuroSAT):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(EuroSAT, self).__init__(root=root, transform=transform,
                                            target_transform=target_transform, download=download)

        self._labels = []
        self._image_files = []

        split_dict = self.read_split()
        root_path = f"{self.root}/eurosat/2750"
        for item in split_dict[split]:
            # self.samples.append((item[1], f"{root_path}/{item[0]}"))
            self._labels.append(item[1])
            self._image_files.append(f"{root_path}/{item[0]}")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def read_split(self):
        with open(os.path.join(DEFAULT_DATA_DIR, "zhou_data_splits/split_zhou_EuroSAT.json")) as f:
            return json.load(f)