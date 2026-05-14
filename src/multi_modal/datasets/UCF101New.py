
from typing import Any, Tuple, Callable, Optional
import json
import os
import PIL

from src.utils.constants import DEFAULT_DATA_DIR


class UCF101:
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        # is_train = split=="train"
        # super(UCF101, self).__init__(root=root, train = is_train, transform=transform,
        #                              annotation_path="",
        #                              frames_per_clip=1,
        #                              )

        self._images_folder = f"{root}/ucf101/UCF-101-midframes"
        self._labels = []
        self._image_files = []
        split_dict = self.read_split()
        self.transform = transform
        self.target_transform = target_transform

        self.class_name_dict = {}
        for item in split_dict[split]:
            self._labels.append(item[1])
            self._image_files.append(f"{self._images_folder}/{item[0]}")
            self.class_name_dict[item[1]] = item[2]

        self.classes = [self.class_name_dict[key] for key in sorted(self.class_name_dict.keys())]

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
        with open(os.path.join(DEFAULT_DATA_DIR, "zhou_data_splits/split_zhou_UCF101.json")) as f:
            return json.load(f)