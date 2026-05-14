from scipy.io import loadmat
from torchvision.datasets import Flowers102
from typing import Any, Tuple, Callable, Optional
import json
import os   
from src.utils.constants import DEFAULT_DATA_DIR

class Flowers102New(Flowers102):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(Flowers102New, self).__init__(root=root, split=split, transform=transform,
                                            target_transform=target_transform, download=download)

        self._labels = []
        self._image_files = []
        split_dict = self.read_split()
        for item in split_dict[split]:
            self._labels.append(item[1])
            self._image_files.append(self._images_folder / f"{item[0]}")

        self.classes = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
            "sweet pea", "english marigold", "tiger lily", "moon orchid",
            "bird of paradise", "monkshood", "globe thistle", "snapdragon",
            "colt's foot", "king protea", "spear thistle", "yellow iris",
            "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
            "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
            "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
            "stemless gentian", "artichoke", "sweet william", "carnation",
            "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
            "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
            "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
            "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
            "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
            "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
            "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
            "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
            "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
            "azalea", "water lily", "rose", "thorn apple", "morning glory",
            "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
            "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
            "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
            "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
            "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
            "blackberry lily"
        ]

    def read_split(self):
        with open(os.path.join(DEFAULT_DATA_DIR, "zhou_data_splits/split_zhou_OxfordFlowers.json")) as f:
            return json.load(f)