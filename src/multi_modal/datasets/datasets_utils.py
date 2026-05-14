import argparse
import os.path
import json
import torchvision
from torchvision.datasets import CIFAR100, Food101, SVHN, CIFAR10, CLEVRClassification
from torch.utils.data import Dataset
from typing import List

from src.multi_modal.datasets.DTDNew import DTD
from src.multi_modal.datasets.EuroSATNew import EuroSAT
from src.multi_modal.datasets.Flowers102New import Flowers102New
from src.multi_modal.datasets.PetsNew import OxfordIIITPet
from src.multi_modal.datasets.SUN397New import SUN397
from src.multi_modal.datasets.UCF101New import UCF101
from src.multi_modal.datasets.imagenet import ImageNet
from src.multi_modal.datasets.imagenet_a import ImageNet_A
from src.multi_modal.datasets.imagenet_r import ImageNet_R
from src.multi_modal.datasets.imagenet_s import ImageNet_S
from src.multi_modal.datasets.imagenetv2 import ImageNetV2
from src.multi_modal.datasets.resisc45 import RESISC45
# from src.multi_modal.datasets.DTDNew import SVHN
from src.utils.constants import HOME_PATH


def get_ds(ds_name, args, preprocess, use_preprocess=True):
    normalize = torchvision.transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
    train_preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

    test_preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

    if not use_preprocess:
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
            ]
        )
        train_preprocess = preprocess
        test_preprocess = preprocess

    if ds_name == "cifar100":
        train_dataset = CIFAR100(os.path.join(args.root_data_path, "cifar100"), transform=preprocess,
                                 download=True, train=True)

        val_dataset = CIFAR100(os.path.join(args.root_data_path, "cifar100"), transform=preprocess,
                               download=True, train=False)
    elif ds_name == "cifar10":
        train_dataset = CIFAR10(os.path.join(args.root_data_path, "cifar10"), transform=preprocess,
                                 download=True, train=True)

        val_dataset = CIFAR10(os.path.join(args.root_data_path, "cifar10"), transform=preprocess,
                               download=True, train=False)
    elif ds_name == "svhn":
        train_dataset = SVHN(os.path.join(args.root_data_path, "svhn"), transform=preprocess,
                                 download=True, split="train")

        train_dataset.classes = [str(_) for _ in range(0, 10)]

        val_dataset = SVHN(os.path.join(args.root_data_path, "svhn"), transform=preprocess,
                               download=True, split="test")
    elif ds_name == "clevr":
        def minus_three(x):
            return x - 3
        train_dataset = CLEVRClassification(f"{args.root_data_path}", transform=preprocess,
                                 download=True, split="train", target_transform=minus_three)

        train_dataset.classes = ["three", "four", "five", "six", "seven", "eight", "nine", "ten"]

        val_dataset = CLEVRClassification(f"{args.root_data_path}", transform=preprocess,
                               download=True, split="val", target_transform=minus_three)
    elif ds_name == "resisc45":
        train_dataset = RESISC45(f"{args.root_data_path}", transform=train_preprocess,
                                 download=True, split="train")

        val_dataset = RESISC45(f"{args.root_data_path}", transform=test_preprocess,
                               download=True, split="test")
    elif ds_name == "dtd":
        train_dataset = DTD(args.root_data_path, transform=train_preprocess,
                            download=True, split="train")

        val_dataset = DTD(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "food101":
        train_dataset = Food101(os.path.join(args.root_data_path, "food101"), transform=train_preprocess,
                            download=True, split="train")

        val_dataset = Food101(os.path.join(args.root_data_path, "food101"), transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "flower102":
        train_dataset = Flowers102New(args.root_data_path, transform=train_preprocess,
                            download=True, split="train")

        val_dataset = Flowers102New(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "eurosat":
        train_dataset = EuroSAT(args.root_data_path, transform=train_preprocess,
                            download=True, split="train")

        val_dataset = EuroSAT(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "pets":
        train_dataset = OxfordIIITPet(args.root_data_path, transform=train_preprocess,
                            download=True, split="train")

        val_dataset = OxfordIIITPet(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "ucf":
        train_dataset = UCF101(args.root_data_path, transform=train_preprocess,
                            download=True, split="train")

        val_dataset = UCF101(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "sun397":
        train_dataset = SUN397(args.root_data_path, transform=train_preprocess,
                            download=True, split="train")

        val_dataset = SUN397(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "imagenet":
        train_dataset = ImageNet(args.root_data_path, transform=train_preprocess,
                            download=True, split="train")

        val_dataset = ImageNet(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "imagenet_a":
        train_dataset = None
        val_dataset = ImageNet_A(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "imagenet_r":
        train_dataset = None
        val_dataset = ImageNet_R(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "imagenet_v2":
        train_dataset = None
        val_dataset = ImageNetV2(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    elif ds_name == "imagenet_s":
        train_dataset = None
        val_dataset = ImageNet_S(args.root_data_path, transform=test_preprocess,
                          download=True, split="test")
    else:
        raise ValueError(f"unknown dataset: {ds_name}")

    return train_dataset, val_dataset

def get_imagenet_sample_indices(args, proxy_dataset):
    if args.DPP:
        json_path = os.path.join(HOME_PATH, "outputs_train/MM", f"{args.DPP}ProxyData_{args.arch}_{proxy_dataset}_imagenet_10000.json")
    else:
        json_path = os.path.join(HOME_PATH, "outputs_train/MM",
                                 f"ProxyData_{args.arch}_{proxy_dataset}_imagenet_10000.json")
    with open(json_path, "r") as f:
        private_data_list = json.load(f)
    private_data_list = private_data_list[:args.num_proxy_samples]  
    return [private_data["idx"] for private_data in private_data_list]

class SimpleUnionDatasetOffsetLabel(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.num_classes_list = [len(ds.classes) for ds in datasets]
 
    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        off_set = 0
        for idx_ds, ds in enumerate(self.datasets):
            if idx < len(ds):
                return  ds[idx][0], off_set + ds[idx][1]
            idx -= len(ds)
            off_set += len(ds.classes)
        raise IndexError(f"Index {idx} is out of bounds for the combined dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')
    args = parser.parse_args()
    args.root_data_path = "/home/jiangws/projects/data"
    ds_name = "imagenet_s"
    train_ds, valid_ds = get_ds(ds_name, args, None)
    # print(len(train_ds))
    print(len(valid_ds))