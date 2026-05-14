import clip
import os
from torch import nn
import argparse
import torch
import random
from torch.cuda.amp import autocast
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
from torch.utils.data import Subset

from mm_eval_mlp import get_expert_dir, DEFAULT_EXPERT_NUM_EPOCHS, DEFAULT_EXPERT_NUM_PROXY_SAMPLES
from src.utils.constants import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, HOME_PATH
from peft import LoraConfig, get_peft_model
from src.multi_modal.datasets.datasets_utils import get_ds, get_imagenet_sample_indices
from torch.utils.data import DataLoader
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arch', type=str, default='vit_b32', choices=['vit_b16', 'vit_b32', 'vit_l14'])
    parser.add_argument('--dataset', type=str, default='pets', choices=['pets', 'flower102', 'eurosat', 'dtd', 'resisc45', 'food101', 'ucf', 'svhn', 'cifar100', 'imagenet'])
    parser.add_argument('--root_data_path', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_proxy_samples', type=int, default=500)
    parser.add_argument('--DPP', action='store_true')

    # Expert-training config; used to locate expert checkpoints saved by cv/scripts/run_train_lora.sh.
    parser.add_argument('--expert_num_epochs', type=int, default=DEFAULT_EXPERT_NUM_EPOCHS)
    parser.add_argument('--expert_num_proxy_samples', type=int, default=DEFAULT_EXPERT_NUM_PROXY_SAMPLES)

    return parser.parse_args()

def get_expert_path(args):
    return get_expert_dir(args.arch, args.dataset, args.DPP,
                          args.expert_num_epochs, args.expert_num_proxy_samples)

if __name__ == "__main__":
    args = parse_args()

    args.output_dir = os.path.join(f'{HOME_PATH}/outputs_train', f"{get_expert_path(args)}")

    print(args)

    device = f"cuda" if torch.cuda.is_available() else "cpu"

    if args.arch == "vit_b32":
        model, preprocess = clip.load('ViT-B/32', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    elif args.arch == "vit_b16":
        model, preprocess = clip.load('ViT-B/16', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    elif args.arch == "vit_l14":
        model, preprocess = clip.load('ViT-L/14', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    else:
        raise ValueError(f"unknown model: {args.model}_{args.arch}")

    train_dataset_list = []
    
    private_train_dataset, private_val_dataset = get_ds(args.dataset, args, preprocess)
    if args.dataset == "imagenet":
        indices = list(range(0, len(private_train_dataset)))
        random.shuffle(indices)
        private_train_dataset = Subset(private_train_dataset, indices[0:3000])

    train_dataset_list.append(private_train_dataset)

    imagenet_train_dataset, imagenet_val_dataset = get_ds("imagenet", args, preprocess)
    proxy_dataset_indices = get_imagenet_sample_indices(args, args.dataset)
    proxy_subset = Subset(imagenet_train_dataset, proxy_dataset_indices)
    train_dataset_list.append(proxy_subset)

    train_dataset = ConcatDataset(train_dataset_list)
    print(f"number of private and proxy samples: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=False, drop_last=True)
    
    model.visual = PeftModel.from_pretrained(model.visual, args.output_dir)

    model.visual.eval()

    model.visual.to(device)

    print(model.visual)

    def get_leaf_modules_with_grad(model):
        module_list = []
        name_list = []
        for name, module in model.named_modules():
            if name.endswith("mlp"):
                module.name = name
                module_list += [module]
                name_list += [name]
        return module_list
    
    alignment_embedding = [{} for _ in range(len(train_loader))]
    index = 0

    # Process batches
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Computing embeddings")):
        images = images.to(device)
        labels = labels.to(device)

        hooks = []
        alignment_embedding_per_data = alignment_embedding[index]

        def track_embedding_hook(module, input, output):
            alignment_embedding_per_data[module.name] = input[0].mean(axis=(0,1)).detach()
            torch.cuda.empty_cache()
            return output

        # Register hooks for all target modules
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            hook = layer.register_forward_hook(track_embedding_hook)
            hooks.append(hook)

        with autocast():
            with torch.no_grad():
                model.visual(images)

        # Remove hooks
        for hook in hooks:
            hook.remove()
        hooks = []
        index += 1


    ## compute the mean of the embeddings
    mean_embeddings = {}
    
    # First, collect all embeddings for each module
    for batch_embeddings in alignment_embedding:
        for module_name, embedding in batch_embeddings.items():
            if module_name not in mean_embeddings:
                mean_embeddings[module_name] = []
            mean_embeddings[module_name].append(embedding)
    
    # Compute mean for each module
    for module_name, embeddings in mean_embeddings.items():
        # Stack all embeddings and compute mean
        stacked_embeddings = torch.stack(embeddings)
        mean_embedding = stacked_embeddings.mean(dim=0)
        mean_embeddings[module_name] = mean_embedding
        print(f"Module {module_name} mean embedding shape: {mean_embedding.shape}")

    # # Save mean embeddings
    save_path = os.path.join(args.output_dir, "router_embedding.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(mean_embeddings, save_path)
    print(f"Saved mean embeddings to {save_path}")
