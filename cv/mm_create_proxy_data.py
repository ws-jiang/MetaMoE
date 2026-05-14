import argparse
import random
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from mm_eval_mlp import load_clip_model
from src.multi_modal.datasets.datasets_utils import get_ds
from src.utils.constants import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, HOME_PATH
from src.utils.dpp_map import fast_map_dpp


class UnionLabeledDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length1 = len(dataset1)
        self.length2 = len(dataset2)
        self.total_length = self.length1 + self.length2

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < self.length1:
            x, *_ = self.dataset1[idx]
            label = 1  # Positive: from local_train_dataset
        else:
            x, *_ = self.dataset2[idx - self.length1]
            label = 0  # Negative: from public_train_dataset
        return x, label


class IndexedDataset(Dataset):
    """Wraps a dataset so each item also carries its index: (image, label, idx)."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, *rest = self.dataset[idx]
        label = rest[0] if rest else 0
        return x, label, idx


class SimpleClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.fc(x)
    
def encode_image(model, images):
    x = model.visual.conv1(images)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = model.visual.ln_post(x[:, 0, :])

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='create proxy data (from public data) for each client based on its local data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--public_data_path', type=str, default='imagenet',
        help='public data path')
    parser.add_argument(
        '--local_data_path', type=str, default='dtd',
        help='local data path')
    parser.add_argument(
        '--num_samples', type=int, default=10000,
        help='number of samples to take from the dataset')
    parser.add_argument(
        '--arch', type=str, default='vit_b32',
        help='huggingface model id or local model path')
    parser.add_argument('--root_data_path', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='batch size in the trainer')
    parser.add_argument(
        '--num_train_epochs', type=int, default=2,
        help='number of training epochs')
    parser.add_argument(
        '--gpu_id', type=str, default='0',
        help='gpu id')
    parser.add_argument(
        '--feature_dim', type=int, default=768,
        help='feature dimension')
    parser.add_argument(
        '--job_id', type=str, default='ProxyData',
        help='job id')
    parser.add_argument(
        '--DPP', default=False, action='store_true',
        help='DPP method')

    args = parser.parse_args()
    args.output_path = os.path.join(HOME_PATH, "outputs_train/MM", f"{args.DPP}ProxyData_{args.arch}_{args.local_data_path}_{args.public_data_path}_{args.num_samples}.json")

    device = torch.device(f'cuda')

    model, preprocess = load_clip_model(args, device)
    

    local_train_dataset, _ = get_ds(args.local_data_path, args, preprocess)
    public_train_dataset, _ = get_ds(args.public_data_path, args, preprocess)
    
    # Create a subdataset of public_train_dataset with 1000 randomly sampled samples
    random.seed(42)  # Set seed for reproducibility
    total_samples = len(public_train_dataset)
    indices = list(range(total_samples))
    random.shuffle(indices)
    sampled_indices = indices[:len(local_train_dataset)]
    public_train_subdataset = Subset(public_train_dataset, sampled_indices)
    
    print(f"Local dataset size: {len(local_train_dataset)}")
    print(f"Original public dataset size: {total_samples}")
    print(f"Sampled subdataset size: {len(public_train_subdataset)}")

    union_dataset = UnionLabeledDataset(local_train_dataset, public_train_subdataset)
    train_loader = torch.utils.data.DataLoader(union_dataset, batch_size=args.batch_size, shuffle=True, num_workers=96)

    classifier = SimpleClassifier(args.feature_dim)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for model and classifier.")
        model = nn.DataParallel(model)
        classifier = nn.DataParallel(classifier)
    classifier = classifier.to(device)
    model.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    for epoch in range(args.num_train_epochs):
        classifier.train()
        total_correct = 0
        total_samples = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device=device, dtype=model.module.visual.conv1.weight.dtype if isinstance(model, nn.DataParallel) else model.visual.conv1.weight.dtype)
            labels = labels.float().to(device)
            with torch.no_grad():
                features = encode_image(model.module if isinstance(model, nn.DataParallel) else model, images)
            features = features.to(dtype=classifier.module.fc.weight.dtype if isinstance(classifier, nn.DataParallel) else classifier.fc.weight.dtype)
            logits = classifier(features).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {100*accuracy:.2f}")

    # Evaluation on public_train_dataset (only first 2000 samples)
    model.eval()
    start_time = time.time()
    classifier.eval()
    # eval_subset = Subset(public_train_dataset, range(2000))
    eval_loader = torch.utils.data.DataLoader(IndexedDataset(public_train_dataset), batch_size=args.batch_size*32, shuffle=False, num_workers=96)
    eval_total = 0
    eval_correct = 0
    eval_loss = 0.0
    all_probs = []
    all_idx = []
    with torch.no_grad():
        for images, _, idx in tqdm(eval_loader, desc="Evaluating on public_train_dataset"):
            images = images.to(device=device, dtype=model.module.visual.conv1.weight.dtype if isinstance(model, nn.DataParallel) else model.visual.conv1.weight.dtype)
            features = encode_image(model.module if isinstance(model, nn.DataParallel) else model, images)
            features = features.to(dtype=classifier.module.fc.weight.dtype if isinstance(classifier, nn.DataParallel) else classifier.fc.weight.dtype)
            logits = classifier(features).squeeze(1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())
            all_idx.extend(idx.cpu().tolist())
            labels = torch.zeros_like(logits)  # All public_train_dataset samples are negative (label 0)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            eval_loss += loss.item() * images.size(0)
            preds = (probs > 0.5).float()
            eval_correct += (preds == labels).sum().item()
            eval_total += labels.size(0)
    eval_loss = eval_loss / eval_total if eval_total > 0 else 0
    eval_acc = eval_correct / eval_total if eval_total > 0 else 0
    print(f"Evaluation on public_train_dataset: Loss: {eval_loss:.4f}, Accuracy: {100*eval_acc:.2f}")

   

    if args.DPP:
        g_scores = all_probs

        # Sort g_scores in descending order and return the score with index
        g_scores_with_index = list(enumerate(g_scores))
        g_scores_with_index_sorted = sorted(g_scores_with_index, key=lambda x: x[1], reverse=True)

        # Get the top 100 scores
        top_2000_g_scores_tuple = g_scores_with_index_sorted[:2000]
        top_2000_indices = [index for index, _ in top_2000_g_scores_tuple]
        top_2000_scores = [score for _, score in top_2000_g_scores_tuple]
        
        # Compute the outer product of top_2000_scores
        
        top_2000_scores_np = np.array(top_2000_scores)
        outer_product_matrix = np.outer(top_2000_scores_np, top_2000_scores_np)

        top_sample_subset = Subset(public_train_dataset, top_2000_indices)
        top_sample_loader = torch.utils.data.DataLoader(IndexedDataset(top_sample_subset), batch_size=args.batch_size*32, shuffle=False, num_workers=96)

        all_features = []
        with torch.no_grad():
            for images, _, idx in tqdm(top_sample_loader, desc="Evaluating on public_train_dataset"):
                images = images.to(device=device, dtype=model.module.visual.conv1.weight.dtype if isinstance(model, nn.DataParallel) else model.visual.conv1.weight.dtype)
                features = encode_image(model.module if isinstance(model, nn.DataParallel) else model, images)
                all_features.append(features)
        
        # Concatenate all features into a single tensor
        all_features = torch.cat(all_features, dim=0)
        # Normalize features to unit vectors
        all_features = torch.nn.functional.normalize(all_features, p=2, dim=1)
        # Compute cosine similarity matrix
        similarity_matrix = all_features @ all_features.T

        # Convert similarity_matrix to numpy array
        similarity_matrix_np = similarity_matrix.cpu().numpy()
        kernel_matrix = similarity_matrix_np * outer_product_matrix

        seleted_items = fast_map_dpp(kernel_matrix, 500)
        seleted_items_indices = np.array(top_2000_indices)[seleted_items]

        top_items = [{"idx": all_idx[i], "prob": 1.0, "similarity": all_probs[i]} for i in seleted_items_indices] + [{"idx": all_idx[i], "prob": 0.0, "similarity": all_probs[i]} for i in top_2000_indices if i not in seleted_items_indices]
    else:
        sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)
        top_indices = sorted_indices[:args.num_samples]
        top_items = [{"idx": all_idx[i], "prob": all_probs[i]} for i in top_indices]
    
    elapsed_time = time.time() - start_time
    print(f"Time taken for DPP/score processing step: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(top_items, f)
    print(f"Saved proxy data to {args.output_path}")
    
    