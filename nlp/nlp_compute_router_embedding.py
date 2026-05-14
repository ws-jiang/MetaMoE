"""
Testing LLMs on Benchmarks
"""
import argparse
import json
import os
import re
from typing import List

import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    GenerationConfig,
)
from transformers.pipelines.pt_utils import KeyDataset

from data import get_formatted_datasets
from nlp_eval_ours import (
    get_expert_dir, get_proxy_data_dir,
    DEFAULT_EXPERT_NUM_SAMPLES, DEFAULT_EXPERT_PROXY_DATA_NUM_SAMPLES,
)
from src import PeftConfig, PeftModelForCausalLM
from src.utils.constants import DEFAULT_CACHE_DIR, ACCESS_TOKEN
from src.utils.embedding_utils import track_embedding

from pathlib import Path


HOME_PATH = Path().resolve()
print(f'HOME_PATH: {HOME_PATH}')

transformers.set_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as ds_Dataset

class SimpleUnionDataset(Dataset):
    def __init__(self, datasets: List[ds_Dataset]):
        self.datasets = datasets
        self.num_labels = len(datasets)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        for ds in self.datasets:
            if idx < len(ds):
                return {"text": ds[idx]['text']}
            idx -= len(ds)
        raise IndexError(f"Index {idx} is out of bounds for the combined dataset")

if __name__ == '__main__':
    # Add arguments
    parser = argparse.ArgumentParser(
        description='Fine-tuning LLMs on training data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument(
        '--model_path', type=str, default='outputs/llama-2-7b-hf-adamole-the8-commonsense-qa',
        help='huggingface model id or local model path')
    parser.add_argument(
        '--data_path', type=str, default='tau/commonsense_qa',
        help='huggingface data id or local data path')
    parser.add_argument(
        '--max_new_tokens', type=int, default=16,
        help='maximum number of new tokens')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='batch size in the pipeline')
    parser.add_argument(
        '--logits', default=False, action='store_true',
        help='checking choice logits instead of generated texts')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='number of samples to evaluate')
    parser.add_argument(
        '--proxy_data_path', type=str, default=None,
        help='path to proxy data')
    parser.add_argument(
        '--proxy_data_num_samples', type=int, default=0,
        help='number of samples to use from proxy data')
    parser.add_argument('--use_base', action='store_true')
    parser.add_argument('--PROXY', action='store_true')
    parser.add_argument('--DPP', action='store_true')
    parser.add_argument(
        '--expert_num_samples', type=int, default=DEFAULT_EXPERT_NUM_SAMPLES,
        help='private-sample count the experts were trained on (used to locate expert checkpoints)')
    parser.add_argument(
        '--expert_proxy_data_num_samples', type=int, default=DEFAULT_EXPERT_PROXY_DATA_NUM_SAMPLES,
        help='proxy-sample count mixed into expert training (used to locate expert checkpoints)')

    # Parse arguments
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path
    model_name = os.path.basename(model_path).lower()
    model_basename = os.path.basename(model_path)  # original case, used to build checkpoint dir names
    data_name = os.path.basename(data_path).lower()
    split = "train"
    is_chat = True if "instruct" in model_name or "chat" in model_name else False

    # Load and format datasets
    train_dataset_list = []
    private_datasets = get_formatted_datasets(data_path=data_path, prompt_only=True, is_chat=is_chat)
    if args.num_samples > 0:
        print("### Use private data")
        private_datasets[split] = private_datasets[split].take(min(args.num_samples, len(private_datasets[split])))
    
    train_dataset_list.append(private_datasets[split])
    
    if args.proxy_data_num_samples > 0:
        print("### Use proxy data")
        temp_proxy_data_path = get_proxy_data_dir(model_basename, data_name)
        if args.DPP:
            proxy_data_path = f"{HOME_PATH}/outputs_train/{temp_proxy_data_path}/DPP_public_with_predictions"
        else:
            proxy_data_path = f"{HOME_PATH}/outputs_train/{temp_proxy_data_path}/public_with_predictions"
    
        proxy_data_num_samples = args.proxy_data_num_samples
        proxy_data_datasets = get_formatted_datasets(proxy_data_path, True, is_chat)
        proxy_data_datasets[split] = proxy_data_datasets[split].take(proxy_data_num_samples)
        train_dataset_list.append(proxy_data_datasets[split])
    else:
        print("### Do NOT use proxy data")
    
    union_dataset = SimpleUnionDataset(train_dataset_list)

    # Load the configuration and model
    if args.use_base:
        print(f"### Use base model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=DEFAULT_CACHE_DIR,
            token=ACCESS_TOKEN,
            # torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            cache_dir=DEFAULT_CACHE_DIR,
            token=ACCESS_TOKEN,
        )
    else:
        if args.PROXY:
            expert_dir = get_expert_dir(model_basename, data_name, args.DPP,
                                        args.expert_num_samples, args.expert_proxy_data_num_samples)
        else:
            # expert trained on private data only (no proxy)
            expert_dir = get_expert_dir(model_basename, data_name, dpp=False,
                                        num_samples=args.expert_num_samples, proxy_data_num_samples=0)
        model_path = os.path.join("outputs_train", expert_dir)
        print(f"### Use finetuned model (Proxy={args.PROXY}, DPP={args.DPP}): {model_path}")
        peft_config = PeftConfig.from_pretrained(model_path)
        print(f"### Use pretrained base model: {peft_config.base_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir=DEFAULT_CACHE_DIR,
            token=ACCESS_TOKEN,
            # torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            padding_side="left",
            cache_dir=DEFAULT_CACHE_DIR,
            token=ACCESS_TOKEN,
        )
        model = PeftModelForCausalLM.from_pretrained(model=base_model, model_id=model_path)

    tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    
    if args.use_base:
        os.makedirs(f"outputs/{model_name}", exist_ok=True)
        embedding_save_path = f"outputs/{model_name}/{data_name}_router_embedding.pt"
    else:
        os.makedirs(f"{model_path}", exist_ok=True)
        embedding_save_path = f"{model_path}/router_embedding.pt"
    
    track_embedding(union_dataset, model, tokenizer, embedding_save_path, 16, is_chat)

    print(f"Embedding saved to {embedding_save_path}")
