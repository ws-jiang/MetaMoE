"""
Fine-Tuning LLMs on Tasks
"""
import argparse
import os
import re
from typing import List

from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as ds_Dataset

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from data import get_formatted_datasets
from src import (
    TaskType,
    LoraConfig,
    PeftTrainer,
    PeftModelForCausalLM,
)
from src.utils.constants import DEFAULT_CACHE_DIR, ACCESS_TOKEN
from nlp_eval_ours import get_proxy_data_dir

from pathlib import Path
HOME_PATH = Path().resolve()
print(f'HOME_PATH: {HOME_PATH}')

transformers.set_seed(0)

class SimpleUnionDataset(Dataset):
    def __init__(self, datasets: List[ds_Dataset]):
        self.datasets = datasets
        self.num_labels = len(datasets)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)
    
    def __getitem__(self, idx):
        for ds in self.datasets: 
            if idx < len(ds):
                return ds[idx]
            idx -= len(ds)
        raise IndexError(f"Index {idx} is out of bounds for the combined dataset")  

if __name__ == '__main__':
    # Add arguments
    parser = argparse.ArgumentParser(
        description='Fine-tuning LLMs on training data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument(
        '--model_path', type=str, default='meta-llama/Llama-2-7b-hf',
        help='huggingface model id or local model path')
    parser.add_argument(
        '--data_path', type=str, default='tau/commonsense_qa',
        help='huggingface data id or local data path')
    parser.add_argument(
        '--peft_type', type=str, default='lora', choices=['lora'],
        help='peft model type to be fine-tuned')
    parser.add_argument(
        '--lora_rank', type=int, default=32,
        help='lora rank')
    parser.add_argument(
        '--target_modules', type=str, default=['q_proj', 'v_proj'], nargs='+',
        help='target modules in lora layers')
    parser.add_argument(
        '--max_length', type=int, default=256,
        help='maximum number of tokens')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='batch size in the trainer')
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='gradient accumulation steps')
    parser.add_argument(
        '--num_train_epochs', type=int, default=1,
        help='number of training epochs')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='learning rate for training')
    parser.add_argument(
        '--lr_scheduler_type', type=str, default="constant_with_warmup",
        help='learning rate scheduler type')
    parser.add_argument(
        '--warmup_steps', type=int, default=200,
        help='number of warmup steps for training')
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='weight decay')
    parser.add_argument(
        '--aux_loss_coeff', type=float, default=None,
        help='auxiliary loss coefficient for moe')
    parser.add_argument('--job_id', type=str,
                        help='job_id for saving results')
    parser.add_argument(
        '--num_samples', type=int, default=-1,
        help='number of samples to evaluate, -1 for all samples')
    parser.add_argument(
        '--proxy_data_path', type=str, default=None,
        help='path to proxy data')
    parser.add_argument(
        '--proxy_data_num_samples', type=int, default=0,
        help='number of samples to use from proxy data')
    parser.add_argument(
        '--DPP', default=False, action='store_true',
        help='use DPP when creating the proxy data')


    # Parse arguments
    args = parser.parse_args()
    print(f'Arguments: {args}')
    model_path = args.model_path
    data_path = args.data_path
    model_name = os.path.basename(model_path).lower()
    model_basename = os.path.basename(model_path)  # original case, used to build checkpoint dir names
    data_name = os.path.basename(data_path).lower()
    max_length = args.max_length
    lora_rank = args.lora_rank
    lora_alpha = lora_rank * 2
    lora_dropout = 0.0
    output_dir = os.path.join('outputs_train', f"{args.job_id}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_chat = True if "instruct" in model_name or "chat" in model_name else False

    # Load and format datasets
    # construct the text (question + answer)
    formatted_datasets = get_formatted_datasets(data_path=data_path, prompt_only=False, is_chat=is_chat)
    if args.num_samples > 0:
        formatted_datasets["train"] = formatted_datasets["train"].take(min(args.num_samples, len(formatted_datasets["train"])))

    # Print 20 samples from the training split for inspection
    num_preview_samples = min(50, len(formatted_datasets["train"]))
    for idx in range(num_preview_samples):
        print(f"Train sample {idx}: {formatted_datasets['train'][idx]}")
    
    # raise Exception("Stop here")


    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        # add_bos_token=True,
        add_eos_token=True,
        cache_dir=DEFAULT_CACHE_DIR,
        token=ACCESS_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets
    if is_chat:
        tokenize_text = lambda examples: tokenizer.apply_chat_template(
            examples["text"],
            add_generation_prompt=True,
            continue_final_message=False,
            padding=True,
            # return_tensors="pt",
            return_dict=True,
        )
    else:
        tokenize_text = lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            # add_generation_prompt=True,
        )
    tokenized_datasets = formatted_datasets.map(
        tokenize_text,
        batched=True,
        remove_columns=formatted_datasets["train"].column_names,
    )
    print(f'Tokenized datasets: {tokenized_datasets}')

    if args.proxy_data_num_samples > 0:
        temp_proxy_data_path = get_proxy_data_dir(model_basename, data_name)
        if args.DPP:
            proxy_data_path = f"{HOME_PATH}/outputs_train/{temp_proxy_data_path}/DPP_public_with_predictions"
        else:
            proxy_data_path = f"{HOME_PATH}/outputs_train/{temp_proxy_data_path}/public_with_predictions"
        proxy_data_num_samples = args.proxy_data_num_samples
        proxy_data_datasets = get_formatted_datasets(proxy_data_path, True, is_chat)
        tokenized_proxy_data_datasets = proxy_data_datasets.map(
            tokenize_text,
            batched=True,
            remove_columns=proxy_data_datasets["train"].column_names,
        )
        tokenized_proxy_data_datasets['train'] = tokenized_proxy_data_datasets['train'].take(proxy_data_num_samples)
        print(f'Tokenized proxy data datasets: {tokenized_proxy_data_datasets}')
        tokenized_datasets['train'] = SimpleUnionDataset([tokenized_datasets['train'], tokenized_proxy_data_datasets['train']])
        

    # Set the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt")

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=DEFAULT_CACHE_DIR,
        token=ACCESS_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # low_cpu_mem_usage=True,
    )
    print(f'Base model loaded from {model_path}')
    print(f'Base model: {base_model}')

    # Get the PEFT model
    peft_config = LoraConfig(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = PeftModelForCausalLM(base_model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print(f'Model: {model}')

    # Set the trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        group_by_length=True,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10000,
        save_strategy="epoch",
        # save_steps=1000,
        optim="adamw_torch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        # fp16=False,
        # bf16=True,
        seed=0,
        data_seed=0,
        report_to=["tensorboard"],
    )
    trainer = PeftTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        aux_loss_coeff=args.aux_loss_coeff,
    )
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as output_file:
        output_file.write(training_args.to_json_string())

    # Train the model
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True

    # Save the model
    trainer.save_model()
    trainer.save_state()
