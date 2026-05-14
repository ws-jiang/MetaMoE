"""
Testing LLMs on Benchmarks
"""
import argparse
import json
import os
import re
import time
from typing import List
import numpy as np

import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    GenerationConfig, DataCollatorForLanguageModeling, TrainingArguments, Trainer, Qwen2ForCausalLM,
)
from transformers.pipelines.pt_utils import KeyDataset

from data import get_formatted_datasets
from src import PeftConfig, PeftModelForCausalLM, MLPMoleConfig, TaskType
from src.mlp_mole.layer import MLPMoLELayer
from src.utils.constants import DEFAULT_CACHE_DIR, ACCESS_TOKEN
from src.utils.model_utils import set_direct_forward_base_layer, print_fraction_trainable_parameters

from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as ds_Dataset
from pathlib import Path
HOME_PATH = Path().resolve()

# transformers.set_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleUnionDataset(Dataset):
    def __init__(self, datasets: List[ds_Dataset]):
        self.datasets = datasets
        self.num_labels = len(datasets)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        for ds in self.datasets:
            if idx < len(ds):
                return  ds[idx]
            idx -= len(ds)
        raise IndexError(f"Index {idx} is out of bounds for the combined dataset")

def predict_choices(examples, tokenizer, model, is_chat=False):
    """
    Predict choices
    """
    if is_chat:
        # print(f"Chat text: {examples['text']}")
        inputs = tokenizer.apply_chat_template(
            examples["text"],
            add_generation_prompt=True,
            continue_final_message=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        inputs = tokenizer(examples['text'], return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]
    choices = [chr(ord('A') + i) for i in range(max(examples['num_choices']))]
    choice_ids = [tokenizer.encode(choice, add_special_tokens=False)[-1] for choice in choices]

    predicted_ids = torch.argmax(logits[:, choice_ids], dim=-1)
    predictions = [choices[predicted_id] for predicted_id in predicted_ids.cpu().numpy()]
    examples['prediction'] = predictions
    print(f"predictions: {predictions}, GT: {examples['answer']}")

    return examples

from transformers import LlamaForCausalLM

def get_leaf_modules_with_grad(model):
        module_list = []
        name_list = []
        for name, module in model.named_modules():
            if name.endswith("mlp"):
                module.name = name
                module_list += [module]
                name_list += [name]
        return module_list

class MyCustomLlamaForCausalLM(LlamaForCausalLM):



    def forward(self, *args, **kwargs):
        with torch.no_grad():
            # print(f"### pretrain_model is not None")
            set_direct_forward_base_layer(self, direct_forward_base_layer=True)

            hooks = []
            alignment_embedding_per_data = {}

            def track_embedding_hook(module, input, output):
                alignment_embedding_per_data[module.name] = input[0].mean(axis=(1)).detach()
                return output

            # Register hooks for all target modules
            leaf_modules_with_grad = get_leaf_modules_with_grad(self)
            for layer in leaf_modules_with_grad:
                hook = layer.register_forward_hook(track_embedding_hook)
                hooks.append(hook)

            super().forward(*args, **kwargs, output_hidden_states=True, return_dict=True)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # print(f"### alignment_embedding_per_data: {alignment_embedding_per_data.keys()}")
            # print(f"### alignment_embedding_per_data: {list(alignment_embedding_per_data.values())[0].shape}")
            # raise ValueError("stop here")


            # outputs = super().forward(*args, **kwargs, output_hidden_states=True, return_dict=True)
            # print(f"### outputs: {outputs.hidden_states}")
            # print(f"### outputs: {len(outputs.hidden_states)}")
            # raise ValueError("stop here")
            # last_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            idx = 0
            for name, module in self.named_modules():
                if name.endswith('moe_layer'):
                    # print(idx, name)
                    module['default'].sample_embeddings = alignment_embedding_per_data[f"model.layers.{idx}.mlp"]
                    idx += 1
            set_direct_forward_base_layer(self, direct_forward_base_layer=False)
            # raise ValueError("stop here")
 
        outputs = super().forward(*args, **kwargs)
        return outputs

class MyCustomQwen2ForCausalLM(Qwen2ForCausalLM):
    def forward(self, *args, **kwargs):
        with torch.no_grad():
            # print(f"### pretrain_model is not None")
            set_direct_forward_base_layer(self, direct_forward_base_layer=True)
            outputs = super().forward(*args, **kwargs, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.hidden_states[-1].mean(dim=1)

            for name, module in self.named_modules():
                if name.endswith('moe_layer'):
                    # print(f"### module name: {name}, type: {type(module)}")
                    module['default'].sample_embeddings = last_hidden_state
            set_direct_forward_base_layer(self, direct_forward_base_layer=False)

        outputs = super().forward(*args, **kwargs)
        return outputs

def evaluate_model(base_model, tokenizer, device, max_new_tokens, batch_size, max_length, args, model_name, data_path_list=None):
    ## evaluate the model
    is_chat = True if "instruct" in model_name or "chat" in model_name else False
    base_model.eval()

    ## Load and format datasets
    if data_path_list is None:
        data_path_list = [
             "tau/commonsense_qa",
            "allenai/cosmos_qa",
            "allenai/social_i_qa",
        ]
    metrics = []
    accs = []
    for data_path in data_path_list:
        data_name = os.path.basename(data_path).lower()
        if data_name in ['openbookqa', 'ai2_arc']:
            split = 'test' 
        else:
            split = 'validation'
        formatted_datasets = get_formatted_datasets(data_path=data_path, prompt_only=True, is_chat=is_chat)
        if args.num_samples > 0:
            formatted_datasets[split] = formatted_datasets[split].take(min(args.num_samples, len(formatted_datasets[split])))

        if not args.logits:
            # Build the pipeline
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            pipeline = TextGenerationPipeline(
                model=base_model,
                tokenizer=tokenizer,
                device=device,
            )

            # Get the model responses
            responses = []
            for response in tqdm(
                pipeline(
                    KeyDataset(formatted_datasets[split], 'text'),
                    generation_config=generation_config,
                    return_full_text=False,
                    batch_size=batch_size,
                ),
                total=len(formatted_datasets[split]),
            ):
                responses.append(response[0]['generated_text'])

            # Print one response
            print(f'Response example:\n{responses[0]}')

            # Get the results
            df = formatted_datasets[split].to_pandas()
            df['response'] = responses
            df['prediction'] = df['response'].str.extract(pat=r'\b([A-Z])\b')[0]
        else:
            # Get predictions
            dataset_with_predictions = formatted_datasets[split].map(
                lambda x: predict_choices(x, tokenizer=tokenizer, model=base_model, is_chat=is_chat), 
                batched=True, batch_size=batch_size)
            df = dataset_with_predictions.to_pandas()

        # Save the results
        result_path = os.path.join("results", f'{args.job_id}_{data_name}_eval_results.csv')
        df.to_csv(result_path, index=False)
        print(f'Results saved to {result_path}')

        # Compute evaluation metrics
        
        for _data_name in df['data_name'].unique():
            df_set = df[df['data_name'] == _data_name]
            accuracy = pd.Series(df_set['answer'] == df_set['prediction']).mean()
            accs.append(accuracy*100)
            print(f'Accuracy of {_data_name}: {accuracy:.2%}')
            metrics.append({
                'data_name': data_name,
                'accuracy': accuracy,
                'model_name': model_name,
                'lambda_': args.lambda_,
            })
    
    results = accs + [sum(accs) / len(accs)]
    print("### average accuracy: ", ",".join([f"{acc:.2f}" for acc in results]))

    # Save evaluation metrics
    metric_path = os.path.join("results", f'{args.job_id}_eval_metrics.csv')
    with open(metric_path, 'w') as file:
        json.dump(metrics, file)
    print(f'Metrics saved to {metric_path}')

# Client domains unified by MetaMoE (see the paper).
CLIENT_DATASETS = ["commonsense_qa", "cosmos_qa", "social_i_qa"]

# Default training config used in the paper (see nlp/scripts/).
DEFAULT_EXPERT_NUM_SAMPLES = 10000            # private samples per expert
DEFAULT_EXPERT_PROXY_DATA_NUM_SAMPLES = 500   # proxy samples mixed into expert training
DEFAULT_PROXY_NUM_SAMPLES = 5000              # proxy-data candidate-pool size


def get_expert_dir(model_basename, dataset, dpp,
                   num_samples=DEFAULT_EXPERT_NUM_SAMPLES,
                   proxy_data_num_samples=DEFAULT_EXPERT_PROXY_DATA_NUM_SAMPLES):
    """Expert checkpoint dir produced by nlp/scripts/run_lora_train_ours.sh.

    Mirrors the (timestamp-free) job_id scheme:
        MetaMoE_{model_basename}_{dataset}_{num_samples}_ProxyDataNum{proxy_data_num_samples}_{DPP|NODPP}
    so experts can be located by config alone, without hard-coded paths.
    Use proxy_data_num_samples=0 for experts trained on private data only.
    """
    dpp_tag = "DPP" if dpp else "NODPP"
    return f"MetaMoE_{model_basename}_{dataset}_{num_samples}_ProxyDataNum{proxy_data_num_samples}_{dpp_tag}"


def get_proxy_data_dir(model_basename, dataset, num_samples=DEFAULT_PROXY_NUM_SAMPLES):
    """Proxy-data dir produced by nlp/scripts/run_create_proxy_data.sh."""
    return f"ProxyData_{model_basename}_{dataset}_{num_samples}"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tuning LLMs on training data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument(
        '--model_path', type=str, default='outputs/llama-2-7b-hf-adamole-the8-commonsense-qa',
        help='huggingface model id or local model path')
    parser.add_argument(
        '--data_path', type=str, default='tau/commonsense_qa',
        help='huggingface model id or local model path')
    parser.add_argument(
        '--max_new_tokens', type=int, default=16,
        help='maximum number of new tokens')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='batch size in the pipeline')
    parser.add_argument(
        '--logits', default=False, action='store_true',
        help='checking choice logits instead of generated texts')
    parser.add_argument('--job_id', type=str,
                        help='job_id for saving results')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='number of samples to evaluate')
    parser.add_argument(
        '--max_steps', type=int, default=-1,
        help='maximum number of steps to train')
    parser.add_argument(
        '--proxy_data_num_samples', type=int, default=0,
        help='number of samples to use from proxy data')
    parser.add_argument(
        '--expert_num_samples', type=int, default=DEFAULT_EXPERT_NUM_SAMPLES,
        help='private-sample count the experts were trained on (used to locate expert checkpoints)')
    parser.add_argument(
        '--expert_proxy_data_num_samples', type=int, default=DEFAULT_EXPERT_PROXY_DATA_NUM_SAMPLES,
        help='proxy-sample count mixed into expert training (used to locate expert checkpoints)')
    parser.add_argument(
        '--max_length', type=int, default=256,
        help='maximum number of tokens')
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='gradient accumulation steps')
    parser.add_argument(
        '--num_train_epochs', type=int, default=0,
        help='number of training epochs')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='learning rate for training')
    parser.add_argument(
        '--lr_scheduler_type', type=str, default="constant_with_warmup",
        help='learning rate scheduler type')
    parser.add_argument(
        '--warmup_steps', type=int, default=0,
        help='number of warmup steps for training')
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='weight decay')
    parser.add_argument(
        '--aux_loss_coeff', type=float, default="0.0",
        help='auxiliary loss coefficient for moe')
    parser.add_argument(
        '--lambda_', type=float, default=0.5,
        help='lambda for gatting')
    parser.add_argument(
        '--which_experts', type=str, default="proxy", choices=["proxy", "gt"],
        help='which experts to use')
    parser.add_argument(
        '--which_data', type=str, default="proxy", choices=["proxy", "gt", "random"],
        help='which proxy data to use')
    parser.add_argument(
        '--DPP', default=False, action='store_true',
        help='use DPP when creating the proxy data')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='seed for reproducibility')
    parser.add_argument(
        '--wo_router_init', default=False, action='store_true',
        help='not initialize the router from the trained experts')

    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Add arguments
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = args.model_path
    model_name = os.path.basename(model_path).lower()
    model_basename = os.path.basename(model_path)  # original case, used to build checkpoint dir names

    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    max_length = args.max_length
    output_dir = os.path.join('outputs_train', args.job_id)
    is_chat = True if "instruct" in model_name or "chat" in model_name else False

    if args.which_experts == "proxy":
        expert_path_list = [
            get_expert_dir(model_basename, ds, args.DPP,
                           args.expert_num_samples, args.expert_proxy_data_num_samples)
            for ds in CLIENT_DATASETS
        ]
    elif args.which_experts == "gt":
        # experts trained on private data only (no proxy)
        expert_path_list = [
            get_expert_dir(model_basename, ds, dpp=False,
                           num_samples=args.expert_num_samples, proxy_data_num_samples=0)
            for ds in CLIENT_DATASETS
        ]
    else:
        raise ValueError(f"Invalid which_experts: {args.which_experts}")

    expert_path_list = [os.path.join("outputs_train", _) for _ in expert_path_list]
    print(f"### expert_path_list (which_experts={args.which_experts}, DPP={args.DPP}): {expert_path_list}")
    
    
    # lora the mole router embedding
    start_time = time.time()
    print("loading the mole router embedding")
    expert_router_embedding_list = []
    for expert_id, model_path in enumerate(expert_path_list):
        router_embedding_path = os.path.join(model_path, "router_embedding.pt")
        router_embedding = torch.load(router_embedding_path)
        expert_router_embedding_list.append(router_embedding)
    
    # Load trained experts
    base_model, tokenizer = None, None
    experts_state_dicts = {}
    router_inialized_state_dict = {}
    adapter_name = "default"
    for expert_id, model_path in enumerate(expert_path_list):
    
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir=DEFAULT_CACHE_DIR,
            token=ACCESS_TOKEN,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                peft_config.base_model_name_or_path,
                padding_side="left",
                cache_dir=DEFAULT_CACHE_DIR,
                token=ACCESS_TOKEN,
            )
            tokenizer.pad_token = tokenizer.eos_token
    
        model = PeftModelForCausalLM.from_pretrained(model=base_model, model_id=model_path)
            
        for layer_id, layer in enumerate(model.base_model.model.model.layers):
            experts_state_dicts[f"model.layers.{layer_id}.mlp.moe_layer.{adapter_name}.experts.{expert_id}.up_lora_A.default.weight"] = layer.mlp.up_proj.lora_A[adapter_name].weight.data
            experts_state_dicts[f"model.layers.{layer_id}.mlp.moe_layer.{adapter_name}.experts.{expert_id}.up_lora_B.default.weight"] = layer.mlp.up_proj.lora_B[adapter_name].weight.data

            experts_state_dicts[f"model.layers.{layer_id}.mlp.moe_layer.{adapter_name}.experts.{expert_id}.down_lora_A.default.weight"] = layer.mlp.down_proj.lora_A[adapter_name].weight.data
            experts_state_dicts[f"model.layers.{layer_id}.mlp.moe_layer.{adapter_name}.experts.{expert_id}.down_lora_B.default.weight"] = layer.mlp.down_proj.lora_B[adapter_name].weight.data
            
            experts_state_dicts[f"model.layers.{layer_id}.mlp.moe_layer.{adapter_name}.experts.{expert_id}.gate_lora_A.default.weight"] = layer.mlp.gate_proj.lora_A[adapter_name].weight.data
            experts_state_dicts[f"model.layers.{layer_id}.mlp.moe_layer.{adapter_name}.experts.{expert_id}.gate_lora_B.default.weight"] = layer.mlp.gate_proj.lora_B[adapter_name].weight.data

            if expert_id == 0:
                if f"base_model.model.model.layers.{layer_id}.mlp" in expert_router_embedding_list[0]:
                    router_inialized_state_dict[f"model.layers.{layer_id}.mlp.lora_gating.{adapter_name}.weight"] = torch.stack([expert_router_embedding_list[_][f"base_model.model.model.layers.{layer_id}.mlp"] for _ in range(len(expert_router_embedding_list))])
                else:
                    router_inialized_state_dict[f"model.layers.{layer_id}.mlp.lora_gating.{adapter_name}.weight"] = torch.stack([expert_router_embedding_list[_][f"base_model.model.model.layers.{layer_id}.mlp.up_proj"] for _ in range(len(expert_router_embedding_list))])
                
    torch.cuda.empty_cache()

    # create a mole model
    print("Creating a mole model")
    peft_config = MLPMoleConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["mlp"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            num_experts=3,
            top_k=1, 
        ) 
    
    if "llama" in model_name:
        class_name = MyCustomLlamaForCausalLM
    elif "qwen" in model_name:
        class_name = MyCustomQwen2ForCausalLM
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    base_model = class_name.from_pretrained(
            args.model_path, 
            cache_dir=DEFAULT_CACHE_DIR,
            token=ACCESS_TOKEN,
            device_map="auto",
            # torch_dtype=torch.bfloat16,
        )
    print(f"### base model: {base_model}")

    def replace_module(module, target_name):
        for child_name, child_module in module.named_children():
            if child_name.endswith(target_name):
                new_module = MLPMoLELayer(
                    base_layer=child_module,
                    adapter_name="default",
                    lora_rank=peft_config.lora_rank,
                    lora_alpha=peft_config.lora_alpha,
                    lora_dropout=peft_config.lora_dropout,
                    num_experts=peft_config.num_experts,
                    top_k=peft_config.top_k,
                    threshold=peft_config.threshold,
                    init_lora_weights=True,
                    lambda_=args.lambda_ 
                )
                new_module.to(child_module.gate_proj.weight.device)
                setattr(module, child_name, new_module)
            else:
                replace_module(child_module, target_name)
    
    replace_module(base_model, "mlp")
    print(f"base_model: {base_model}")
    # raise ValueError("stop here")

    for name, module in base_model.named_modules():
        print(f"### module name: {name}")

    print("base model after replacing MLP with MLPMoLELayer: ", base_model)

    for name, param in base_model.named_parameters():
        if name in experts_state_dicts.keys():
            print(f"### loading expert state dict: {name}")
            param.data = experts_state_dicts[name].to(param.device)
        if name in router_inialized_state_dict.keys() and not args.wo_router_init:
            print(f"### loading router state dict: {name}")
            param.data = router_inialized_state_dict[name].to(param.device)

    for name, param in base_model.named_parameters():
        if "lora_gating" in name or "lora_A" in name or "lora_B" in name or "lambda_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # router_param_count = sum(
    #     param.numel()
    #     for name, param in base_model.named_parameters()
    #     if "lora_gating" in name
    # )
    # print(f"### number of parameters in the MoE router: {router_param_count}")
    # raise Exception("stop here")
    
    # Count trainable parameters
    print_fraction_trainable_parameters(base_model)

    

    # train the router
    ## 1. load the proxy data
    ## 2. train the router embedding
    if args.num_train_epochs > 0:
        print(f">>> training the router embeddings")
        proxy_data_num_samples = args.proxy_data_num_samples
        if is_chat:
            tokenize_text = lambda examples: tokenizer.apply_chat_template(
                examples["text"],
                add_generation_prompt=True,
                continue_final_message=False,
                padding=True,
                # return_tensors="pt",
                return_dict=True,
                max_length=max_length,
            )
        else:
            tokenize_text = lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                # padding=True,
                # return_tensors="pt",
            )
        
        if args.which_data == "proxy":
            proxy_data_datasets_list = []
            proxy_data_path_list = [get_proxy_data_dir(model_basename, ds) for ds in CLIENT_DATASETS]
            for proxy_data_path in proxy_data_path_list:
                if args.DPP:
                    proxy_data_path = f"{HOME_PATH}/outputs_train/{proxy_data_path}/DPP_public_with_predictions"
                else:
                    proxy_data_path = f"{HOME_PATH}/outputs_train/{proxy_data_path}/public_with_predictions"
                proxy_data_datasets = get_formatted_datasets(proxy_data_path, False, is_chat)
                tokenized_proxy_data_datasets = proxy_data_datasets.map(
                    tokenize_text,
                    batched=True,
                    remove_columns=proxy_data_datasets["train"].column_names,
                )
                tokenized_proxy_data_datasets['train'] = tokenized_proxy_data_datasets['train'].take(proxy_data_num_samples)

                proxy_data_datasets_list.append(tokenized_proxy_data_datasets["train"])
        elif args.which_data == "random":
            proxy_data_datasets_list = []
            proxy_data_path_list = [get_proxy_data_dir(model_basename, ds) for ds in CLIENT_DATASETS]
            for proxy_data_path in proxy_data_path_list:
                if args.DPP:
                    proxy_data_path = f"{HOME_PATH}/outputs_train/{proxy_data_path}/DPP_public_with_predictions"
                else:
                    proxy_data_path = f"{HOME_PATH}/outputs_train/{proxy_data_path}/public_with_predictions"
                proxy_data_datasets = get_formatted_datasets(proxy_data_path, False, is_chat)
                tokenized_proxy_data_datasets = proxy_data_datasets.map(
                    tokenize_text,
                    batched=True,
                    remove_columns=proxy_data_datasets["train"].column_names,
                )
                tokenized_proxy_data_datasets['train'] = tokenized_proxy_data_datasets['train'].shuffle(seed=42).take(proxy_data_num_samples)

                proxy_data_datasets_list.append(tokenized_proxy_data_datasets["train"])
        elif args.which_data == "gt":
            proxy_data_path_list = [
                    "allenai/cosmos_qa",  
                    "allenai/social_i_qa", 
                    "tau/commonsense_qa", 
                    ]
            proxy_data_datasets_list = []
            for proxy_data_path in proxy_data_path_list:
                proxy_data_datasets = get_formatted_datasets(proxy_data_path, False, is_chat)
                tokenized_proxy_data_datasets = proxy_data_datasets.map(
                    tokenize_text,
                    batched=True,
                    remove_columns=proxy_data_datasets["train"].column_names,
                )
                tokenized_proxy_data_datasets['train'] = tokenized_proxy_data_datasets['train'].take(proxy_data_num_samples)

                proxy_data_datasets_list.append(tokenized_proxy_data_datasets["train"])
        else:
            raise ValueError(f"Invalid proxy type: {args.which_data}")

        proxy_data_datasets = SimpleUnionDataset(proxy_data_datasets_list)
        print(f'len of tokenized proxy data datasets: {len(proxy_data_datasets)}')
        data_collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt")
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in base_model.named_parameters() if any(nd in n for nd in ['lora_gating', 'lora_A', 'lora_B'])],
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate,
            },
            {
                'params': [p for n, p in base_model.named_parameters() if "lambda_" in n],
                'weight_decay': args.weight_decay,
                'lr': 0.001,
            }  
        ]

        from transformers import AdamW, get_constant_schedule_with_warmup
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

        # Set the trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            group_by_length=True,
            remove_unused_columns=False,
            logging_strategy="steps",
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=5000,
            save_strategy="no",
            # optim="adamw_torch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=False,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            # learning_rate=args.learning_rate,
            # lr_scheduler_type=args.lr_scheduler_type,
            # warmup_steps=args.warmup_steps,
            # weight_decay=args.weight_decay,
            # fp16=True,
            seed=args.seed,
            data_seed=args.seed,
        )
        trainer = Trainer(
            model=base_model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=proxy_data_datasets,
            eval_dataset=proxy_data_datasets,
            optimizers=(optimizer, lr_scheduler)
            # aux_loss_coeff=args.aux_loss_coeff,
        )

        # Train the model
        # base_model.print_trainable_parameters()
        # for name, param in base_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"trainable parameter: {name}: {param}")
        #         if "lambda_" in name:
        #             print(f"Lambda parameter value: {param.data.item():.4f}")
        # base_model.config.use_cache = False
        # torch.cuda.empty_cache()
        trainer.train()
        # base_model.config.use_cache = True

        # for name, param in base_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"trainable parameter: {name}: {param}")
        #         if "lambda_" in name:
        #             print(f"Lambda parameter value: {param.data.item():.4f}")
    
    unify_time = time.time() - start_time
    eval_start_time = time.time()
    evaluate_model(base_model, tokenizer, device, max_new_tokens, min(16, batch_size*2), max_length, args, model_name)
    eval_end_time = time.time()
    elapsed_time = eval_end_time - eval_start_time
    
    # Calculate total number of samples
    total_samples = 6160
    
    inference_speed = total_samples / elapsed_time if elapsed_time > 0 else 0
    print(f"### [Ours] Model: {model_name}, Unifying time: {unify_time:.2f} seconds ({unify_time/60:.2f} minutes), Inference speed: {inference_speed:.2f} samples/second, inference time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


