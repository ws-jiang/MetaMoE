"""
create proxy data (from public data) for each client based on its local data.

public data: stanford_alpaca: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
"""
from dataclasses import dataclass

import json
import time
import evaluate
import torch
import numpy as np
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorForLanguageModeling

from src.utils.constants import DEFAULT_CACHE_DIR, ACCESS_TOKEN
import os, argparse
from typing import List, Sequence, Dict

from torch.utils.data import Dataset, Subset

from datasets import load_dataset, concatenate_datasets, DatasetDict
from datasets.arrow_dataset import Dataset as ds_Dataset

from src.utils.dpp_map import fast_map_dpp


def format_text(example, data_name: str, prompt_only: bool = True, is_chat: bool = False):
    """
    Format an example into one text
    """
    if data_name == 'commonsense_qa':
        """
        The sanctions against the school were a punishing blow, and they seemed to what the efforts the 
        school had made to change?
        Choices:
        A. ignore
        B. enforce
        C. authoritarian
        D. yell at
        E. avoid
        Answer: A
        """
        question = example['question']
        text = f"Question: {example['question']}\nChoices:\n"
        choices = example['choices']
        for label, choice in zip(choices['label'], choices['text']):
            text += f"{label}. {choice}\n"
        # text += "Answer: "
        example['answer'] = example['answerKey']
        example['num_choices'] = 5

    elif data_name == 'cosmos_qa':
        """
        Context: Good Old War and person L : I saw both of these bands Wednesday night , and they both blew me away . 
        seriously . Good Old War is acoustic and makes me smile . I really can not help but be happy when I listen to 
        them ; I think it 's the fact that they seemed so happy themselves when they played .
        Question: In the future , will this person go to see other bands play ?
        Choices:
        A. None of the above choices .
        B. This person likes music and likes to see the show , they will see other bands play .
        C. This person only likes Good Old War and Person L , no other bands .
        D. Other Bands is not on tour and this person can not see them .
        Answer: B
        """
        question = f"{example['context']} {example['question']}"
        text = f"Context: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
        text += f"A. {example['answer0']}\n"
        text += f"B. {example['answer1']}\n"
        text += f"C. {example['answer2']}\n"
        text += f"D. {example['answer3']}\n"
        # text += "Answer: "
        example['answer'] = chr(ord('A') + example['label'])
        example['num_choices'] = 4

    elif data_name == 'social_i_qa':
        """
        Context: Cameron decided to have a barbecue and gathered her friends together.
        Question: How would Others feel as a result?
        Choices:
        A. like attending
        B. like staying home
        C. a good friend to have
        Answer: A
        """
        question = f"{example['context']} {example['question']}"
        text = f"Context: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
        text += f"A. {example['answerA']}\n"
        text += f"B. {example['answerB']}\n"
        text += f"C. {example['answerC']}\n"
        # text += "Answer: "
        example['answer'] = chr(ord('A') + int(example['label']) - 1)
        example['num_choices'] = 3
    elif data_name == 'alpaca':
        """
        Question: What is the capital of France?
        Answer: Paris
        """
        question = f"{example['instruction']} {example['input']}"
    else:
        raise ValueError(f"Data name {data_name} not supported")

    new_example = {}    
    if is_chat:
        new_example['question'] = [{"role": "user", "content": question.strip()}]
    else:
        new_example['question'] = question.strip()
    new_example['ds_name'] = data_name
    return new_example


def get_formatted_datasets(data_path: str, prompt_only: bool, num_samples: int = -1, is_chat: bool = False):
    """
    Get formatted datasets
    """
    data_name = os.path.basename(data_path).lower()
    datasets = load_dataset(path=data_path, trust_remote_code=True)
    print(f"Loaded datasets: {datasets}")
    # Format datasets
    formatted_datasets = datasets.map(
        lambda example: format_text(example, data_name, prompt_only=prompt_only, is_chat=is_chat),
        batched=False, load_from_cache_file=False)
    print(f'Formatted datasets: {formatted_datasets}')
    print(f"Formatted example: {formatted_datasets['train'][0]}")
    print(f"question example:\n{formatted_datasets['train']['question'][0]}")

    return formatted_datasets

class UnionDataset(Dataset):
    def __init__(self, datasets: List[ds_Dataset]):
        self.datasets = datasets
        self.num_labels = len(datasets)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)
    
    def __getitem__(self, idx):
        label = 0 
        for ds in self.datasets: 
            if idx < len(ds):
                return ds[idx] | {'label': label}
            idx -= len(ds)
            label += 1
        raise IndexError(f"Index {idx} is out of bounds for the combined dataset")  
            

@dataclass
class DataCollatorForBinaryClassification(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.Tensor(instance['input_ids']).long() for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = [instance['label'] for instance in instances]
        labels = torch.tensor(labels).long()

        return dict(
            input_ids=input_ids,
            labels=labels,
        )

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='create proxy data (from public data) for each client based on its local data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--public_data_path', type=str, default='tatsu-lab/alpaca',
        help='public data path')
    parser.add_argument(
        '--local_data_path', type=str, default='tau/commonsense_qa',
        help='local data path')
    parser.add_argument(
        '--num_samples', type=int, default=5000,
        help='number of samples to take from the dataset')
    parser.add_argument(
        '--max_length', type=int, default=256,
        help='maximum number of tokens')
    parser.add_argument(
        '--model_path', type=str, default='meta-llama/Llama-3.2-1B',
        help='huggingface model id or local model path')
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
        '--learning_rate', type=float, default=1e-5,
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
    parser.add_argument('--job_id', type=str, default='alpaca_commonsense_qa',
                        help='job_id for saving results')
    parser.add_argument(
        '--DPP', default=False, action='store_true',
        help='use DPP when creating the proxy data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.model_path
    max_length = args.max_length
    output_dir = os.path.join('outputs_train', args.job_id)
    model_name = os.path.basename(model_path).lower()
    is_chat = True if "instruct" in model_name or "chat" in model_name else False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        # add_bos_token=True,
        add_eos_token=True,
        cache_dir=DEFAULT_CACHE_DIR,
        token=ACCESS_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if is_chat:
        tokenize_text = lambda examples:tokenizer.apply_chat_template(
            examples["question"],
            add_generation_prompt=False,
            continue_final_message=True,
            padding=True,
            return_dict=True,
        )
    else:
        tokenize_text = lambda examples: tokenizer(
            examples["question"],
            truncation=True,
            max_length=max_length,
        )

    local_ds = get_formatted_datasets(args.local_data_path, True, num_samples=args.num_samples, is_chat=is_chat)
    tokenized_local_ds = local_ds.map(
        tokenize_text,
        batched=True,
        remove_columns=local_ds['train'].column_names,
    )
    tokenized_local_ds['train'] = tokenized_local_ds['train'].take(args.num_samples)

    print(tokenized_local_ds)

    public_ds = get_formatted_datasets(args.public_data_path, True, num_samples=args.num_samples, is_chat=is_chat) 
    tokenized_public_ds = public_ds.map(
        tokenize_text,
        batched=True,
        remove_columns=public_ds['train'].column_names,
    )
    # Split tokenized_alpaca_ds into train and validation
    tokenized_public_split = tokenized_public_ds['train'].train_test_split(test_size=0.9, seed=42, shuffle=True)
    tokenized_public_split['train'] = tokenized_public_split['train'].take(args.num_samples)
    
    union_ds_train = UnionDataset([tokenized_public_split['train'], tokenized_local_ds['train']])
    union_ds_val = UnionDataset([tokenized_public_split['test'], tokenized_local_ds['validation']])

    print(f"Union dataset train length: {len(union_ds_train)}")
    print(f"Union dataset test length: {len(union_ds_val)}")

    data_collator = DataCollatorForBinaryClassification(tokenizer=tokenizer)

    # Set the trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        group_by_length=True,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="no",
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
        # fp16=True,
        seed=0,
        data_seed=0,
        report_to=["tensorboard"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        cache_dir=DEFAULT_CACHE_DIR,
        token=ACCESS_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=union_ds_train,
        eval_dataset=union_ds_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate public_ds_train and save predictions
    model.eval()

    start_time = time.time()
    
    # Tokenize public_ds_train for evaluation
    public_eval_tokenized = public_ds['train'].map(
        tokenize_text,
        batched=True,
        remove_columns=public_ds['train'].column_names,
    )
    
    public_eval_tokenized = UnionDataset([public_eval_tokenized])
    # Get predictions from the trained model
    predictions = trainer.predict(public_eval_tokenized)
    softmax_predictions = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)
    predicted_labels = np.argmax(softmax_predictions, axis=1)
    print(f"accuracy: {100*(1-sum(predicted_labels)/len(predicted_labels)):.2f}%")

    # raise Exception("Stop here")

    
    if args.DPP:
        g_scores = softmax_predictions[:,1].tolist()

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

        data_collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt")

        public_eval_tokenized = public_ds['train'].map(
            tokenize_text,
            batched=True,
            remove_columns=public_ds['train'].column_names,
        )
        top_sample_subset = Subset(public_eval_tokenized, top_2000_indices)
        top_sample_loader = torch.utils.data.DataLoader(top_sample_subset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=False, num_workers=16)

        all_features = []
        # model.to(device)
        with torch.no_grad():
            for inputs in tqdm(top_sample_loader, desc="Evaluating on public_train_dataset"):
                # inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(input_ids=inputs['input_ids'], output_hidden_states=True, return_dict=True)
                last_hidden_state = outputs.hidden_states[-1].mean(dim=1)
                all_features.append(last_hidden_state)
        
        # Concatenate all features into a single tensor
        all_features = torch.cat(all_features, dim=0)
        # Normalize features to unit vectors
        all_features = torch.nn.functional.normalize(all_features, p=2, dim=1)
        # Compute cosine similarity matrix
        similarity_matrix = all_features @ all_features.T

        # Convert similarity_matrix to numpy array
        similarity_matrix_np = similarity_matrix.float().cpu().numpy()
        kernel_matrix = similarity_matrix_np * outer_product_matrix

        seleted_items = fast_map_dpp(kernel_matrix, 500)
        seleted_items_indices = np.array(top_2000_indices)[seleted_items]
        dpp_score_list = np.zeros(len(public_eval_tokenized))
        for i, idx in enumerate(seleted_items_indices):
            dpp_score_list[idx] = 1.0
        
        public_ds['train'] = public_ds['train'].add_column("predicted_label", predicted_labels.tolist())
        public_ds['train'] = public_ds['train'].add_column("prediction_score", dpp_score_list.tolist())
        public_ds['train'] = public_ds['train'].add_column("org_prediction_score", softmax_predictions[:,1].tolist())
    else:
        public_ds['train'] = public_ds['train'].add_column("predicted_label", predicted_labels.tolist())
        public_ds['train'] = public_ds['train'].add_column("prediction_score", softmax_predictions[:,1].tolist())
        public_ds['train'] = public_ds['train'].add_column("org_prediction_score", softmax_predictions[:,1].tolist())
    
    elapsed_time = time.time() - start_time
    print(f"Time taken for DPP/score processing step: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # raise Exception("Stop here")

    print(f"Added predictions to public_ds_train. Sample with prediction:")
    print(f"Question: {public_ds['train'][0]['question']}")
    print(f"Predicted label: {public_ds['train'][0]['predicted_label']}")
    print(f"Prediction score: {public_ds['train'][0]['prediction_score']:.4f}")
    print(f"Org prediction score: {public_ds['train'][0]['org_prediction_score']:.4f}")
    
    
    # Save the dataset with predictions
    target_file_name = f"DPP_public_with_predictions" if args.DPP else "public_with_predictions"
    score_file_name = f"DPP_score_list.json" if args.DPP else "score_list.json"
    
    public_ds.save_to_disk(os.path.join(output_dir, target_file_name))

    score_list = public_ds['train'].sort('prediction_score', reverse=True)['prediction_score']

    with open(os.path.join(output_dir, score_file_name), 'w') as f:
        json.dump(score_list, f)

    print(f"Saved proxy data to {os.path.join(output_dir, target_file_name)}")
    print(f"Saved score list to {os.path.join(output_dir, score_file_name)}")