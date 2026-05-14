"""
Save training datasets to numpy memmap format
"""
import os
import numpy as np
from transformers import AutoTokenizer
from data import get_formatted_datasets
from src.utils.constants import DEFAULT_CACHE_DIR, ACCESS_TOKEN

# Dataset paths
DATASET_PATHS = [
    'tau/commonsense_qa',
    'allenai/social_i_qa',
    'allenai/cosmos_qa'
]

def save_datasets_to_memmap(dataset_paths, output_dir='saved_datasets', model_name='meta-llama/Llama-2-7b-hf', max_length=512):
    """
    Load training sets from multiple datasets, tokenize them, and save as numpy memmap format
    
    Args:
        dataset_paths: List of dataset paths
        output_dir: Directory to save the numpy arrays
        model_name: Model name for tokenizer
        max_length: Maximum sequence length for tokenization
    """
    
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=DEFAULT_CACHE_DIR,
        token=ACCESS_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Process each dataset
    target_output_dir = os.path.join(output_dir, f"{len(dataset_paths)}")
    os.makedirs(target_output_dir, exist_ok=True)
    for idx, data_path in enumerate(dataset_paths):
        all_tokenized_texts = []
        all_answers = []
        all_data_names = []
        all_num_choices = []
        print(f"\nProcessing dataset: {data_path}")
        data_name = os.path.basename(data_path).lower()

        try:
            # Load and format dataset
            formatted_datasets = get_formatted_datasets(data_path=data_path, prompt_only=False, is_chat=False)
            
            # Check if train split exists
            if "train" not in formatted_datasets:
                print(f"Warning: No 'train' split found in {data_path}. Available splits: {list(formatted_datasets.keys())}")
                continue
                
            train_dataset = formatted_datasets["train"]
            print(f"Loaded {len(train_dataset)} training samples from {data_path}")
            
            # Tokenize and collect data
            for i, example in enumerate(train_dataset):
                text = example['text']
                
                # Handle chat format (list of dicts) vs regular format (string)
                if isinstance(text, list):
                    # For chat format, convert to string using tokenizer's chat template
                    text_str = tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=False)
                else:
                    text_str = text
                
                # Tokenize text
                tokenized = tokenizer(
                    text_str,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='np'
                )
                
                all_tokenized_texts.append(tokenized['input_ids'][0])
                
                # Save metadata
                all_answers.append(example.get('answer', ''))
                all_data_names.append(example.get('data_name', data_name))
                all_num_choices.append(example.get('num_choices', 0))
                
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(train_dataset)} samples")
        except Exception as e:
            print(f"Error processing {data_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
        # Check if we have any data
        if len(all_tokenized_texts) == 0:
            print("Error: No training data was collected from any dataset!")
            return None, None
        
        # Convert to numpy arrays
        print(f"\nConverting to numpy arrays...")
        print(f"Total samples collected: {len(all_tokenized_texts)}")
        tokenized_array = np.array(all_tokenized_texts, dtype=np.int32)
        
        # Save tokenized texts as memmap
        tokenized_file = os.path.join(target_output_dir, f'train_{idx}.bin')
        tokenized_shape = tokenized_array.shape
        tokenized_dtype = tokenized_array.dtype
        
        # Create memmap file
        fp = np.memmap(tokenized_file, dtype=tokenized_dtype, mode='w+', shape=tokenized_shape)
        fp[:] = tokenized_array[:]
        fp.flush()
        del fp
        
        print(f"Saved tokenized texts to {tokenized_file} with shape {tokenized_shape}")
        
        # Save metadata
        metadata = {
            'answers': all_answers,
            'data_names': all_data_names,
            'num_choices': np.array(all_num_choices, dtype=np.int32),
            'shape': tokenized_shape,
            'dtype': str(tokenized_dtype),
            'max_length': max_length,
            'num_samples': len(all_tokenized_texts)
        }
        
        # Save metadata as numpy file (can be loaded separately)
        metadata_file = os.path.join(target_output_dir, f'metadata_{idx}.npz')
        np.savez(
            metadata_file,
            answers=np.array(all_answers, dtype=object),
            data_names=np.array(all_data_names, dtype=object),
            num_choices=np.array(all_num_choices, dtype=np.int32),
            shape=tokenized_shape,
            max_length=max_length,
            num_samples=len(all_tokenized_texts)
        )
        print(f"Saved metadata to {metadata_file}")
        
        # Save info file for easy loading
        info_file = os.path.join(target_output_dir, f'info_{idx}.txt')
        with open(info_file, 'w') as f:
            f.write(f"Tokenized texts file: {tokenized_file}\n")
            f.write(f"Shape: {tokenized_shape}\n")
            f.write(f"Dtype: {tokenized_dtype}\n")
            f.write(f"Max length: {max_length}\n")
            f.write(f"Number of samples: {len(all_tokenized_texts)}\n")
            f.write(f"Metadata file: {metadata_file}\n")
        
        print(f"\nSaved all data to {target_output_dir}")
        print(f"To load tokenized texts, use:")
        print(f"  tokenized = np.memmap('{tokenized_file}', dtype={tokenized_dtype}, mode='r', shape={tokenized_shape})")
        print(f"To load metadata, use:")
        print(f"  metadata = np.load('{metadata_file}', allow_pickle=True)")
    
    return tokenized_file, metadata_file

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/research/d2/rshr/wsjiang/projects/CoMiGS/data', help='Output directory')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B', help='Model name for tokenizer')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    args = parser.parse_args() 
    
    save_datasets_to_memmap(
        dataset_paths=DATASET_PATHS,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length 
    )
