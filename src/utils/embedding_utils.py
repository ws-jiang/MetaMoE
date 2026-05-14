import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.lora import LinearLoraLayer


def track_embedding(dataset, model, tokenizer, embedding_save_path, batch_size=16, is_chat=False):
    model.eval()

    def get_leaf_modules_with_grad(model):
        module_list = []
        name_list = []
        for name, module in model.named_modules():
            # if isinstance(module, LinearLoraLayer):
            if name.endswith('mlp.up_proj'):
                module.name = name
                module_list += [module]
                name_list += [name]
        # print(f"Found {len(module_list)} target modules")
        # print(f"the module list is {name_list}")
        return module_list

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Initialize storage for embeddings
    alignment_embedding = [{} for _ in range(len(dataloader))]
    index = 0

    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing embeddings")):
        hooks = []
        alignment_embedding_per_data = alignment_embedding[index]

        def track_embedding_hook(module, input, output):
            # print(f"input shape: {input[1].shape}")
            alignment_embedding_per_data[module.name] = input[0].mean(axis=(0,1)).detach()
            torch.cuda.empty_cache()
            return output

        # Register hooks for all target modules
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            hook = layer.register_forward_hook(track_embedding_hook)
            hooks.append(hook)

        # Tokenize inputs and move to GPU
        if is_chat:
            inputs = tokenizer.apply_chat_template(
                batch["text"],
                add_generation_prompt=True,
                continue_final_message=False,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = tokenizer(
                batch["text"],  # Assuming the dataset returns a "text" field
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to("cuda")

        with torch.no_grad():
            model(**inputs)  # Unpack the tokenizer outputs (input_ids, attention_mask, etc.)

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
    # save_path = os.path.join(embedding_save_path, "router_embedding.pt")
    torch.save(mean_embeddings, embedding_save_path)
    print(f"Saved mean embeddings to {embedding_save_path}")

def compute_sample_embedding(sample_batch, model, tokenizer):
    """
    Compute and return the embedding (last hidden state) for the given sample_batch.
    sample_batch: dict with a 'text' field (list of str)
    model: HuggingFace model
    tokenizer: HuggingFace tokenizer
    Returns: torch.Tensor of shape (batch_size, seq_len, hidden_dim)
    """
    model.eval()
    with torch.no_grad():
        # Tokenize the input batch
        inputs = tokenizer(
            sample_batch["text"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        # Forward pass with hidden states
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
        return last_hidden_state
    
    