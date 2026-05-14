import clip
import os
import time
from torch import nn
import argparse
import torch
import json
from clip.model import VisionTransformer
from torch.optim import SGD
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Subset
from torch.cuda.amp import autocast

from mm_utils import refine_classname, MyCLIP, validate
from src.utils.model_utils import set_direct_forward_base_layer, print_fraction_trainable_parameters
from src.multi_modal.datasets.datasets_utils import get_ds
from src.utils.constants import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, HOME_PATH
from peft import PeftModel 
from src.mlp_mole import CLIP_MLP_MoLELayer
from src.multi_modal.datasets.datasets_utils import get_imagenet_sample_indices

import random
random.seed(42)  # Set seed for reproducibility

# Default expert-training config used in the paper (see cv/scripts/run_train_lora.sh).
DEFAULT_EXPERT_NUM_EPOCHS = 10
DEFAULT_EXPERT_NUM_PROXY_SAMPLES = 500


def get_expert_dir(arch, dataset, dpp,
                   num_epochs=DEFAULT_EXPERT_NUM_EPOCHS,
                   num_proxy_samples=DEFAULT_EXPERT_NUM_PROXY_SAMPLES):
    """Expert checkpoint directory name produced by cv/scripts/run_train_lora.sh.

    Mirrors the (timestamp-free) job_id scheme:
        MM_MetaMoE_{arch}_{dataset}_{num_epochs}_Proxy{num_proxy_samples}_{DPP|NODPP}
    so experts can be located by config alone, without hard-coded paths.
    """
    dpp_tag = "DPP" if dpp else "NODPP"
    return f"MM_MetaMoE_{arch}_{dataset}_{num_epochs}_Proxy{num_proxy_samples}_{dpp_tag}"


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arch', type=str, default='vit_b32', choices=['vit_b16', 'vit_b32', 'vit_l14'])
    parser.add_argument('--lora_target_modules', type=str, default='c_fc,c_proj')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--dataset', type=str, default='pets,flower102,eurosat')
    parser.add_argument('--root_data_path', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=0)
    parser.add_argument('--job_id', type=str, default='JOBID')
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--num_proxy_samples', type=int, default=500)

    # Expert-training config; used to locate expert checkpoints saved by cv/scripts/run_train_lora.sh.
    parser.add_argument('--expert_num_epochs', type=int, default=DEFAULT_EXPERT_NUM_EPOCHS)
    parser.add_argument('--expert_num_proxy_samples', type=int, default=DEFAULT_EXPERT_NUM_PROXY_SAMPLES)

    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--lambda_', type=float, default=0.0)
    parser.add_argument('--which_data', type=str, default='proxy', choices=['proxy', 'random'])
    parser.add_argument('--init_router', default=True, action='store_false')
    parser.add_argument('--DPP', default=False, action='store_true')

    return parser.parse_args()

def load_clip_model(args, device):
    if args.arch == "vit_b32":
        model, preprocess = clip.load('ViT-B/32', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    elif args.arch == "vit_b16":
        model, preprocess = clip.load('ViT-B/16', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    elif args.arch == "vit_l14":
        model, preprocess = clip.load('ViT-L/14', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    else:
        raise ValueError(f"unknown model: {args.model}_{args.arch}")
    return model, preprocess

def get_leaf_modules_with_grad(model):
    module_list = []
    name_list = []
    for name, module in model.named_modules():
        if name.endswith("mlp"):
            module.name = name
            module_list += [module]
            name_list += [name]
    return module_list

class MyCustomVisionTransformer(nn.Module):
    def __init__(self, visual_encoder):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.conv1 = self.visual_encoder.conv1
        self.expert_id = -1

    def forward(self, x: torch.Tensor):
        images = x.clone()
        
        with torch.no_grad():
            set_direct_forward_base_layer(self.visual_encoder, direct_forward_base_layer=True)

            # images = images.to(device)
            embeeding_dict = {}

            hooks = []

            
            def track_embedding_hook(module, input, output):
                # print(input[0].shape)
                # print(input.shape)
                # raise ValueError("stop here")
                embeeding_dict[module.name] = input[0].mean(axis=(1)).detach()
                return output
            

            # Register hooks for all target modules
            leaf_modules_with_grad = get_leaf_modules_with_grad(self.visual_encoder)
            for layer in leaf_modules_with_grad:
                hook = layer.register_forward_hook(track_embedding_hook)
                hooks.append(hook)

            with autocast():
                with torch.no_grad():
                    self.visual_encoder(images)

            # Remove hooks
            for hook in hooks:
                hook.remove()
            hooks = []

            idx = 0
            for name, module in self.named_modules():
                if name.endswith('moe_layer'):
                    module['default'].sample_embeddings = embeeding_dict[f"transformer.resblocks.{idx}.mlp"]
                    module['default'].expert_id = self.expert_id
                    # print(idx,name)
                    idx += 1
            # raise ValueError("stop here")
            
            set_direct_forward_base_layer(self.visual_encoder, direct_forward_base_layer=False)
 
        outputs = self.visual_encoder.forward(x)
        return outputs



    # def forward(self, x: torch.Tensor):
    #     old_x = x.clone()
    #     with torch.no_grad():
    #         set_direct_forward_base_layer(self.visual_encoder, direct_forward_base_layer=True)
            
    #         x = self.visual_encoder.conv1(x)  # shape = [*, width, grid, grid]
    #         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    #         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    #         x = torch.cat([self.visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    #         x = x + self.visual_encoder.positional_embedding.to(x.dtype)
    #         x = self.visual_encoder.ln_pre(x)

    #         x = x.permute(1, 0, 2)  # NLD -> LND
    #         x = self.visual_encoder.transformer(x)
    #         x = x.permute(1, 0, 2)  # LND -> NLD

    #         x = self.visual_encoder.ln_post(x[:, 0, :])

    #         for name, module in self.named_modules():
    #             if name.endswith('moe_layer'):
    #                 module['default'].sample_embeddings = x
            
    #         set_direct_forward_base_layer(self.visual_encoder, direct_forward_base_layer=False)
 
    #     outputs = self.visual_encoder.forward(old_x)
    #     return outputs

def eval_mm(args, preprocess, my_clip_model, criterion, device, model):
    acc_list = []
    for dataset_name in args.dataset:
        train_dataset, val_dataset = get_ds(dataset_name, args, preprocess)
        print(f"### [{dataset_name}] number of validation samples: {len(val_dataset)}")
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=False)

        class_names = train_dataset.classes
        class_names = refine_classname(class_names)
        template = 'This is a photo of a {}'
        texts = [template.format(label) for label in class_names]

        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)

        if my_clip_model is None:
            my_clip_model = MyCLIP(model, text_features)
            my_clip_model = my_clip_model.to(device)
        else:
            my_clip_model.text_features = text_features
        my_clip_model.eval()

        avg_loss, accuracy = validate(val_loader, my_clip_model, criterion, args, epoch=-1, device=device)
        acc_list.append(accuracy)
        print(f"### {dataset_name} accuracy: {accuracy:.2f} and loss: {avg_loss:.4f}")
    
    results = acc_list + [sum(acc_list) / len(acc_list)]
    print("### average accuracy: ", ",".join([f"{acc:.2f}" for acc in results]))

if __name__ == "__main__":
    args = parse_args()
    assert args.arch in ["vit_b32", "vit_b16"]

    args.output_dir = os.path.join(f'{HOME_PATH}/outputs_train', f"{args.job_id}")

    args.dataset = args.dataset.split(',')
    args.num_experts = len(args.dataset)

    print(args)

    device = f"cuda" if torch.cuda.is_available() else "cpu"

    #########################################################
    #########################################################
    ## create the MoE model
    model, preprocess = load_clip_model(args, device)

    start_time = time.time()

    def replace_module(module, target_name):
        for child_name, child_module in module.named_children():
            if child_name.endswith(target_name):
                new_module = CLIP_MLP_MoLELayer(
                    base_layer=child_module,
                    adapter_name="default",
                    lora_rank=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    num_experts=args.num_experts,
                    top_k=args.top_k,
                    init_lora_weights=True,
                    lambda_=args.lambda_ 
                )
                setattr(module, child_name, new_module)
            else:
                replace_module(child_module, target_name)
    
    replace_module(model.visual, "mlp")

    print(model.visual)
    

    #########################################################
    #########################################################
    # load the experts and router embedding
    

    expert_path_list = [
        os.path.join(
            "outputs_train",
            get_expert_dir(args.arch, dataset, args.DPP,
                           args.expert_num_epochs, args.expert_num_proxy_samples),
        )
        for dataset in args.dataset
    ]
    
    print("loading the mole router embedding")
    expert_router_embedding_list = []
    for expert_id, model_path in enumerate(expert_path_list):
        router_embedding_path = os.path.join(model_path, "router_embedding.pt")
        router_embedding = torch.load(router_embedding_path)
        expert_router_embedding_list.append(router_embedding)

    experts_state_dicts = {}
    router_inialized_state_dict = {}
    adapter_name = "default"
    for expert_id, model_path in enumerate(expert_path_list):

        base_model, _ = load_clip_model(args, device)
    
        base_model.visual = PeftModel.from_pretrained(base_model.visual, os.path.join(HOME_PATH, model_path))
        base_model.to(device) 

        for layer_id, layer in enumerate(base_model.visual.base_model.model.transformer.resblocks):
            experts_state_dicts[f"visual_encoder.transformer.resblocks.{layer_id}.mlp.moe_layer.default.experts.{expert_id}.fc_lora_A.default.weight"] = layer.mlp.c_fc.lora_A[adapter_name].weight.data
            experts_state_dicts[f"visual_encoder.transformer.resblocks.{layer_id}.mlp.moe_layer.default.experts.{expert_id}.fc_lora_B.default.weight"] = layer.mlp.c_fc.lora_B[adapter_name].weight.data

            experts_state_dicts[f"visual_encoder.transformer.resblocks.{layer_id}.mlp.moe_layer.default.experts.{expert_id}.proj_lora_A.default.weight"] = layer.mlp.c_proj.lora_A[adapter_name].weight.data
            experts_state_dicts[f"visual_encoder.transformer.resblocks.{layer_id}.mlp.moe_layer.default.experts.{expert_id}.proj_lora_B.default.weight"] = layer.mlp.c_proj.lora_B[adapter_name].weight.data

            if expert_id == 0:
                router_inialized_state_dict[f"visual_encoder.transformer.resblocks.{layer_id}.mlp.lora_gating.default.weight"] = torch.stack([expert_router_embedding_list[_][f"visual.base_model.model.transformer.resblocks.{layer_id}.mlp"] for _ in range(len(expert_router_embedding_list))])
                
    torch.cuda.empty_cache()

    model.visual = MyCustomVisionTransformer(model.visual)

    for name, param in model.visual.named_parameters():
        if name in experts_state_dicts.keys():
            param.data = experts_state_dicts[name]
        if args.init_router and args.num_epochs > 0:
            if name in router_inialized_state_dict.keys():
                param.data = router_inialized_state_dict[name]
    
    # router_param_count = sum(
    #     param.numel()
    #     for name, param in model.visual.named_parameters()
    #     if "lora_gating" in name
    # )
    # print(f"### number of parameters in the MoE router: {router_param_count}")
    # raise Exception("stop here")
    
    
    print(model)
    criterion = nn.CrossEntropyLoss()

    my_clip_model = None

    #########################################################
    #########################################################
    ## train the router and ffn
    if args.num_epochs > 0:
        for name, param in model.named_parameters():
            if "lora_gating" in name or "lora_A" in name or "lora_B" in name or "lambda_" in name:
                param.requires_grad = True
                print(f"(model) {name}: {param.shape}, {param.data}")
            else:
                param.requires_grad = False
        
        print_fraction_trainable_parameters(model)

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['lora_gating', 'lora_A', 'lora_B'])],
                'lr': args.lr,
            },
            {
                'params': [p for n, p in model.named_parameters() if "lambda_" in n],
                'lr': 0.001,
            }  
        ]
        optimizer = SGD(optimizer_grouped_parameters, momentum=0.9, weight_decay=0.0001, nesterov=True)

        ## create the proxy training dataset
        if args.which_data == "random":
            imagenet_train_dataset, val_dataset = get_ds("imagenet", args, preprocess)
            proxy_train_dataset_list = []
            random_indices = random.sample(range(len(imagenet_train_dataset)), args.num_proxy_samples*len(args.dataset))
            proxy_subset = Subset(imagenet_train_dataset, random_indices)
            train_loader = DataLoader(proxy_subset,
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=True)
            
            class_names = imagenet_train_dataset.classes
        else:
            imagenet_train_dataset, val_dataset = get_ds("imagenet", args, preprocess)
            proxy_train_dataset_list = []
            for proxy_dataset in args.dataset: # , "dtd", "ucf"
                proxy_dataset_indices = get_imagenet_sample_indices(args, proxy_dataset)
                proxy_subset = Subset(imagenet_train_dataset, proxy_dataset_indices)
                proxy_train_dataset_list.append(proxy_subset)
            proxy_train_dataset = ConcatDataset(proxy_train_dataset_list)
            print(f"number of proxy samples: {len(proxy_train_dataset)}")
            
            train_loader = DataLoader(proxy_train_dataset,
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=False)
            
            class_names = imagenet_train_dataset.classes


        class_names = refine_classname(class_names)
        template = 'This is a photo of a {}'
        texts = [template.format(label) for label in class_names]

        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)

        my_clip_model = MyCLIP(model, text_features)
        my_clip_model = my_clip_model.to(device)
        
        for epoch in range(args.num_epochs):
            my_clip_model.train()
            running_loss = 0.0
            total = 0
            correct = 0 
            step = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                my_clip_model.zero_grad()

                outputs = my_clip_model(images)
                loss = criterion(outputs, labels)
                print("loss: ", loss.item())
                loss.backward()

                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if (step + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                step += 1

                model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
    
    unify_time = time.time() - start_time

    #########################################################
    #########################################################
    ## evaluate the model

    eval_start_time = time.time()
    eval_mm(args, preprocess, my_clip_model, criterion, device, model)
    eval_end_time = time.time()
    elapsed_time = eval_end_time - eval_start_time
    
    # Calculate total number of validation images
    total_val_images = 14232
    
    inference_speed = total_val_images / elapsed_time if elapsed_time > 0 else 0
    print(f"### [BTMoE] Model: {args.arch}, Unifying time: {unify_time:.2f} seconds ({unify_time/60:.2f} minutes), Inference speed: {inference_speed:.0f} images/second, inference time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
