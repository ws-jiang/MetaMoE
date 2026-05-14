import os

import clip
import torch
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Subset
from tqdm import tqdm

from mm_eval_mlp import eval_mm
from mm_utils import parse_args, refine_classname, MyCLIP
from src.utils.model_utils import print_fraction_trainable_parameters
from src.multi_modal.datasets.datasets_utils import SimpleUnionDatasetOffsetLabel
from src.multi_modal.datasets.datasets_utils import get_ds
from src.multi_modal.datasets.datasets_utils import get_imagenet_sample_indices
from src.utils.constants import DEFAULT_CACHE_DIR, HOME_PATH

if __name__ == "__main__":
    args = parse_args() 

    args.output_dir = os.path.join(f'{HOME_PATH}/outputs_train', f"{args.job_id}")

    print(args)

    device = f"cuda" if torch.cuda.is_available() else "cpu"

    template = 'This is a photo of a {}'

    if args.arch == "vit_b32":
        model, preprocess = clip.load('ViT-B/32', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    elif args.arch == "vit_b16":
        model, preprocess = clip.load('ViT-B/16', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    elif args.arch == "vit_l14":
        model, preprocess = clip.load('ViT-L/14', device, jit=False, download_root=DEFAULT_CACHE_DIR)
    else:
        raise ValueError(f"unknown model: {args.model}_{args.arch}")

    print(model)
    print(preprocess)

    # load proxy data
    if args.num_proxy_samples > 0 and args.num_epochs > 0 and args.proxy_all != "yes":
        imagenet_train_dataset, _ = get_ds("imagenet", args, preprocess)
        proxy_dataset_indices = get_imagenet_sample_indices(args, args.dataset)
        proxy_subset = Subset(imagenet_train_dataset, proxy_dataset_indices)
        proxy_subset.classes = imagenet_train_dataset.classes

        private_train_dataset, private_val_dataset = get_ds(args.dataset, args, preprocess)
        train_dataset = SimpleUnionDatasetOffsetLabel([private_train_dataset, proxy_subset])

        train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size, pin_memory=True,
                                    num_workers=args.num_workers, shuffle=True)

        val_loader = DataLoader(private_val_dataset,
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=False)
        
        class_names = private_train_dataset.classes + imagenet_train_dataset.classes
    elif args.proxy_all == "yes":
        imagenet_train_dataset, validation_dataset = get_ds("imagenet", args, preprocess)
        proxy_train_dataset_list = []
        for proxy_dataset in [ "pets", "eurosat", "flower102" ]: # ,
            proxy_dataset_indices = get_imagenet_sample_indices(args, proxy_dataset)
            proxy_subset = Subset(imagenet_train_dataset, proxy_dataset_indices)
            proxy_train_dataset_list.append(proxy_subset)
        proxy_train_dataset = ConcatDataset(proxy_train_dataset_list)
        print(f"number of proxy samples: {len(proxy_train_dataset)}")
        class_names = imagenet_train_dataset.classes
        train_loader = DataLoader(proxy_train_dataset,
                                    batch_size=args.batch_size, pin_memory=True,
                                    num_workers=args.num_workers, shuffle=True)
        val_loader = DataLoader(validation_dataset,
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=False)
    else:
        private_train_dataset, private_val_dataset = get_ds(args.dataset, args, preprocess)
        train_dataset = private_train_dataset
        train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size, pin_memory=True,
                                    num_workers=args.num_workers, shuffle=True)

        val_loader = DataLoader(private_val_dataset, 
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=False)
        
        class_names = private_train_dataset.classes

    class_names = refine_classname(class_names)
    texts = [template.format(label) for label in class_names]

    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)

    if args.use_lora:
        model.visual = PeftModel.from_pretrained(model.visual, args.output_dir)
    else:

        target_modules = args.lora_target_modules.split(',')
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )

        model.visual = get_peft_model(model.visual, lora_config)
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        print_fraction_trainable_parameters(model)

    my_clip_model = MyCLIP(model, text_features)
    my_clip_model = my_clip_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.num_epochs > 0:
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            my_clip_model = torch.nn.DataParallel(my_clip_model)

        # Training setup
        
        # optimizer = torch.optim.AdamW(my_clip_model.model.visual.parameters(),
        #                             lr=1e-5, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8)
        optimizer = torch.optim.SGD(my_clip_model.model.visual.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True) 

        for epoch in range(args.num_epochs):
            my_clip_model.train()
            running_loss = 0.0
            total = 0
            correct = 0
            step = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]"):
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

                # if (step + 1) % 100 == 0:
                #     print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}", flush=True)
                step += 1

                model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%", flush=True)


    # ## reset the text features
    # class_names = private_train_dataset.classes
    # class_names = refine_classname(class_names)
    # texts = [template.format(label) for label in class_names]

    # with torch.no_grad():
    #     text_tokens = clip.tokenize(texts).to(device)
    #     text_features = model.encode_text(text_tokens)
    #     text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)
    
    # my_clip_model.text_features = text_features
    # validate(val_loader, my_clip_model, criterion, args, epoch=-1, device=device)

    my_clip_model.model.visual.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")

    #########################################################
    #########################################################
    # eval the model
    args.dataset = ['pets', 'flower102', 'eurosat']
    eval_mm(args, preprocess, my_clip_model=None, criterion=criterion, device=device, model=model)
