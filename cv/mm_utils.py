import argparse

import clip
import torch
from torch import nn

from src.utils.constants import DEFAULT_DATA_DIR


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def validate(val_loader, model, criterion, args, epoch, device):
    model.eval()
    running_loss = 0.0
    total = 0 
    correct = 0
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            running_loss += loss.item() * images.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / total if total > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    print(f"Validation Epoch [{epoch+1}/{args.num_epochs}] Loss: {avg_loss:.4f} Acc: {accuracy:.2f} %")
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, default='vit_b32', choices=['vit_b16', 'vit_b32', 'vit_l14'])
    parser.add_argument('--lora_target_modules', type=str, default='c_fc,c_proj')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--dataset', type=str, default='pets', choices=['pets', 'flower102', 'eurosat', 'dtd',
                                                                        'resisc45', 'food101', 'ucf', 'svhn',
                                                                        'cifar100', 'imagenet'])
    parser.add_argument('--root_data_path', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--job_id', type=str, default='JOBID')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--num_proxy_samples', type=int, default=0)
    parser.add_argument('--DPP', default=False, action='store_true')
    parser.add_argument('--proxy_all', default="", type=str)
    parser.add_argument('--rebuttal_forgetting', default="no", type=str)

    return parser.parse_args()


class MyCLIP(nn.Module):
    def __init__(self, model, text_features):
        super().__init__()
        self.model = model
        self.text_features = text_features

    def forward(self, image):
        image_features = self.model.visual(image.type(self.model.dtype))
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-6)
        logit_scale = self.model.logit_scale.exp()
        text_features = self.text_features.to(image_features.device)
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image