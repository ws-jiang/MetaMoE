"""
Shared model utilities used by both the CV and NLP pipelines.
"""


def set_direct_forward_base_layer(model, direct_forward_base_layer: bool = True):
    for name, module in model.named_modules():
        if name.endswith('mlp'):
            module.direct_forward_base_layer = direct_forward_base_layer


def print_fraction_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        total_params += param.numel()

    trainable_fraction = trainable_params / total_params if total_params > 0 else 0
    print(f"\nParameter Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Fraction of trainable parameters: {trainable_fraction:.4f} ({trainable_fraction * 100:.2f}%)")
