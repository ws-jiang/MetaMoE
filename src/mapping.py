"""
Configure and Model Mappings

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""

from .mlp_mole import MLPMoleConfig, MLPMoleModel
from .lora import LoraConfig, LoraModel
from .utils.peft_types import PeftType

PEFT_TYPE_TO_CONFIG_MAPPING = {
    PeftType.LORA: LoraConfig,
    PeftType.MLPMOLE: MLPMoleConfig,
}
PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.MLPMOLE: MLPMoleModel,
}
