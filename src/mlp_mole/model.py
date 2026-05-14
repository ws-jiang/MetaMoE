"""
MoLE Model
"""
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP

from .config import MLPMoleConfig
from .layer import MLPMoLE, MLPMoLELayer
from ..lora import LoraModel


class MLPMoleModel(LoraModel):
    """
    MoLE (Mixture of LoRA Experts) Model
    """
    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name="default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self, mole_config: MLPMoleConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:
        """
        Inplace replacement of the target module with the adapter layer
        """
        kwargs = {
            "lora_rank": mole_config.lora_rank,
            "lora_alpha": mole_config.lora_alpha,
            "lora_dropout": mole_config.lora_dropout,
            "init_lora_weights": mole_config.init_lora_weights,
            "num_experts": mole_config.num_experts,
            "top_k": mole_config.top_k,
            "threshold": mole_config.threshold,
        }

        if isinstance(target, MLPMoLE):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(adapter_name: str, target: nn.Module, **kwargs: Any) -> nn.Module:
        """
        Create the new LoRA module for the target module
        """
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, LlamaMLP):
            new_module = MLPMoLELayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module `{target}` is not supported. "
                f"Currently, only the following modules are supported: `LlamaMLP`.")

        return new_module
    
    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        """
        Replace the module
        """
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        for name, module in new_module.named_modules():
            if self.prefix in name:
                module.cuda()
    
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Make only adapters as trainable
        """
        for name, param in model.named_parameters():
            if self.prefix not in name:
                param.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            elif bias == "all":
                for name, param in model.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            elif bias == "lora_only":
                raise ValueError("`lora_only` bias is not supported for MLPMoLE.")
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def get_aux_loss(self, adapter_name="default") -> torch.Tensor:
        """
        Get the load balancing loss for the whole model
        """
        model_loss = torch.tensor(0, dtype=torch.float).to(self.model.device)

        for name, module in self.model.named_modules():
            if name.endswith('moe_layer'):
                layer_loss = module[adapter_name].layer_loss
                model_loss += layer_loss

        return model_loss
    
