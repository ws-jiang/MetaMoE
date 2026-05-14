"""
MoLE Layer
"""
import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import autocast


class TopKMoELayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer with the Top-k

    Adapted from https://github.com/mistralai/mistral-src
    """

    def __init__(self, experts: nn.ModuleList, gate: nn.Module, top_k: int, lambda_: float):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.top_k = top_k
        self.layer_loss = None
        self.lambda_ = nn.Parameter(torch.tensor(lambda_, dtype=torch.float32))

    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        # with torch.autocast(device_type="cuda"):
        # print(f"sample embeddings: {self.sample_embeddings.shape}")
        inputs = inputs.to(self.gate.weight.device)
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))

        # Clamp lambda_ to be between 0 and 1
        lambda_clamped = torch.clamp(self.lambda_, 0.0, 1.0)
        
        # Use lambda_clamped as a parameter (it will be automatically converted to tensor)
        if lambda_clamped > 0.01:
            gatting_inputs = (1.0-lambda_clamped)*inputs + lambda_clamped*self.sample_embeddings.to(inputs.device).unsqueeze(1).expand_as(inputs)
        else:
            gatting_inputs = inputs
        
        gate_logits = F.softmax(self.gate(gatting_inputs.view(-1, gatting_inputs.shape[-1])), dim=-1)

        weights, selected_experts = torch.topk(input=gate_logits, k=self.top_k, dim=-1)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
        results = torch.zeros_like(self.experts[0](flattened_inputs))

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += \
                weights[batch_idx, nth_expert, None] * expert(flattened_inputs[batch_idx])

        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        if inputs.requires_grad:
            self.layer_loss = self.get_layer_loss(gate_logits=gate_logits, selected_experts=selected_experts)
        return results

class MLPLoRAExpert(nn.Module, BaseTunerLayer, ABC):
    """
    LoRA Layer for MLP
    """

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        # print(f"kwargs: {kwargs}")
        # print(f"base_layer: {base_layer}")
        super().__init__()
        self.lora_rank = {}
        self.lora_alpha = {}
        self.base_layer = base_layer
        
        self.up_lora_A = nn.ModuleDict({})
        self.up_lora_B = nn.ModuleDict({})
        self.up_lora_dropout = nn.ModuleDict({})
        self.up_scaling = {}

        self.down_lora_A = nn.ModuleDict({})
        self.down_lora_B = nn.ModuleDict({})
        self.down_lora_dropout = nn.ModuleDict({})
        self.down_scaling = {}

        self.gate_lora_A = nn.ModuleDict({})
        self.gate_lora_B = nn.ModuleDict({})
        self.gate_lora_dropout = nn.ModuleDict({})
        self.gate_scaling = {}

        self.up_in_features = base_layer.up_proj.in_features
        self.up_out_features = base_layer.up_proj.out_features

        self.down_in_features = base_layer.down_proj.in_features
        self.down_out_features = base_layer.down_proj.out_features

        self.gate_in_features = base_layer.gate_proj.in_features
        self.gate_out_features = base_layer.gate_proj.out_features

    def zero_layer(self, adapter_name: str):
        self.up_scaling[adapter_name] = 0
        self.down_scaling[adapter_name] = 0
        self.gate_scaling[adapter_name] = 0
    
    def update_layer(
        self, adapter_name: str, lora_rank: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
    ) -> None:
        """
        Update the layer
        """
        # print("updating LoraLayer")
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        self.lora_rank[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha

        self.up_lora_A[adapter_name] = nn.Linear(self.up_in_features, lora_rank, bias=False)
        self.up_lora_B[adapter_name] = nn.Linear(lora_rank, self.up_out_features, bias=False)
        self.up_lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.up_scaling[adapter_name] = lora_alpha / lora_rank
        
        self.down_lora_A[adapter_name] = nn.Linear(self.down_in_features, lora_rank, bias=False)
        self.down_lora_B[adapter_name] = nn.Linear(lora_rank, self.down_out_features, bias=False)
        self.down_lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.down_scaling[adapter_name] = lora_alpha / lora_rank

        self.gate_lora_A[adapter_name] = nn.Linear(self.gate_in_features, lora_rank, bias=False)
        self.gate_lora_B[adapter_name] = nn.Linear(lora_rank, self.gate_out_features, bias=False)
        self.gate_lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.gate_scaling[adapter_name] = lora_alpha / lora_rank

        self.reset_parameters(adapter_name, init_lora_weights)
        self.set_adapter(self.active_adapters)

    def reset_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        """
        Reset the parameters
        """
        if init_lora_weights is False:
            return
        elif adapter_name in self.up_lora_A.keys():
            # Initialize up projection LoRA parameters
            nn.init.kaiming_uniform_(self.up_lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_lora_B[adapter_name].weight)
            
            # Initialize down projection LoRA parameters
            nn.init.kaiming_uniform_(self.down_lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_lora_B[adapter_name].weight)
            
            # Initialize gate projection LoRA parameters
            nn.init.kaiming_uniform_(self.gate_lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate_lora_B[adapter_name].weight)
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights
        """
        raise NotImplementedError

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base weights
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        previous_dtype = x.dtype
        
        assert len(self.active_adapters) == 1, "MLP-MoLE only supports one adapter"

        for active_adapter in self.active_adapters:
            if active_adapter not in self.up_lora_A.keys():
                continue

            with autocast(device_type="cuda"):
                x = x.to(self.up_lora_A[active_adapter].weight.dtype)
                x = x.to(self.up_lora_A[active_adapter].weight.device)

                up_base_result = self.base_layer.up_proj(x)
                up_lora_result = self.up_lora_B[active_adapter](self.up_lora_A[active_adapter](self.up_lora_dropout[active_adapter](x))) * self.up_scaling[active_adapter]
                # print(f"up_lora_result: {up_lora_result.device}")
                up_result = up_base_result + up_lora_result.to(up_base_result.device)

                gate_base_result = self.base_layer.gate_proj(x)
                gate_lora_result = self.gate_lora_B[active_adapter](self.gate_lora_A[active_adapter](self.gate_lora_dropout[active_adapter](x))) * self.gate_scaling[active_adapter]
                gate_result = self.base_layer.act_fn(gate_base_result + gate_lora_result.to(gate_base_result.device))

                up_output = up_result *  gate_result

                down_base_result = self.base_layer.down_proj(up_output)
                down_lora_result = self.down_lora_B[active_adapter](self.down_lora_A[active_adapter](self.down_lora_dropout[active_adapter](up_output))) * self.down_scaling[active_adapter]
                down_result = down_base_result + down_lora_result.to(down_base_result.device)

        return down_result.to(previous_dtype)

class MLPMoLE(MLPLoRAExpert, ABC):
    """
    MoLE Layer
    """

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.lora_gating = nn.ModuleDict({})
        self.moe_layer = nn.ModuleDict({})

    def update_layer(
        self, adapter_name: str, lora_rank: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
        num_experts: int, top_k: int, threshold: float, lambda_: float, zero_first_expert: bool = False,
    ) -> None:
        """
        Update the layer
        """

        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        if (top_k is not None) and (threshold is not None):
            raise ValueError(f"Only one of the top-k {top_k} and the threshold {threshold} can be used.")
        elif (top_k is None) and (threshold is None):
            raise ValueError(f"At least one of the top-k {top_k} and the threshold {threshold} should be used.")

        self.lora_gating[adapter_name] = nn.Linear(self.base_layer.up_proj.in_features, num_experts, bias=False)
        experts = nn.ModuleList(MLPLoRAExpert(base_layer=self.base_layer) for i in range(num_experts))

        for expert in experts:
            expert.update_layer(adapter_name, lora_rank, lora_alpha, lora_dropout, init_lora_weights)

        if zero_first_expert:
            experts[0].zero_layer(adapter_name)

        self.moe_layer[adapter_name] = TopKMoELayer(
            experts=experts, gate=self.lora_gating[adapter_name], top_k=top_k, lambda_=lambda_)

        self.set_adapter(self.active_adapters)
        # print(self.active_adapters, flush=True)

class MLPMoLELayer(MLPMoLE):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 4,
        top_k: int = None,
        threshold: float = None,
        lambda_: float = 1.0,
        zero_first_expert: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(base_layer=base_layer, **kwargs)
        # print(f"kwargs: {kwargs}")
        # MLPMoLE.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, lora_rank, lora_alpha, lora_dropout, init_lora_weights, num_experts, top_k, threshold, lambda_, zero_first_expert)
        self.direct_forward_base_layer = False

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights
        """
        pass

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base weights
        """
        pass

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        previous_dtype = x.dtype

        assert len(self.active_adapters) == 1, "MLP-MoLE only supports one adapter"

        for active_adapter in self.active_adapters:
            if active_adapter not in self.moe_layer.keys():
                continue

            if self.direct_forward_base_layer:
                result = self.base_layer(x)
            else:
                moe_layer = self.moe_layer[active_adapter]
                x = x.to(moe_layer.experts[0].up_lora_A[active_adapter].weight.dtype)
                result = moe_layer(x)

        return result.to(previous_dtype)
