from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RelaxedCURProjection(nn.Module):
    def __init__(
        self,
        C_eff: torch.Tensor,
        U: torch.Tensor,
        R_eff: torch.Tensor,
        col_indices: List[int],
        row_indices: List[int],
        col_offset_norms: List[float],
        row_offset_norms: List[float],
    ):
        super().__init__()
        
        self.C_eff = nn.Parameter(C_eff.clone())
        self.U = nn.Parameter(U.clone())
        self.R_eff = nn.Parameter(R_eff.clone())
        
        self.register_buffer('col_indices_tensor', torch.tensor(col_indices, dtype=torch.long))
        self.register_buffer('row_indices_tensor', torch.tensor(row_indices, dtype=torch.long))
        self.register_buffer('col_offset_norms_tensor', torch.tensor(col_offset_norms, dtype=torch.float32))
        self.register_buffer('row_offset_norms_tensor', torch.tensor(row_offset_norms, dtype=torch.float32))
        
        self.col_indices = col_indices
        self.row_indices = row_indices
        self.col_offset_norms = col_offset_norms
        self.row_offset_norms = row_offset_norms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.R_eff.T
        out = out @ self.U.T
        out = out @ self.C_eff.T
        return out
    
    def storage_elements(self) -> int:
        param_elements = self.C_eff.numel() + self.U.numel() + self.R_eff.numel()
        buffer_elements = (self.col_indices_tensor.numel() + 
                          self.row_indices_tensor.numel() +
                          self.col_offset_norms_tensor.numel() + 
                          self.row_offset_norms_tensor.numel())
        return param_elements + buffer_elements
    
    def get_interpretability_metrics(self) -> Dict:
        return {
            'col_indices': self.col_indices,
            'row_indices': self.row_indices,
            'col_offset_norms': self.col_offset_norms,
            'row_offset_norms': self.row_offset_norms,
            'avg_col_offset': np.mean(self.col_offset_norms),
            'avg_row_offset': np.mean(self.row_offset_norms),
            'col_reliability': [1.0 / (1.0 + n) for n in self.col_offset_norms],
        }


class RelaxedCURExpert(nn.Module):
    def __init__(self, hidden_act: str = "silu"):
        super().__init__()
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[hidden_act]
        
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
    
    def set_projection(self, proj_name: str, projection: RelaxedCURProjection):
        setattr(self, proj_name, projection)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down
    
    def storage_elements(self) -> int:
        return (self.gate_proj.storage_elements() + 
                self.up_proj.storage_elements() + 
                self.down_proj.storage_elements())
    
    def original_elements(self, hidden_size: int, intermediate_size: int) -> int:
        return 2 * intermediate_size * hidden_size + hidden_size * intermediate_size
    
    def get_interpretability_metrics(self) -> Dict:
        return {
            'gate_proj': self.gate_proj.get_interpretability_metrics(),
            'up_proj': self.up_proj.get_interpretability_metrics(),
            'down_proj': self.down_proj.get_interpretability_metrics(),
        }


class RelaxedCURMoE(nn.Module):
    def __init__(
        self,
        config: dict,
        experts: nn.ModuleList,
        gate_weight: torch.Tensor,
        shared_experts: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config.get('moe_intermediate_size', config.get('intermediate_size', 1408))
        self.n_experts = len(experts)
        self.num_experts_per_tok = config['num_experts_per_tok']
        
        self.gate_weight = nn.Parameter(gate_weight)
        self.experts = experts
        self.shared_experts = shared_experts
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        router_logits = F.linear(hidden_states_flat, self.gate_weight)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weight, topk_idx = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        
        final_hidden = torch.zeros_like(hidden_states_flat)
        
        for expert_idx in range(self.n_experts):
            expert = self.experts[expert_idx]
            mask = (topk_idx == expert_idx).any(dim=-1)
            
            if mask.sum() == 0:
                continue
            
            expert_input = hidden_states_flat[mask]
            expert_output = expert(expert_input)
            
            weight_mask = (topk_idx == expert_idx)
            expert_weights = (topk_weight * weight_mask).sum(dim=-1)[mask]
            
            final_hidden[mask] += expert_output * expert_weights.unsqueeze(-1)
        
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)
            final_hidden = final_hidden + shared_output
        
        return final_hidden.view(batch_size, seq_len, hidden_dim)
    
    def storage_elements(self) -> int:
        total = self.gate_weight.numel()
        for expert in self.experts:
            total += expert.storage_elements()
        if self.shared_experts is not None:
            total += sum(p.numel() for p in self.shared_experts.parameters())
            total += sum(b.numel() for b in self.shared_experts.buffers())
        return total
    
    def original_elements(self) -> int:
        total = self.gate_weight.numel()
        for expert in self.experts:
            total += expert.original_elements(self.hidden_size, self.intermediate_size)
        if self.shared_experts is not None:
            total += sum(p.numel() for p in self.shared_experts.parameters())
            total += sum(b.numel() for b in self.shared_experts.buffers())
        return total
