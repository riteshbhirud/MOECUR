import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers.activations import ACT2FN


class CURDeepseekMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rank_gate: int,
        rank_up: int,
        rank_down: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.rank_gate = rank_gate
        self.rank_up = rank_up
        self.rank_down = rank_down
        
        self.gate_C = nn.Linear(rank_gate, intermediate_size, bias=False)
        self.gate_U = nn.Linear(rank_gate, rank_gate, bias=False)
        self.gate_R = nn.Linear(hidden_size, rank_gate, bias=False)
        
        self.up_C = nn.Linear(rank_up, intermediate_size, bias=False)
        self.up_U = nn.Linear(rank_up, rank_up, bias=False)
        self.up_R = nn.Linear(hidden_size, rank_up, bias=False)
        
        self.down_C = nn.Linear(rank_down, hidden_size, bias=False)
        self.down_U = nn.Linear(rank_down, rank_down, bias=False)
        self.down_R = nn.Linear(intermediate_size, rank_down, bias=False)
        
        self.act_fn = ACT2FN[hidden_act]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.gate_C(self.gate_U(self.gate_R(x)))
        gate_activated = self.act_fn(gate_out)
        
        up_out = self.up_C(self.up_U(self.up_R(x)))
        
        intermediate = gate_activated * up_out
        
        out = self.down_C(self.down_U(self.down_R(intermediate)))
        
        return out


class CURDeepseekMLPSharedR(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rank_gate: int,
        rank_up: int,
        rank_down: int,
        shared_R_gate: nn.Module,
        shared_R_up: nn.Module,
        shared_R_down: nn.Module,
        hidden_act: str = "silu",
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.rank_gate = rank_gate
        self.rank_up = rank_up
        self.rank_down = rank_down
        
        self.gate_C = nn.Linear(rank_gate, intermediate_size, bias=False)
        self.gate_U = nn.Linear(rank_gate, rank_gate, bias=False)
        
        self.up_C = nn.Linear(rank_up, intermediate_size, bias=False)
        self.up_U = nn.Linear(rank_up, rank_up, bias=False)
        
        self.down_C = nn.Linear(rank_down, hidden_size, bias=False)
        self.down_U = nn.Linear(rank_down, rank_down, bias=False)
        
        self.shared_R_gate = shared_R_gate
        self.shared_R_up = shared_R_up
        self.shared_R_down = shared_R_down
        
        self.act_fn = ACT2FN[hidden_act]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.gate_C(self.gate_U(self.shared_R_gate(x)))
        gate_activated = self.act_fn(gate_out)
        
        up_out = self.up_C(self.up_U(self.shared_R_up(x)))
        
        intermediate = gate_activated * up_out
        
        out = self.down_C(self.down_U(self.shared_R_down(intermediate)))
        
        return out


class DeepseekMoEGate(nn.Module):
    def __init__(self, hidden_size: int, n_routed_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.weight = nn.Parameter(torch.empty((n_routed_experts, hidden_size)))
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        logits = F.linear(hidden_states, self.weight, None)
        scores = F.softmax(logits, dim=-1)
        
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        
        return topk_idx, topk_weight


class CURDeepseekMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        n_shared_experts: int,
        num_experts_per_tok: int,
        rank_gate: int,
        rank_up: int,
        rank_down: int,
        shared_rank_gate: Optional[int] = None,
        shared_rank_up: Optional[int] = None,
        shared_rank_down: Optional[int] = None,
        hidden_act: str = "silu",
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        self.rank_gate = rank_gate
        self.rank_up = rank_up
        self.rank_down = rank_down
        
        self.shared_rank_gate = shared_rank_gate or rank_gate
        self.shared_rank_up = shared_rank_up or rank_up
        self.shared_rank_down = shared_rank_down or rank_down
        
        self.gate = DeepseekMoEGate(hidden_size, n_routed_experts, num_experts_per_tok)
        
        self.shared_R_gate = nn.Linear(hidden_size, rank_gate, bias=False)
        self.shared_R_up = nn.Linear(hidden_size, rank_up, bias=False)
        self.shared_R_down = nn.Linear(moe_intermediate_size, rank_down, bias=False)
        
        nn.init.zeros_(self.shared_R_gate.weight)
        nn.init.zeros_(self.shared_R_up.weight)
        nn.init.zeros_(self.shared_R_down.weight)
        
        self.experts = nn.ModuleList([
            CURDeepseekMLPSharedR(
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size,
                rank_gate=rank_gate,
                rank_up=rank_up,
                rank_down=rank_down,
                shared_R_gate=self.shared_R_gate,
                shared_R_up=self.shared_R_up,
                shared_R_down=self.shared_R_down,
                hidden_act=hidden_act,
            )
            for _ in range(n_routed_experts)
        ])
        
        if n_shared_experts is not None and n_shared_experts > 0:
            shared_intermediate_size = moe_intermediate_size * n_shared_experts
            self.shared_experts = CURDeepseekMLP(
                hidden_size=hidden_size,
                intermediate_size=shared_intermediate_size,
                rank_gate=self.shared_rank_gate,
                rank_up=self.shared_rank_up,
                rank_down=self.shared_rank_down,
                hidden_act=hidden_act,
            )
        else:
            self.shared_experts = None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        
        topk_idx, topk_weight = self.gate(hidden_states)
        
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        y = self._moe_forward(hidden_states, flat_topk_idx, topk_weight)
        y = y.view(*orig_shape)
        
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        
        return y
    
    def _moe_forward(
        self,
        hidden_states: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_cache = torch.zeros_like(hidden_states)
        
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy()
        
        if len(tokens_per_expert) < self.n_routed_experts:
            tokens_per_expert = list(tokens_per_expert) + [0] * (self.n_routed_experts - len(tokens_per_expert))
        
        tokens_per_expert_cumsum = [0] + list(torch.tensor(tokens_per_expert).cumsum(0).numpy())
        token_idxs = idxs // self.num_experts_per_tok
        
        for i in range(self.n_routed_experts):
            start_idx = tokens_per_expert_cumsum[i]
            end_idx = tokens_per_expert_cumsum[i + 1]
            
            if start_idx == end_idx:
                continue
            
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = hidden_states[exp_token_idx]
            
            expert_out = expert(expert_tokens)
            expert_out = expert_out * expert_weights.view(-1, 1)[idxs[start_idx:end_idx]]
            
            expert_cache.scatter_add_(
                0, 
                exp_token_idx.view(-1, 1).expand(-1, hidden_states.shape[-1]), 
                expert_out
            )
        
        return expert_cache


def is_deepseek_moe_layer(layer) -> bool:
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        return hasattr(mlp, 'experts') and hasattr(mlp, 'gate')
    return False


def get_deepseek_moe_config(config) -> dict:
    return {
        'hidden_size': config.hidden_size,
        'moe_intermediate_size': getattr(config, 'moe_intermediate_size', config.intermediate_size // 4),
        'n_routed_experts': getattr(config, 'n_routed_experts', 64),
        'n_shared_experts': getattr(config, 'n_shared_experts', 2),
        'num_experts_per_tok': getattr(config, 'num_experts_per_tok', 6),
        'hidden_act': getattr(config, 'hidden_act', 'silu'),
        'first_k_dense_replace': getattr(config, 'first_k_dense_replace', 1),
        'moe_layer_freq': getattr(config, 'moe_layer_freq', 1),
    }
