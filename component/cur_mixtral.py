import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from transformers.activations import ACT2FN


class CURMixtralExpert(nn.Module):

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


class CURMixtralExpertSharedR(nn.Module):

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


class MixtralMoEGate(nn.Module):

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size)))
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


class CURMixtralSparseMoeBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rank_gate: int,
        rank_up: int,
        rank_down: int,
        hidden_act: str = "silu",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.rank_gate = rank_gate
        self.rank_up = rank_up
        self.rank_down = rank_down

        self.gate = MixtralMoEGate(hidden_size, num_experts, num_experts_per_tok)

        self.shared_R_gate = nn.Linear(hidden_size, rank_gate, bias=False)
        self.shared_R_up = nn.Linear(hidden_size, rank_up, bias=False)
        self.shared_R_down = nn.Linear(intermediate_size, rank_down, bias=False)

        nn.init.zeros_(self.shared_R_gate.weight)
        nn.init.zeros_(self.shared_R_up.weight)
        nn.init.zeros_(self.shared_R_down.weight)

        self.experts = nn.ModuleList([
            CURMixtralExpertSharedR(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rank_gate=rank_gate,
                rank_up=rank_up,
                rank_down=rank_down,
                shared_R_gate=self.shared_R_gate,
                shared_R_up=self.shared_R_up,
                shared_R_down=self.shared_R_down,
                hidden_act=hidden_act,
            )
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape

        topk_idx, topk_weight = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        y = self._moe_forward(hidden_states, flat_topk_idx, topk_weight)
        y = y.view(*orig_shape)

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

        if len(tokens_per_expert) < self.num_experts:
            tokens_per_expert = list(tokens_per_expert) + [0] * (self.num_experts - len(tokens_per_expert))

        tokens_per_expert_cumsum = [0] + list(torch.tensor(tokens_per_expert).cumsum(0).numpy())
        token_idxs = idxs // self.num_experts_per_tok

        for i in range(self.num_experts):
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


class MixtralRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MixtralRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class CURMixtralAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rank_q: Optional[int] = None,
        rank_k: Optional[int] = None,
        rank_v: Optional[int] = None,
        rank_o: Optional[int] = None,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = MixtralRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def is_mixtral_moe_layer(layer) -> bool:
    return hasattr(layer, 'block_sparse_moe')


def get_mixtral_moe_config(config) -> dict:
    return {
        'hidden_size': config.hidden_size,
        'intermediate_size': config.intermediate_size,
        'num_experts': getattr(config, 'num_local_experts', 8),
        'num_experts_per_tok': getattr(config, 'num_experts_per_tok', 2),
        'hidden_act': getattr(config, 'hidden_act', 'silu'),
        'num_hidden_layers': config.num_hidden_layers,
    }


def get_mixtral_expert_projections(expert) -> dict:
    return {
        'gate_proj': expert.w1,
        'up_proj': expert.w3,
        'down_proj': expert.w2,
    }