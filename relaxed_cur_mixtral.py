#!/usr/bin/env python3

import os
import sys
import json
import argparse
import copy
import gc
import tempfile
import shutil
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import torch.multiprocessing as torch_mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cur import select_rows_and_columns
from component.cur_mixtral import (
    is_mixtral_moe_layer,
    get_mixtral_moe_config,
    CURMixtralExpert,
    CURMixtralExpertSharedR,
    CURMixtralSparseMoeBlock,
    MixtralMoEGate,
)


def get_optimization_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_num_parallel_workers():
    if torch.cuda.is_available():
        return min(4, mp.cpu_count())
    else:
        return min(8, mp.cpu_count())


@dataclass
class RelaxedCURResult:
    C_eff: torch.Tensor
    U: torch.Tensor
    R_eff: torch.Tensor

    C_base: torch.Tensor
    R_base: torch.Tensor

    col_indices: List[int]
    row_indices: List[int]

    col_offset_norms: List[float]
    row_offset_norms: List[float]

    col_cosine_sims: List[float]
    row_cosine_sims: List[float]

    initial_error: float
    final_error: float

    def reconstruct(self) -> torch.Tensor:
        return self.C_eff @ self.U @ self.R_eff

    def get_column_reliability(self) -> List[float]:
        return [1.0 / (1.0 + norm) for norm in self.col_offset_norms]

    def storage_size(self) -> int:
        return self.C_eff.numel() + self.U.numel() + self.R_eff.numel()


def optimize_relaxed_cur(
    W: torch.Tensor,
    col_indices: List[int],
    row_indices: List[int],
    n_iterations: int = 500,
    lr: float = 0.01,
    lambda_reg: float = 0.1,
    verbose: bool = False,
    device: Optional[torch.device] = None,
) -> RelaxedCURResult:
    m, n = W.shape
    k = len(col_indices)

    if device is None:
        device = get_optimization_device()

    W_f = W.float().to(device)

    col_idx_tensor = torch.tensor(col_indices, dtype=torch.long, device=device)
    row_idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)

    C_base = W_f.index_select(1, col_idx_tensor).clone()
    R_base = W_f.index_select(0, row_idx_tensor).clone()

    try:
        C_pinv = torch.linalg.pinv(C_base)
        R_pinv = torch.linalg.pinv(R_base)
        U_init = C_pinv @ W_f @ R_pinv
    except:
        U_init = torch.eye(k, device=device, dtype=torch.float32) * 0.01

    W_norm = torch.norm(W_f)
    W_approx_init = C_base @ U_init @ R_base
    initial_error = (torch.norm(W_f - W_approx_init) / W_norm).item()

    delta_C = torch.zeros(m, k, device=device, dtype=torch.float32, requires_grad=True)
    delta_R = torch.zeros(k, n, device=device, dtype=torch.float32, requires_grad=True)
    U = U_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([delta_C, delta_R, U], lr=lr * 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)

    best_error = float('inf')
    best_state = None
    patience = 20
    no_improve_count = 0
    min_improvement = 1e-4
    check_interval = 5

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        C_eff = C_base + delta_C
        R_eff = R_base + delta_R

        W_approx = C_eff @ U @ R_eff

        recon_loss = torch.norm(W_f - W_approx) ** 2

        reg_loss = lambda_reg * (torch.norm(delta_C) ** 2 + torch.norm(delta_R) ** 2)

        total_loss = recon_loss + reg_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % check_interval == 0:
            current_error = (torch.sqrt(recon_loss) / W_norm).item()
            if current_error < best_error - min_improvement:
                best_error = current_error
                best_state = {
                    'delta_C': delta_C.detach().clone(),
                    'delta_R': delta_R.detach().clone(),
                    'U': U.detach().clone(),
                }
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                if verbose:
                    print(f"      Early stopping at iter {iteration+1} (no improvement for {patience * check_interval} iters)")
                break

        if verbose and (iteration + 1) % 100 == 0:
            current_error = (torch.sqrt(recon_loss) / W_norm).item()
            offset_norm = (torch.norm(delta_C) + torch.norm(delta_R)).item()
            print(f"      Iter {iteration+1}: error={current_error:.4f}, "
                  f"offset_norm={offset_norm:.4f}")

    delta_C_final = best_state['delta_C']
    delta_R_final = best_state['delta_R']
    U_final = best_state['U']

    col_base_norms = torch.norm(C_base, dim=0) + 1e-10
    row_base_norms = torch.norm(R_base, dim=1) + 1e-10
    col_offset_norms = (torch.norm(delta_C_final, dim=0) / col_base_norms).cpu().tolist()
    row_offset_norms = (torch.norm(delta_R_final, dim=1) / row_base_norms).cpu().tolist()

    C_eff_final = C_base + delta_C_final
    R_eff_final = R_base + delta_R_final

    col_cosine_sims = []
    for i in range(k):
        c_base = C_base[:, i]
        c_eff = C_eff_final[:, i]
        cos_sim = F.cosine_similarity(c_base.unsqueeze(0), c_eff.unsqueeze(0)).item()
        col_cosine_sims.append(cos_sim)

    row_cosine_sims = []
    for i in range(k):
        r_base = R_base[i, :]
        r_eff = R_eff_final[i, :]
        cos_sim = F.cosine_similarity(r_base.unsqueeze(0), r_eff.unsqueeze(0)).item()
        row_cosine_sims.append(cos_sim)

    W_approx_final = C_eff_final @ U_final @ R_eff_final
    final_error = (torch.norm(W_f - W_approx_final) / W_norm).item()

    return RelaxedCURResult(
        C_eff=C_eff_final.cpu(),
        U=U_final.cpu(),
        R_eff=R_eff_final.cpu(),
        C_base=C_base.cpu(),
        R_base=R_base.cpu(),
        col_indices=col_indices,
        row_indices=row_indices,
        col_offset_norms=col_offset_norms,
        row_offset_norms=row_offset_norms,
        col_cosine_sims=col_cosine_sims,
        row_cosine_sims=row_cosine_sims,
        initial_error=initial_error,
        final_error=final_error,
    )


def relaxed_cur_decompose(
    W: torch.Tensor,
    rank: int = 512,
    cur_mode: str = 'deim',
    n_iterations: int = 500,
    lr: float = 0.01,
    lambda_reg: float = 0.1,
    verbose: bool = False,
    device: Optional[torch.device] = None,
) -> RelaxedCURResult:
    m, n = W.shape
    actual_rank = min(rank, m - 1, n - 1)

    W_cpu = W.float().cpu()
    S = W_cpu.abs()
    row_indices, col_indices = select_rows_and_columns(
        W_cpu, S, actual_rank, actual_rank,
        aux_mode='weight', cur_mode=cur_mode
    )
    col_indices = list(col_indices)
    row_indices = list(row_indices)

    result = optimize_relaxed_cur(
        W,
        col_indices,
        row_indices,
        n_iterations=n_iterations,
        lr=lr,
        lambda_reg=lambda_reg,
        verbose=verbose,
        device=device,
    )

    return result


def batch_optimize_projections(
    weights: List[torch.Tensor],
    col_indices_list: List[List[int]],
    row_indices_list: List[List[int]],
    n_iterations: int = 500,
    lr: float = 0.01,
    lambda_reg: float = 0.1,
    device: Optional[torch.device] = None,
) -> List[RelaxedCURResult]:
    if device is None:
        device = get_optimization_device()

    results = []
    for W, col_indices, row_indices in zip(weights, col_indices_list, row_indices_list):
        result = optimize_relaxed_cur(
            W,
            col_indices,
            row_indices,
            n_iterations=n_iterations,
            lr=lr,
            lambda_reg=lambda_reg,
            verbose=False,
            device=device,
        )
        results.append(result)

    return results


def random_cur_decompose(
    W: torch.Tensor,
    rank: int = 512,
    n_iterations: int = 500,
    lr: float = 0.01,
    lambda_reg: float = 0.1,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> RelaxedCURResult:
    m, n = W.shape
    actual_rank = min(rank, m - 1, n - 1)

    torch.manual_seed(seed)
    col_indices = torch.randperm(n)[:actual_rank].tolist()
    row_indices = torch.randperm(m)[:actual_rank].tolist()

    result = optimize_relaxed_cur(
        W,
        col_indices,
        row_indices,
        n_iterations=n_iterations,
        lr=lr,
        lambda_reg=lambda_reg,
        verbose=False,
        device=device,
    )

    return result


class RelaxedCURProjection(nn.Module):

    def __init__(
        self,
        C_eff: torch.Tensor,
        U: torch.Tensor,
        R_eff: torch.Tensor,
        C_base: torch.Tensor,
        R_base: torch.Tensor,
        col_indices: List[int],
        row_indices: List[int],
        col_offset_norms: List[float],
        row_offset_norms: List[float],
        col_cosine_sims: List[float],
        row_cosine_sims: List[float],
    ):
        super().__init__()

        self.C_eff = nn.Parameter(C_eff.clone())
        self.U = nn.Parameter(U.clone())
        self.R_eff = nn.Parameter(R_eff.clone())

        self.register_buffer('C_base', C_base.clone())
        self.register_buffer('R_base', R_base.clone())

        self.register_buffer('col_indices_tensor', torch.tensor(col_indices, dtype=torch.long))
        self.register_buffer('row_indices_tensor', torch.tensor(row_indices, dtype=torch.long))

        self.register_buffer('col_offset_norms_tensor', torch.tensor(col_offset_norms, dtype=torch.float32))
        self.register_buffer('row_offset_norms_tensor', torch.tensor(row_offset_norms, dtype=torch.float32))
        self.register_buffer('col_cosine_sims_tensor', torch.tensor(col_cosine_sims, dtype=torch.float32))
        self.register_buffer('row_cosine_sims_tensor', torch.tensor(row_cosine_sims, dtype=torch.float32))

        self.col_indices = col_indices
        self.row_indices = row_indices
        self.col_offset_norms = col_offset_norms
        self.row_offset_norms = row_offset_norms
        self.col_cosine_sims = col_cosine_sims
        self.row_cosine_sims = row_cosine_sims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.R_eff.T
        out = out @ self.U.T
        out = out @ self.C_eff.T
        return out

    def storage_elements(self) -> int:
        param_elements = self.C_eff.numel() + self.U.numel() + self.R_eff.numel()
        index_elements = self.col_indices_tensor.numel() + self.row_indices_tensor.numel()
        return param_elements + index_elements

    def total_storage_with_analysis(self) -> int:
        return sum(p.numel() for p in self.parameters()) + sum(b.numel() for b in self.buffers())

    def compute_current_cosine_sims(self) -> Tuple[List[float], List[float]]:
        col_sims = []
        for i in range(self.C_eff.shape[1]):
            c_base = self.C_base[:, i].float()
            c_eff = self.C_eff[:, i].float()
            sim = F.cosine_similarity(c_base.unsqueeze(0), c_eff.unsqueeze(0)).item()
            col_sims.append(sim)

        row_sims = []
        for i in range(self.R_eff.shape[0]):
            r_base = self.R_base[i, :].float()
            r_eff = self.R_eff[i, :].float()
            sim = F.cosine_similarity(r_base.unsqueeze(0), r_eff.unsqueeze(0)).item()
            row_sims.append(sim)

        return col_sims, row_sims

    def get_interpretability_metrics(self) -> Dict:
        return {
            'col_indices': self.col_indices,
            'row_indices': self.row_indices,
            'col_offset_norms': self.col_offset_norms,
            'row_offset_norms': self.row_offset_norms,
            'col_cosine_sims': self.col_cosine_sims,
            'row_cosine_sims': self.row_cosine_sims,
            'avg_col_offset': np.mean(self.col_offset_norms),
            'avg_row_offset': np.mean(self.row_offset_norms),
            'avg_col_cosine': np.mean(self.col_cosine_sims),
            'avg_row_cosine': np.mean(self.row_cosine_sims),
            'col_reliability': [1.0 / (1.0 + n) for n in self.col_offset_norms],
        }


class RelaxedCURMixtralExpert(nn.Module):

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


class RelaxedCURMixtralMoE(nn.Module):

    def __init__(
        self,
        config: dict,
        experts: nn.ModuleList,
        gate_weight: torch.Tensor,
    ):
        super().__init__()

        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.n_experts = len(experts)
        self.num_experts_per_tok = config['num_experts_per_tok']

        self.gate_weight = nn.Parameter(gate_weight)
        self.experts = experts

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

        return final_hidden.view(batch_size, seq_len, hidden_dim)

    def storage_elements(self) -> int:
        total = self.gate_weight.numel()
        for expert in self.experts:
            total += expert.storage_elements()
        return total

    def original_elements(self) -> int:
        total = self.gate_weight.numel()
        for expert in self.experts:
            total += expert.original_elements(self.hidden_size, self.intermediate_size)
        return total


class RelaxedCURMixtralCompressor:

    def __init__(
        self,
        rank: int = 512,
        cur_mode: str = 'deim',
        n_iterations: int = 500,
        lr: float = 0.01,
        lambda_reg: float = 0.1,
        num_workers: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.rank = rank
        self.cur_mode = cur_mode
        self.n_iterations = n_iterations
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.num_workers = num_workers or get_num_parallel_workers()
        self.device = device or get_optimization_device()

    def compress_projection(
        self,
        weight: torch.Tensor,
        verbose: bool = False,
    ) -> Tuple[RelaxedCURProjection, Dict]:

        result = relaxed_cur_decompose(
            weight,
            rank=self.rank,
            cur_mode=self.cur_mode,
            n_iterations=self.n_iterations,
            lr=self.lr,
            lambda_reg=self.lambda_reg,
            verbose=verbose,
            device=self.device,
        )

        projection = RelaxedCURProjection(
            C_eff=result.C_eff,
            U=result.U,
            R_eff=result.R_eff,
            C_base=result.C_base,
            R_base=result.R_base,
            col_indices=result.col_indices,
            row_indices=result.row_indices,
            col_offset_norms=result.col_offset_norms,
            row_offset_norms=result.row_offset_norms,
            col_cosine_sims=result.col_cosine_sims,
            row_cosine_sims=result.row_cosine_sims,
        )

        original_size = weight.numel()
        compressed_size = result.storage_size()

        stats = {
            'initial_error': result.initial_error,
            'final_error': result.final_error,
            'error_reduction': result.initial_error - result.final_error,
            'original_elements': original_size,
            'compressed_elements': compressed_size,
            'compression_ratio': compressed_size / original_size,
            'space_saving': 1 - compressed_size / original_size,
            'avg_col_cosine_sim': np.mean(result.col_cosine_sims),
            'avg_row_cosine_sim': np.mean(result.row_cosine_sims),
            'avg_col_offset_norm': np.mean(result.col_offset_norms),
            'avg_row_offset_norm': np.mean(result.row_offset_norms),
        }

        return projection, stats

    def compress_expert(
        self,
        expert: nn.Module,
        verbose: bool = False,
    ) -> Tuple[RelaxedCURMixtralExpert, Dict]:

        compressed = RelaxedCURMixtralExpert(hidden_act='silu')
        expert_stats = {
            'projections': {},
            'total_original': 0,
            'total_compressed': 0,
        }

        proj_mapping = {
            'gate_proj': 'w1',
            'up_proj': 'w3',
            'down_proj': 'w2',
        }

        for proj_name, mixtral_name in proj_mapping.items():
            module = getattr(expert, mixtral_name)
            W = module.weight.data

            if verbose:
                print(f"      {proj_name} ({mixtral_name}): shape={tuple(W.shape)}")

            projection, stats = self.compress_projection(W, verbose=verbose)
            compressed.set_projection(proj_name, projection)

            expert_stats['projections'][proj_name] = stats
            expert_stats['total_original'] += stats['original_elements']
            expert_stats['total_compressed'] += stats['compressed_elements']

            if verbose:
                print(f"        CUR error: {stats['initial_error']:.4f} -> "
                      f"Relaxed: {stats['final_error']:.4f} "
                      f"(down {stats['error_reduction']:.4f})")
                print(f"        Avg cosine sim: {stats['avg_col_cosine_sim']:.4f}")

        expert_stats['compression_ratio'] = expert_stats['total_compressed'] / expert_stats['total_original']
        expert_stats['space_saving'] = 1 - expert_stats['compression_ratio']

        return compressed, expert_stats

    def compress_moe_layer(
        self,
        moe: nn.Module,
        moe_config: dict,
        device: str = 'cpu',
        verbose: bool = False,
        parallel: bool = True,
    ) -> Tuple[RelaxedCURMixtralMoE, Dict]:

        n_experts = len(moe.experts)
        compressed_experts = [None] * n_experts
        original_dtype = next(moe.parameters()).dtype
        target_device = torch.device(device) if isinstance(device, str) else device

        layer_stats = {
            'n_experts': n_experts,
            'experts': {},
            'summary': {
                'initial_errors': [],
                'final_errors': [],
                'error_reductions': [],
                'col_cosine_sims': [],
                'col_offset_norms': [],
                'original_elements': 0,
                'compressed_elements': 0,
            }
        }

        for exp_idx in tqdm(range(n_experts), desc="    Compressing experts", leave=False):
            expert = moe.experts[exp_idx]

            if verbose:
                print(f"    Expert {exp_idx}:")

            compressed_expert, stats = self.compress_expert(expert, verbose=verbose)
            compressed_expert = compressed_expert.to(device=target_device, dtype=original_dtype)

            compressed_experts[exp_idx] = compressed_expert
            layer_stats['experts'][exp_idx] = stats

            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                layer_stats['summary']['initial_errors'].append(
                    stats['projections'][proj_name]['initial_error']
                )
                layer_stats['summary']['final_errors'].append(
                    stats['projections'][proj_name]['final_error']
                )
                layer_stats['summary']['error_reductions'].append(
                    stats['projections'][proj_name]['error_reduction']
                )
                layer_stats['summary']['col_cosine_sims'].append(
                    stats['projections'][proj_name]['avg_col_cosine_sim']
                )
                layer_stats['summary']['col_offset_norms'].append(
                    stats['projections'][proj_name]['avg_col_offset_norm']
                )

            layer_stats['summary']['original_elements'] += stats['total_original']
            layer_stats['summary']['compressed_elements'] += stats['total_compressed']

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        compressed_experts = nn.ModuleList(compressed_experts)

        layer_stats['summary']['avg_initial_error'] = np.mean(layer_stats['summary']['initial_errors'])
        layer_stats['summary']['avg_final_error'] = np.mean(layer_stats['summary']['final_errors'])
        layer_stats['summary']['avg_error_reduction'] = np.mean(layer_stats['summary']['error_reductions'])
        layer_stats['summary']['avg_col_cosine_sim'] = np.mean(layer_stats['summary']['col_cosine_sims'])
        layer_stats['summary']['avg_col_offset_norm'] = np.mean(layer_stats['summary']['col_offset_norms'])
        layer_stats['summary']['compression_ratio'] = (
            layer_stats['summary']['compressed_elements'] /
            layer_stats['summary']['original_elements']
        )
        layer_stats['summary']['space_saving'] = 1 - layer_stats['summary']['compression_ratio']

        compressed_moe = RelaxedCURMixtralMoE(
            config=moe_config,
            experts=compressed_experts,
            gate_weight=moe.gate.weight.data.clone(),
        )

        compressed_moe = compressed_moe.to(device=target_device, dtype=original_dtype)

        return compressed_moe, layer_stats


@torch.no_grad()
def cache_teacher_logits(
    teacher_model: nn.Module,
    calib_data: List[Dict],
    cache_dir: str,
    device: str,
) -> str:
    teacher_model.eval()
    os.makedirs(cache_dir, exist_ok=True)

    cached_logits = []

    print(f"  Caching teacher logits to {cache_dir}...")

    for batch_idx, batch in enumerate(tqdm(calib_data, desc="  Caching teacher logits")):
        input_ids = batch['input_ids'].to(device)

        outputs = teacher_model(input_ids, use_cache=False)

        logits = outputs.logits.cpu().to(torch.bfloat16)
        cached_logits.append(logits)

    cache_file = os.path.join(cache_dir, 'teacher_logits.pt')
    torch.save(cached_logits, cache_file)

    cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    print(f"  Cached {len(cached_logits)} batches ({cache_size_mb:.1f} MB)")

    return cache_file


def load_cached_logits(cache_file: str) -> List[torch.Tensor]:
    return torch.load(cache_file)


class RelaxedCURFineTuner:

    def __init__(
        self,
        learning_rate: float = 1e-5,
        num_steps: int = 300,
        distill_weight: float = 0.5,
        warmup_steps: int = 30,
        gradient_accumulation_steps: int = 1,
    ):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.distill_weight = distill_weight
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def finetune(
        self,
        student_model: nn.Module,
        calib_data: List[Dict],
        device: str,
        cached_logits: Optional[List[torch.Tensor]] = None,
        teacher_model: Optional[nn.Module] = None,
    ) -> Dict:
        use_cached = cached_logits is not None

        if not use_cached and teacher_model is None:
            raise ValueError("Either cached_logits or teacher_model must be provided")

        if use_cached:
            print("    Using cached teacher logits (memory efficient mode)")
        else:
            print("    Using live teacher model (requires more GPU memory)")
            teacher_model.eval()

        optimizer_steps = self.num_steps // self.gradient_accumulation_steps

        if self.gradient_accumulation_steps > 1:
            print(f"    Gradient accumulation: {self.gradient_accumulation_steps} steps")
            print(f"    Total samples: {self.num_steps}, Optimizer updates: {optimizer_steps}")

        student_model.train()

        params_to_train = []
        for name, param in student_model.named_parameters():
            if any(x in name for x in ['C_eff', 'R_eff', '.U', 'gate_weight']):
                params_to_train.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False

        print(f"    Training {len(params_to_train)} parameter groups")

        optimizer = torch.optim.AdamW(params_to_train, lr=self.learning_rate)

        effective_warmup = max(1, self.warmup_steps // self.gradient_accumulation_steps)

        def lr_lambda(step):
            if step < effective_warmup:
                return step / max(1, effective_warmup)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        log = {'lm_loss': [], 'distill_loss': [], 'total_loss': []}

        n_data = len(calib_data)
        data_indices = list(range(n_data)) * ((self.num_steps // n_data) + 1)

        pbar = tqdm(range(optimizer_steps), desc="    Fine-tuning")

        optimizer.zero_grad()
        sample_idx = 0

        for step in pbar:
            accumulated_lm_loss = 0
            accumulated_distill_loss = 0
            accumulated_total_loss = 0

            for accum_step in range(self.gradient_accumulation_steps):
                if sample_idx >= self.num_steps:
                    break

                batch_idx = data_indices[sample_idx]
                batch = calib_data[batch_idx]
                input_ids = batch['input_ids'].to(device)

                if use_cached:
                    teacher_logits = cached_logits[batch_idx].to(device)
                else:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(input_ids, use_cache=False)
                        teacher_logits = teacher_outputs.logits

                student_outputs = student_model(input_ids, labels=input_ids, use_cache=False)
                lm_loss = student_outputs.loss
                student_logits = student_outputs.logits

                temperature = 2.0
                student_probs = F.log_softmax(student_logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
                distill_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
                distill_loss = distill_loss * (temperature ** 2)

                total_loss = (1 - self.distill_weight) * lm_loss + self.distill_weight * distill_loss
                scaled_loss = total_loss / self.gradient_accumulation_steps

                scaled_loss.backward()

                accumulated_lm_loss += lm_loss.item() / self.gradient_accumulation_steps
                accumulated_distill_loss += distill_loss.item() / self.gradient_accumulation_steps
                accumulated_total_loss += total_loss.item() / self.gradient_accumulation_steps

                del teacher_logits, student_outputs, student_logits
                del lm_loss, distill_loss, total_loss, scaled_loss

                sample_idx += 1

            torch.nn.utils.clip_grad_norm_(params_to_train, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            log['lm_loss'].append(accumulated_lm_loss)
            log['distill_loss'].append(accumulated_distill_loss)
            log['total_loss'].append(accumulated_total_loss)

            pbar.set_postfix({
                'lm': f"{accumulated_lm_loss:.3f}",
                'kd': f"{accumulated_distill_loss:.3f}",
            })

            if step % 50 == 0:
                torch.cuda.empty_cache()

        student_model.eval()
        return log


class InterpretabilityAnalyzer:

    @staticmethod
    def analyze_cosine_similarity(compressed_moe: RelaxedCURMixtralMoE) -> Dict:
        all_col_sims = []
        all_row_sims = []

        per_expert_stats = {}

        for exp_idx, expert in enumerate(compressed_moe.experts):
            expert_col_sims = []
            expert_row_sims = []

            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(expert, proj_name)
                expert_col_sims.extend(proj.col_cosine_sims)
                expert_row_sims.extend(proj.row_cosine_sims)

            all_col_sims.extend(expert_col_sims)
            all_row_sims.extend(expert_row_sims)

            per_expert_stats[exp_idx] = {
                'avg_col_cosine': np.mean(expert_col_sims),
                'avg_row_cosine': np.mean(expert_row_sims),
            }

        results = {
            'column_cosine_similarity': {
                'min': min(all_col_sims),
                'max': max(all_col_sims),
                'mean': np.mean(all_col_sims),
                'median': np.median(all_col_sims),
                'std': np.std(all_col_sims),
                'pct_above_0.9': sum(1 for s in all_col_sims if s > 0.9) / len(all_col_sims),
                'pct_above_0.8': sum(1 for s in all_col_sims if s > 0.8) / len(all_col_sims),
                'pct_above_0.7': sum(1 for s in all_col_sims if s > 0.7) / len(all_col_sims),
                'pct_above_0.5': sum(1 for s in all_col_sims if s > 0.5) / len(all_col_sims),
            },
            'row_cosine_similarity': {
                'min': min(all_row_sims),
                'max': max(all_row_sims),
                'mean': np.mean(all_row_sims),
                'median': np.median(all_row_sims),
                'std': np.std(all_row_sims),
            },
            'per_expert_stats': per_expert_stats,
            'total_columns': len(all_col_sims),
        }

        return results

    @staticmethod
    def analyze_offset_distribution(compressed_moe: RelaxedCURMixtralMoE) -> Dict:
        all_col_offsets = []
        all_row_offsets = []

        for expert in compressed_moe.experts:
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(expert, proj_name)
                all_col_offsets.extend(proj.col_offset_norms)
                all_row_offsets.extend(proj.row_offset_norms)

        thresholds = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
        effectively_exact = {}
        for thresh in thresholds:
            count = sum(1 for o in all_col_offsets if o < thresh)
            effectively_exact[f'<{int(thresh*100)}%'] = {
                'count': count,
                'total': len(all_col_offsets),
                'ratio': count / len(all_col_offsets),
            }

        return {
            'column_offset_stats': {
                'min': min(all_col_offsets),
                'max': max(all_col_offsets),
                'mean': np.mean(all_col_offsets),
                'median': np.median(all_col_offsets),
                'std': np.std(all_col_offsets),
            },
            'row_offset_stats': {
                'min': min(all_row_offsets),
                'max': max(all_row_offsets),
                'mean': np.mean(all_row_offsets),
                'median': np.median(all_row_offsets),
            },
            'effectively_exact_columns': effectively_exact,
            'total_columns': len(all_col_offsets),
        }

    @staticmethod
    def analyze_expert_specialization(
        compressed_moe: RelaxedCURMixtralMoE,
        top_k: int = 50,
    ) -> Dict:
        n_experts = len(compressed_moe.experts)

        expert_features = {}
        all_features = []

        for exp_idx, expert in enumerate(compressed_moe.experts):
            gate_indices = expert.gate_proj.col_indices[:top_k]
            expert_features[exp_idx] = {
                'gate_top_features': gate_indices,
                'up_top_features': expert.up_proj.col_indices[:top_k],
            }
            all_features.extend(gate_indices)

        feature_counts = Counter(all_features)

        shared_threshold = max(1, n_experts // 4)
        shared_features = [
            (feat, count) for feat, count in feature_counts.most_common()
            if count >= shared_threshold
        ]

        unique_features = {exp_idx: [] for exp_idx in range(n_experts)}
        for feat, count in feature_counts.items():
            if count <= 2:
                for exp_idx in range(n_experts):
                    if feat in expert_features[exp_idx]['gate_top_features']:
                        unique_features[exp_idx].append(feat)

        experts_with_unique = sum(
            1 for exp_idx in unique_features
            if len(unique_features[exp_idx]) > 0
        )

        return {
            'n_experts': n_experts,
            'total_unique_features': len(feature_counts),
            'shared_features': shared_features[:20],
            'unique_features': unique_features,
            'experts_with_unique': experts_with_unique,
            'feature_frequency': dict(feature_counts.most_common(30)),
        }

    @staticmethod
    def analyze_cross_layer_consistency(
        model: nn.Module,
        selected_layers: List[int],
        top_k: int = 30,
    ) -> Dict:
        layer_features = {}
        all_features_across_layers = []

        for layer_idx in selected_layers:
            layer = model.model.layers[layer_idx]
            if not hasattr(layer, 'block_sparse_moe'):
                continue

            compressed_moe = layer.block_sparse_moe
            layer_all_features = []

            for expert in compressed_moe.experts:
                layer_all_features.extend(expert.gate_proj.col_indices[:top_k])

            layer_features[layer_idx] = Counter(layer_all_features)
            all_features_across_layers.extend(layer_all_features)

        global_feature_counts = Counter(all_features_across_layers)

        n_layers = len(selected_layers)
        cross_layer_features = []

        for feat, count in global_feature_counts.most_common():
            layers_with_feat = sum(
                1 for layer_idx in layer_features
                if feat in layer_features[layer_idx]
            )
            if layers_with_feat >= n_layers // 2:
                cross_layer_features.append({
                    'feature': feat,
                    'total_count': count,
                    'layers_present': layers_with_feat,
                })

        return {
            'n_layers': n_layers,
            'cross_layer_features': cross_layer_features[:20],
            'total_unique_features': len(global_feature_counts),
            'features_in_all_layers': sum(
                1 for feat, count in global_feature_counts.items()
                if sum(1 for lf in layer_features.values() if feat in lf) == n_layers
            ),
        }


def get_calibration_loader(tokenizer, nsamples=64, seqlen=512, seed=42):
    data = load_dataset('allenai/c4', 'en', split='train', streaming=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    samples = []
    for i, example in enumerate(data):
        if len(samples) >= nsamples * 2:
            break
        if len(example['text']) > 100:
            samples.append(example['text'])

    encodings = []
    for text in samples:
        if len(encodings) >= nsamples:
            break
        tokens = tokenizer(
            text, truncation=True, max_length=seqlen,
            padding='max_length', return_tensors='pt'
        )
        if tokens['input_ids'].shape[1] == seqlen:
            encodings.append({
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask']
            })

    return encodings


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, max_samples=None):
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(data['text'])

    if max_samples:
        text = text[:max_samples * 100]

    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=50000)

    model.eval()
    nlls = []
    seq_len = 512
    device = next(model.parameters()).device

    for begin_loc in tqdm(range(0, min(encodings.input_ids.size(1), 8192), seq_len),
                          desc="Evaluating PPL", leave=False):
        end_loc = min(begin_loc + seq_len, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

        outputs = model(input_ids, labels=input_ids, use_cache=False)
        nlls.append(outputs.loss * (end_loc - begin_loc))

    ppl = torch.exp(torch.stack(nlls).sum() / min(encodings.input_ids.size(1), 8192))
    return ppl.item()


def compute_current_cosine_similarities(model: nn.Module, selected_layers: List[int]) -> Dict:
    all_col_sims = []
    all_row_sims = []
    layer_stats = {}

    for layer_idx in selected_layers:
        layer = model.model.layers[layer_idx]

        if not hasattr(layer, 'block_sparse_moe'):
            continue

        moe = layer.block_sparse_moe

        if not hasattr(moe.experts[0].gate_proj, 'compute_current_cosine_sims'):
            continue

        layer_col_sims = []
        layer_row_sims = []

        for expert in moe.experts:
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(expert, proj_name)
                col_sims, row_sims = proj.compute_current_cosine_sims()
                layer_col_sims.extend(col_sims)
                layer_row_sims.extend(row_sims)

        all_col_sims.extend(layer_col_sims)
        all_row_sims.extend(layer_row_sims)

        layer_stats[layer_idx] = {
            'avg_col_cosine': np.mean(layer_col_sims),
            'avg_row_cosine': np.mean(layer_row_sims),
            'min_col_cosine': min(layer_col_sims),
            'max_col_cosine': max(layer_col_sims),
        }

    if len(all_col_sims) == 0:
        return {
            'global': {'avg_col_cosine': 0, 'avg_row_cosine': 0},
            'layers': {},
            'n_columns': 0,
        }

    return {
        'global': {
            'avg_col_cosine': np.mean(all_col_sims),
            'avg_row_cosine': np.mean(all_row_sims),
            'std_col_cosine': np.std(all_col_sims),
            'min_col_cosine': min(all_col_sims),
            'max_col_cosine': max(all_col_sims),
            'pct_above_0.9': sum(1 for s in all_col_sims if s > 0.9) / len(all_col_sims),
            'pct_above_0.8': sum(1 for s in all_col_sims if s > 0.8) / len(all_col_sims),
            'pct_above_0.7': sum(1 for s in all_col_sims if s > 0.7) / len(all_col_sims),
        },
        'layers': layer_stats,
        'n_columns': len(all_col_sims),
    }


def print_cosine_similarity_comparison(
    before_stats: Dict,
    after_stats: Dict,
    label_before: str = "Before Fine-tuning",
    label_after: str = "After Fine-tuning",
) -> None:
    print(f"\n  Interpretability Comparison (Cosine Similarity):")
    print(f"  {'Metric':<25} {label_before:<20} {label_after:<20} {'Change':<15}")

    before_global = before_stats.get('global', {})
    after_global = after_stats.get('global', {})

    metrics = [
        ('Avg Column Cosine', 'avg_col_cosine'),
        ('Std Column Cosine', 'std_col_cosine'),
        ('Min Column Cosine', 'min_col_cosine'),
        ('Max Column Cosine', 'max_col_cosine'),
        ('% Above 0.9', 'pct_above_0.9'),
        ('% Above 0.8', 'pct_above_0.8'),
        ('% Above 0.7', 'pct_above_0.7'),
    ]

    for label, key in metrics:
        before_val = before_global.get(key, 0)
        after_val = after_global.get(key, 0)

        if 'pct' in key or '%' in label:
            before_str = f"{before_val*100:.1f}%"
            after_str = f"{after_val*100:.1f}%"
            change = (after_val - before_val) * 100
            change_str = f"{change:+.1f}%"
        else:
            before_str = f"{before_val:.4f}"
            after_str = f"{after_val:.4f}"
            change = after_val - before_val
            change_str = f"{change:+.4f}"

        print(f"  {label:<25} {before_str:<20} {after_str:<20} {change_str:<15}")

    before_avg = before_global.get('avg_col_cosine', 0)
    after_avg = after_global.get('avg_col_cosine', 0)
    drift = before_avg - after_avg

    print(f"\n  Interpretability Drift: {drift:.4f}")
    if drift < 0.01:
        print(f"  Excellent: Fine-tuning preserved interpretability almost perfectly")
    elif drift < 0.05:
        print(f"  Good: Fine-tuning caused minimal interpretability loss")
    elif drift < 0.10:
        print(f"  Moderate: Some interpretability was lost during fine-tuning")
    else:
        print(f"  Significant: Fine-tuning caused substantial interpretability loss")


def count_compression_storage(model, selected_layers: List[int]) -> Dict:
    non_moe_params = 0
    moe_original = 0
    moe_compressed = 0

    for name, param in model.named_parameters():
        is_compressed_moe = False
        for layer_idx in selected_layers:
            if f'layers.{layer_idx}.block_sparse_moe' in name:
                is_compressed_moe = True
                break

        if not is_compressed_moe:
            non_moe_params += param.numel()

    for layer_idx in selected_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'storage_elements'):
            moe_compressed += layer.block_sparse_moe.storage_elements()
            moe_original += layer.block_sparse_moe.original_elements()

    return {
        'non_moe_params': non_moe_params,
        'moe_original': moe_original,
        'moe_compressed': moe_compressed,
        'total_original': non_moe_params + moe_original,
        'total_compressed': non_moe_params + moe_compressed,
        'moe_saving': 1 - moe_compressed / moe_original if moe_original > 0 else 0,
        'total_saving': 1 - (non_moe_params + moe_compressed) / (non_moe_params + moe_original) if moe_original > 0 else 0,
    }


def compute_expected_compression(original_shape: Tuple[int, int], rank: int) -> Dict:
    m, n = original_shape
    k = rank

    original_elements = m * n
    compressed_elements = m * k + k * k + k * n

    return {
        'original_elements': original_elements,
        'compressed_elements': compressed_elements,
        'compression_ratio': compressed_elements / original_elements,
        'space_saving': 1 - compressed_elements / original_elements,
    }


def apply_relaxed_cur_compression(
    model: nn.Module,
    selected_layers: List[int],
    rank: int,
    cur_mode: str,
    n_iterations: int,
    lr: float,
    lambda_reg: float,
    device: str,
    verbose: bool = False,
    parallel: bool = True,
) -> Tuple[Dict, Dict]:

    config = model.config
    moe_config = get_mixtral_moe_config(config)

    opt_device = get_optimization_device()

    compressor = RelaxedCURMixtralCompressor(
        rank=rank,
        cur_mode=cur_mode,
        n_iterations=n_iterations,
        lr=lr,
        lambda_reg=lambda_reg,
        device=opt_device,
    )

    compression_log = {}
    total_original_elements = 0
    total_compressed_elements = 0

    layers = model.model.layers

    print(f"\nApplying Relaxed CUR compression to Mixtral...")
    print(f"  Rank: {rank}")
    print(f"  Optimization iterations: {n_iterations}")
    print(f"  Lambda (regularization): {lambda_reg}")
    print(f"  Optimization device: {opt_device}")
    print(f"  Parallel processing: {parallel}")

    for layer_idx in tqdm(selected_layers, desc="Compressing layers"):
        layer = layers[layer_idx]

        if not hasattr(layer, 'block_sparse_moe'):
            print(f"  Layer {layer_idx}: No MoE block found, skipping...")
            continue

        moe = layer.block_sparse_moe
        layer_device = next(layer.parameters()).device

        print(f"\n  Layer {layer_idx}:")

        compressed_moe, layer_stats = compressor.compress_moe_layer(
            moe, moe_config, device=layer_device, verbose=verbose, parallel=parallel
        )

        layer.block_sparse_moe = compressed_moe

        total_original_elements += layer_stats['summary']['original_elements']
        total_compressed_elements += layer_stats['summary']['compressed_elements']

        summary = layer_stats['summary']
        print(f"    Avg CUR error:     {summary['avg_initial_error']:.4f}")
        print(f"    Avg Relaxed error: {summary['avg_final_error']:.4f}")
        print(f"    Avg cosine sim:    {summary['avg_col_cosine_sim']:.4f}")
        print(f"    Compression:       {summary['space_saving']*100:.1f}% saved")

        compression_log[layer_idx] = layer_stats
        torch.cuda.empty_cache()

    storage_stats = {
        'compressed_layers_original': total_original_elements,
        'compressed_layers_new': total_compressed_elements,
        'compressed_layers_saving': 1 - total_compressed_elements / total_original_elements,
    }

    return compression_log, storage_stats


def main():
    parser = argparse.ArgumentParser(description='Relaxed CUR for Mixtral 8x7B Compression (GPU-Optimized)')

    parser.add_argument('--model', type=str, default='mistralai/Mixtral-8x7B-v0.1')
    parser.add_argument('--selected_layers', type=str, default='0,1,2,3,4,5,6,7')

    parser.add_argument('--rank', type=int, default=512)
    parser.add_argument('--cur_mode', type=str, default='deim')
    parser.add_argument('--n_iterations', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda_reg', type=float, default=0.1)

    parser.add_argument('--finetune_steps', type=int, default=300)
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--distill_weight', type=float, default=0.5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps (increase to reduce memory usage)')
    parser.add_argument('--skip_finetune', action='store_true')
    parser.add_argument('--logits_cache_dir', type=str, default=None,
                        help='Directory to cache teacher logits. If not specified, uses a temp directory.')
    parser.add_argument('--reuse_cached_logits', type=str, default=None,
                        help='Path to pre-computed teacher logits file to reuse.')

    parser.add_argument('--calib_nsamples', type=int, default=64)
    parser.add_argument('--calib_seqlen', type=int, default=512)

    parser.add_argument('--no_parallel', action='store_true',
                        help='Disable parallel expert compression')

    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--run_analysis', action='store_true',
                        help='Run comprehensive interpretability analysis')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.selected_layers == 'all':
        selected_layers = list(range(32))
    else:
        selected_layers = [int(x) for x in args.selected_layers.split(',')]

    print(f"\nModel: {args.model}")
    print(f"Selected layers: {selected_layers}")
    print(f"\nRelaxed CUR config:")
    print(f"  Rank: {args.rank}")
    print(f"  Optimization iterations: {args.n_iterations}")
    print(f"  Lambda (regularization): {args.lambda_reg}")
    print(f"\nGPU Optimization:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Parallel processing: {not args.no_parallel}")

    if not args.skip_finetune:
        print(f"\nCached Teacher Distillation:")
        print(f"  Fine-tune steps: {args.finetune_steps}")
        print(f"  Fine-tune LR: {args.finetune_lr}")
        print(f"  Distill weight: {args.distill_weight}")
        if args.gradient_accumulation_steps > 1:
            optimizer_steps = args.finetune_steps // args.gradient_accumulation_steps
            print(f"  Gradient accumulation: {args.gradient_accumulation_steps} steps")
            print(f"    -> {args.finetune_steps} samples seen, {optimizer_steps} optimizer updates")
            print(f"    -> Effective batch size: {args.gradient_accumulation_steps}x")
        if args.reuse_cached_logits:
            print(f"  Reusing cached logits from: {args.reuse_cached_logits}")
        else:
            cache_dir = args.logits_cache_dir or args.save_path or "temp_logits_cache"
            print(f"  Logits cache dir: {cache_dir}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    baseline_params = sum(p.numel() for p in model.parameters())
    print(f"Baseline parameters: {baseline_params:,}")

    if args.run_analysis:
        print("Creating original model copy for analysis...")
        original_model = copy.deepcopy(model)

    print("\nLoading calibration data...")
    calib_data = get_calibration_loader(
        tokenizer, args.calib_nsamples, args.calib_seqlen, args.seed
    )

    cached_logits = None
    cache_file = None
    temp_cache_dir = None

    if not args.skip_finetune:
        if args.reuse_cached_logits:
            print(f"\nLoading cached teacher logits from: {args.reuse_cached_logits}")
            cached_logits = load_cached_logits(args.reuse_cached_logits)
            print(f"  Loaded {len(cached_logits)} cached logit batches")
        else:
            print("\n" + "=" * 60)
            print("CACHING TEACHER LOGITS (for memory-efficient fine-tuning)")
            print("=" * 60)

            if args.logits_cache_dir:
                cache_dir = args.logits_cache_dir
            elif args.save_path:
                cache_dir = os.path.join(args.save_path, 'teacher_logits_cache')
            else:
                temp_cache_dir = tempfile.mkdtemp(prefix='relaxed_cur_mixtral_logits_')
                cache_dir = temp_cache_dir

            cache_file = cache_teacher_logits(
                teacher_model=model,
                calib_data=calib_data,
                cache_dir=cache_dir,
                device=args.device,
            )

            cached_logits = load_cached_logits(cache_file)
            print(f"  Teacher logits cached and loaded successfully")

            torch.cuda.empty_cache()

    if args.evaluate:
        print("\nEvaluating baseline...")
        baseline_ppl = evaluate_perplexity(model, tokenizer, max_samples=100)
        print(f"Baseline PPL: {baseline_ppl:.2f}")

    print("STAGE 1: Relaxed CUR Compression")

    compression_log, storage_stats = apply_relaxed_cur_compression(
        model=model,
        selected_layers=selected_layers,
        rank=args.rank,
        cur_mode=args.cur_mode,
        n_iterations=args.n_iterations,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        device=args.device,
        verbose=args.verbose,
        parallel=not args.no_parallel,
    )

    print("\nRestoring model to GPU...")
    if hasattr(model, 'hf_device_map'):
        delattr(model, 'hf_device_map')
    model = model.to(args.device)
    torch.cuda.empty_cache()

    compressed_storage = count_compression_storage(model, selected_layers)

    if not args.skip_finetune:
        print("STAGE 2: Knowledge Distillation Fine-tuning (Using Cached Logits)")
        print("  Note: Teacher model NOT loaded - using cached logits to save GPU memory")

        torch.cuda.empty_cache()
        gc.collect()

        print("\n  Computing cosine similarities before fine-tuning...")
        cosine_before = compute_current_cosine_similarities(model, selected_layers)
        print(f"  Avg cosine similarity (before): {cosine_before['global']['avg_col_cosine']:.4f}")

        param_snapshots_before = {}
        for name, param in model.named_parameters():
            if any(x in name for x in ['C_eff', 'R_eff', '.U']):
                param_snapshots_before[name] = param.data.clone()

        finetuner = RelaxedCURFineTuner(
            learning_rate=args.finetune_lr,
            num_steps=args.finetune_steps,
            distill_weight=args.distill_weight,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        ft_log = finetuner.finetune(
            student_model=model,
            calib_data=calib_data,
            device=args.device,
            cached_logits=cached_logits,
            teacher_model=None,
        )

        print("\n  Computing cosine similarities after fine-tuning...")
        cosine_after = compute_current_cosine_similarities(model, selected_layers)
        print(f"  Avg cosine similarity (after): {cosine_after['global']['avg_col_cosine']:.4f}")

        print("\n  Parameter Change Verification:")
        c_eff_changes = []
        r_eff_changes = []
        u_changes = []

        for name, param in model.named_parameters():
            if name in param_snapshots_before:
                before = param_snapshots_before[name]
                after = param.data

                diff_norm = torch.norm(after - before).item()
                before_norm = torch.norm(before).item()
                relative_change = diff_norm / (before_norm + 1e-10)

                if 'C_eff' in name:
                    c_eff_changes.append(relative_change)
                elif 'R_eff' in name:
                    r_eff_changes.append(relative_change)
                elif '.U' in name:
                    u_changes.append(relative_change)

        if c_eff_changes:
            print(f"    C_eff relative change: {np.mean(c_eff_changes):.6f} (avg), {max(c_eff_changes):.6f} (max)")
        if r_eff_changes:
            print(f"    R_eff relative change: {np.mean(r_eff_changes):.6f} (avg), {max(r_eff_changes):.6f} (max)")
        if u_changes:
            print(f"    U relative change:     {np.mean(u_changes):.6f} (avg), {max(u_changes):.6f} (max)")

        del param_snapshots_before

        print_cosine_similarity_comparison(cosine_before, cosine_after)

        interpretability_stats = {
            'before_finetuning': cosine_before,
            'after_finetuning': cosine_after,
            'drift': cosine_before['global']['avg_col_cosine'] - cosine_after['global']['avg_col_cosine'],
        }

        del cached_logits
        torch.cuda.empty_cache()
    else:
        interpretability_stats = None

    if temp_cache_dir and os.path.exists(temp_cache_dir):
        print(f"\nCleaning up temporary cache directory: {temp_cache_dir}")
        shutil.rmtree(temp_cache_dir)

    print("FINAL SUMMARY")


    print(f"\nStorage (REAL compression - excludes analysis buffers):")
    print(f"  Compressed MoE layers:")
    print(f"    Original:   {compressed_storage['moe_original']:,} elements")
    print(f"    Compressed: {compressed_storage['moe_compressed']:,} elements")
    print(f"    Saving:     {compressed_storage['moe_saving']*100:.1f}%")
    print(f"  Total model:")
    print(f"    Original:   {compressed_storage['total_original']:,} elements")
    print(f"    Compressed: {compressed_storage['total_compressed']:,} elements")
    print(f"    Saving:     {compressed_storage['total_saving']*100:.2f}%")

    if args.evaluate:
        print(f"\nPerplexity:")
        print(f"  Baseline:   {baseline_ppl:.2f}")

    if interpretability_stats is not None:
        print(f"\nInterpretability (Cosine Similarity with Original Columns):")
        before_avg = interpretability_stats['before_finetuning']['global']['avg_col_cosine']
        after_avg = interpretability_stats['after_finetuning']['global']['avg_col_cosine']
        drift = interpretability_stats['drift']
        print(f"  Before fine-tuning: {before_avg:.4f}")
        print(f"  After fine-tuning:  {after_avg:.4f}")
        print(f"  Drift:              {drift:.4f} ({'minimal' if drift < 0.05 else 'moderate' if drift < 0.1 else 'significant'})")

    if args.run_analysis:
        print("INTERPRETABILITY ANALYSIS")

        for layer_idx in selected_layers:
            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'block_sparse_moe'):
                compressed_moe = layer.block_sparse_moe
                print(f"\nLayer {layer_idx}:")

                cosine_results = InterpretabilityAnalyzer.analyze_cosine_similarity(compressed_moe)
                print(f"  Cosine Similarity:")
                stats = cosine_results['column_cosine_similarity']
                print(f"    Mean: {stats['mean']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
                print(f"    >0.9: {stats['pct_above_0.9']*100:.1f}%, >0.8: {stats['pct_above_0.8']*100:.1f}%")

                offset_results = InterpretabilityAnalyzer.analyze_offset_distribution(compressed_moe)
                print(f"  Offset Distribution:")
                stats = offset_results['column_offset_stats']
                print(f"    Mean: {stats['mean']*100:.1f}%, Median: {stats['median']*100:.1f}%")

                spec_results = InterpretabilityAnalyzer.analyze_expert_specialization(compressed_moe)
                print(f"  Expert Specialization:")
                print(f"    Unique features: {spec_results['total_unique_features']}")
                print(f"    Experts with unique features: {spec_results['experts_with_unique']}/{spec_results['n_experts']}")

        del original_model
        torch.cuda.empty_cache()

    if args.save_path:
        print(f"\nSaving to {args.save_path}...")
        os.makedirs(args.save_path, exist_ok=True)
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)

        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        full_log = {
            'compression': convert(compression_log),
            'storage': convert(compressed_storage),
        }

        if args.evaluate:
            full_log['perplexity'] = {
                'baseline': baseline_ppl,
            }

        if interpretability_stats is not None:
            full_log['interpretability'] = {
                'before_finetuning': {
                    'avg_col_cosine': interpretability_stats['before_finetuning']['global']['avg_col_cosine'],
                    'std_col_cosine': interpretability_stats['before_finetuning']['global'].get('std_col_cosine', 0),
                    'pct_above_0.9': interpretability_stats['before_finetuning']['global'].get('pct_above_0.9', 0),
                    'pct_above_0.8': interpretability_stats['before_finetuning']['global'].get('pct_above_0.8', 0),
                    'pct_above_0.7': interpretability_stats['before_finetuning']['global'].get('pct_above_0.7', 0),
                },
                'after_finetuning': {
                    'avg_col_cosine': interpretability_stats['after_finetuning']['global']['avg_col_cosine'],
                    'std_col_cosine': interpretability_stats['after_finetuning']['global'].get('std_col_cosine', 0),
                    'pct_above_0.9': interpretability_stats['after_finetuning']['global'].get('pct_above_0.9', 0),
                    'pct_above_0.8': interpretability_stats['after_finetuning']['global'].get('pct_above_0.8', 0),
                    'pct_above_0.7': interpretability_stats['after_finetuning']['global'].get('pct_above_0.7', 0),
                },
                'drift': interpretability_stats['drift'],
            }

        with open(os.path.join(args.save_path, 'compression_log.json'), 'w') as f:
            json.dump(convert(full_log), f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()