import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .decomposition import relaxed_cur_decompose
from .modules import RelaxedCURProjection, RelaxedCURExpert, RelaxedCURMoE


class RelaxedCURCompressor:
    def __init__(
        self,
        rank: int = 512,
        cur_mode: str = 'deim',
        n_iterations: int = 500,
        lr: float = 0.01,
        lambda_reg: float = 0.1,
    ):
        self.rank = rank
        self.cur_mode = cur_mode
        self.n_iterations = n_iterations
        self.lr = lr
        self.lambda_reg = lambda_reg
    
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
        )
        
        projection = RelaxedCURProjection(
            C_eff=result.C_eff,
            U=result.U,
            R_eff=result.R_eff,
            col_indices=result.col_indices,
            row_indices=result.row_indices,
            col_offset_norms=result.col_offset_norms,
            row_offset_norms=result.row_offset_norms,
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
        }
        
        return projection, stats
    
    def compress_expert(
        self,
        expert: nn.Module,
        verbose: bool = False,
    ) -> Tuple[RelaxedCURExpert, Dict]:
        compressed = RelaxedCURExpert(hidden_act='silu')
        expert_stats = {
            'projections': {},
            'total_original': 0,
            'total_compressed': 0,
        }
        
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            module = getattr(expert, proj_name)
            W = module.weight.data
            
            if verbose:
                print(f"      {proj_name}: shape={tuple(W.shape)}")
            
            projection, stats = self.compress_projection(W, verbose=verbose)
            compressed.set_projection(proj_name, projection)
            
            expert_stats['projections'][proj_name] = stats
            expert_stats['total_original'] += stats['original_elements']
            expert_stats['total_compressed'] += stats['compressed_elements']
            
            if verbose:
                print(f"        CUR error: {stats['initial_error']:.4f} → "
                      f"Relaxed: {stats['final_error']:.4f} "
                      f"(↓{stats['error_reduction']:.4f})")
                print(f"        Compression: {stats['original_elements']:,} → "
                      f"{stats['compressed_elements']:,} "
                      f"({stats['space_saving']*100:.1f}% saved)")
        
        expert_stats['compression_ratio'] = expert_stats['total_compressed'] / expert_stats['total_original']
        expert_stats['space_saving'] = 1 - expert_stats['compression_ratio']
        
        return compressed, expert_stats
    
    def compress_moe_layer(
        self,
        moe: nn.Module,
        moe_config: dict,
        device: str = 'cpu',
        verbose: bool = False,
    ) -> Tuple[RelaxedCURMoE, Dict]:
        n_experts = len(moe.experts)
        compressed_experts = nn.ModuleList()
        
        original_dtype = next(moe.parameters()).dtype
        
        layer_stats = {
            'n_experts': n_experts,
            'experts': {},
            'summary': {
                'initial_errors': [],
                'final_errors': [],
                'error_reductions': [],
                'original_elements': 0,
                'compressed_elements': 0,
            }
        }
        
        for exp_idx in tqdm(range(n_experts), desc="    Compressing experts", leave=False):
            expert = moe.experts[exp_idx]
            
            if verbose:
                print(f"    Expert {exp_idx}:")
            
            compressed_expert, stats = self.compress_expert(
                expert, verbose=verbose
            )
            
            compressed_expert = compressed_expert.to(device=device, dtype=original_dtype)
            
            compressed_experts.append(compressed_expert)
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
            
            layer_stats['summary']['original_elements'] += stats['total_original']
            layer_stats['summary']['compressed_elements'] += stats['total_compressed']
        
        layer_stats['summary']['avg_initial_error'] = np.mean(layer_stats['summary']['initial_errors'])
        layer_stats['summary']['avg_final_error'] = np.mean(layer_stats['summary']['final_errors'])
        layer_stats['summary']['avg_error_reduction'] = np.mean(layer_stats['summary']['error_reductions'])
        layer_stats['summary']['compression_ratio'] = (
            layer_stats['summary']['compressed_elements'] / 
            layer_stats['summary']['original_elements']
        )
        layer_stats['summary']['space_saving'] = 1 - layer_stats['summary']['compression_ratio']
        
        shared_experts = None
        if hasattr(moe, 'shared_experts') and moe.shared_experts is not None:
            shared_experts = copy.deepcopy(moe.shared_experts)
        
        compressed_moe = RelaxedCURMoE(
            config=moe_config,
            experts=compressed_experts,
            gate_weight=moe.gate.weight.data.clone(),
            shared_experts=shared_experts,
        )
        
        compressed_moe = compressed_moe.to(device=device, dtype=original_dtype)
        
        return compressed_moe, layer_stats
