from dataclasses import dataclass
from typing import List

import torch
import numpy as np

from cur import select_rows_and_columns


@dataclass
class RelaxedCURResult:
    C_eff: torch.Tensor
    U: torch.Tensor
    R_eff: torch.Tensor
    col_indices: List[int]
    row_indices: List[int]
    col_offset_norms: List[float]
    row_offset_norms: List[float]
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
) -> RelaxedCURResult:
    m, n = W.shape
    k = len(col_indices)
    
    W_f = W.float().cpu()
    device = W_f.device
    
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
    
    optimizer = torch.optim.Adam([delta_C, delta_R, U], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)
    
    best_error = float('inf')
    best_state = None
    
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
        
        current_error = (torch.sqrt(recon_loss) / W_norm).item()
        if current_error < best_error:
            best_error = current_error
            best_state = {
                'delta_C': delta_C.detach().clone(),
                'delta_R': delta_R.detach().clone(),
                'U': U.detach().clone(),
            }
        
        if verbose and (iteration + 1) % 100 == 0:
            offset_norm = (torch.norm(delta_C) + torch.norm(delta_R)).item()
            print(f"      Iter {iteration+1}: error={current_error:.4f}, "
                  f"offset_norm={offset_norm:.4f}")
    
    delta_C_final = best_state['delta_C']
    delta_R_final = best_state['delta_R']
    U_final = best_state['U']
    
    col_base_norms = torch.norm(C_base, dim=0) + 1e-10
    row_base_norms = torch.norm(R_base, dim=1) + 1e-10
    col_offset_norms = (torch.norm(delta_C_final, dim=0) / col_base_norms).tolist()
    row_offset_norms = (torch.norm(delta_R_final, dim=1) / row_base_norms).tolist()
    
    C_eff_final = C_base + delta_C_final
    R_eff_final = R_base + delta_R_final
    
    W_approx_final = C_eff_final @ U_final @ R_eff_final
    final_error = (torch.norm(W_f - W_approx_final) / W_norm).item()
    
    return RelaxedCURResult(
        C_eff=C_eff_final,
        U=U_final,
        R_eff=R_eff_final,
        col_indices=col_indices,
        row_indices=row_indices,
        col_offset_norms=col_offset_norms,
        row_offset_norms=row_offset_norms,
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
    )
    
    return result
