import math
import torch
from torch import linalg as LA
from typing import Tuple, Optional, List


@torch.no_grad()
def _matrix_sqrt_psd(Sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    Sigma = 0.5 * (Sigma + Sigma.t())
    d = Sigma.shape[0]
    Sigma = Sigma + eps * torch.eye(d, dtype=Sigma.dtype, device=Sigma.device)
    evals, evecs = torch.linalg.eigh(Sigma)
    evals = torch.clamp(evals, min=0.0).sqrt()
    return (evecs * evals.unsqueeze(0)) @ evecs.t()


@torch.no_grad()
def cur_deim_gpu(
    W: torch.Tensor,
    r: int,
    use_lowrank: bool = True,
    oversample: int = 20,
    niter: int = 2
) -> Tuple[List[int], List[int]]:
    if use_lowrank:
        qmax = min(W.shape[0], W.shape[1])
        q = min(r + oversample, qmax)
        U, S, V = torch.svd_lowrank(W.float(), q=q, niter=niter)
        U, V = U[:, :r], V[:, :r]
    else:
        U_full, S, Vh = LA.svd(W.float(), full_matrices=False)
        U, V = U_full[:, :r], Vh.T[:, :r]

    m, n = W.shape
    irow = torch.empty(r, dtype=torch.long, device=W.device)
    icol = torch.empty(r, dtype=torch.long, device=W.device)
    mask_r = torch.zeros(m, dtype=torch.bool, device=W.device)
    mask_c = torch.zeros(n, dtype=torch.bool, device=W.device)

    for k in range(r):
        u_vec = torch.where(mask_r, torch.zeros_like(U[:, k]), U[:, k].abs())
        v_vec = torch.where(mask_c, torch.zeros_like(V[:, k]), V[:, k].abs())
        
        row_k = torch.argmax(u_vec)
        col_k = torch.argmax(v_vec)

        irow[k] = row_k
        icol[k] = col_k
        mask_r[row_k] = True
        mask_c[col_k] = True

        if k + 1 < r:
            alpha_r = U[row_k, :k+1]
            alpha_c = V[col_k, :k+1]

            denom_r = (alpha_r @ alpha_r).clamp_min(1e-12)
            denom_c = (alpha_c @ alpha_c).clamp_min(1e-12)
            
            U[:, k+1:] -= (U[:, :k+1] @ alpha_r.unsqueeze(1)) / denom_r
            V[:, k+1:] -= (V[:, :k+1] @ alpha_c.unsqueeze(1)) / denom_c

    return irow.tolist(), icol.tolist()


def select_rows_and_columns(
    W: torch.Tensor,
    A: Optional[torch.Tensor],
    num_rows: int,
    num_cols: int,
    aux_mode: str = 'wanda',
    cur_mode: str = 'deim',
) -> Tuple[List[int], List[int]]:
    m, n = W.shape
    r = min(num_rows, num_cols, m, n)

    if cur_mode == 'random':
        k_rows = min(num_rows, m)
        k_cols = min(num_cols, n)
        row_indices = torch.randperm(m, device=W.device)[:k_rows].tolist()
        col_indices = torch.randperm(n, device=W.device)[:k_cols].tolist()
        return row_indices, col_indices

    if aux_mode == 'wanda':
        if A is not None:
            act = A.view(1, -1).to(W.device, dtype=W.dtype)
            S = W.abs() * act
        else:
            S = W.abs()
    
    elif aux_mode == 'weight':
        S = W.abs()
    
    elif aux_mode == 'cov_fast':
        if A is not None:
            scale = A.view(1, -1).to(W.device, dtype=W.dtype)
            S = W * scale
        else:
            S = W.abs()
    
    elif aux_mode == 'cov':
        if A is not None and A.dim() == 2:
            Sigma = A.to(W.device, dtype=W.dtype)
            D = _matrix_sqrt_psd(Sigma)
            S = W @ D
        else:
            S = W.abs()
    else:
        S = W.abs()

    if cur_mode == 'deim':
        row_indices, col_indices = cur_deim_gpu(S, r, use_lowrank=True)
        return row_indices, col_indices
    
    elif cur_mode == 'deim_full':
        row_indices, col_indices = cur_deim_gpu(S, r, use_lowrank=False)
        return row_indices, col_indices
    
    elif cur_mode == 'magnitude':
        frobenius_norm = torch.norm(S, p='fro')
        col_norms = torch.norm(S, p=2, dim=0)
        row_norms = torch.norm(S, p=2, dim=1)

        k_cols = min(num_cols, col_norms.numel())
        k_rows = min(num_rows, row_norms.numel())

        cond_bad = (
            (not torch.isfinite(frobenius_norm)) or (frobenius_norm <= 0) or
            (not torch.isfinite(col_norms).all()) or (not torch.isfinite(row_norms).all()) or
            (col_norms.sum() <= 0) or (row_norms.sum() <= 0)
        )
        if cond_bad:
            col_indices = torch.topk(col_norms, k_cols, largest=True).indices
            row_indices = torch.topk(row_norms, k_rows, largest=True).indices
            return row_indices.tolist(), col_indices.tolist()

        col_prob = (col_norms / col_norms.sum()).clamp_min(0)
        row_prob = (row_norms / row_norms.sum()).clamp_min(0)

        col_indices = torch.multinomial(col_prob, num_samples=k_cols, replacement=False)
        row_indices = torch.multinomial(row_prob, num_samples=k_rows, replacement=False)
        return row_indices.tolist(), col_indices.tolist()
    
    else:
        raise ValueError(f"Unknown cur_mode: {cur_mode}")


def cur_decomposition(
    W: torch.Tensor,
    A: Optional[torch.Tensor],
    num_rows: int,
    num_cols: int,
    aux_mode: str = 'wanda',
    cur_mode: str = 'deim',
    use_float64: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    orig_dtype = W.dtype
    if use_float64 and orig_dtype != torch.float64:
        W = W.to(torch.float64)
        if A is not None:
            A = A.to(torch.float64)

    row_indices, col_indices = select_rows_and_columns(
        W, A, num_rows, num_cols,
        aux_mode=aux_mode, cur_mode=cur_mode
    )

    C = W[:, col_indices]
    R = W[row_indices, :]

    rc = 1e-12 if W.dtype == torch.float64 else 1e-6

    if aux_mode == 'wanda' or aux_mode == 'weight' or A is None:
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ W
            @ torch.linalg.pinv(R, rcond=rc)
        )

    elif aux_mode == 'cov_fast':
        if A is not None and A.dim() == 1 and A.numel() == W.shape[1]:
            Dvec = A.view(1, -1).to(W.device, dtype=W.dtype)
            WD = W * Dvec
            RD = R * Dvec
            U = (
                torch.linalg.pinv(C, rcond=rc)
                @ WD
                @ torch.linalg.pinv(RD, rcond=rc)
            )
        else:
            U = torch.linalg.pinv(C, rcond=rc) @ W @ torch.linalg.pinv(R, rcond=rc)

    elif aux_mode == 'cov':
        if A is not None and A.dim() == 2 and A.shape[0] == W.shape[1]:
            Sigma = A.to(W.device, dtype=W.dtype)
            D = _matrix_sqrt_psd(Sigma)
            WD = W @ D
            RD = R @ D
            U = (
                torch.linalg.pinv(C, rcond=rc)
                @ WD
                @ torch.linalg.pinv(RD, rcond=rc)
            )
        else:
            U = torch.linalg.pinv(C, rcond=rc) @ W @ torch.linalg.pinv(R, rcond=rc)

    else:
        U = torch.linalg.pinv(C, rcond=rc) @ W @ torch.linalg.pinv(R, rcond=rc)

    if use_float64 and orig_dtype != torch.float64:
        C = C.to(orig_dtype)
        R = R.to(orig_dtype)
        U = U.to(orig_dtype)

    return C, U, R, row_indices, col_indices


@torch.no_grad()
def energy_rank(
    W: torch.Tensor,
    A: Optional[torch.Tensor],
    aux_mode: str,
    energy: float = 0.98,
    use_lowrank: bool = True,
    niter: int = 2,
    round_to_128: bool = True,
) -> int:
    m, n = W.shape

    if aux_mode == 'wanda':
        if A is not None and A.dim() == 1 and A.numel() == n:
            act = A.view(1, -1).to(W.device, dtype=W.dtype)
            M = W.abs() * act
        else:
            M = W.abs()
    
    elif aux_mode == 'weight':
        M = W.abs()
    
    elif aux_mode == 'cov_fast':
        if A is not None and A.dim() == 1 and A.numel() == n:
            scale = A.view(1, -1).to(W.device, dtype=W.dtype)
            M = W * scale
        else:
            M = W.abs()
    
    elif aux_mode == 'cov':
        if A is not None and A.dim() == 2 and A.shape[0] == n:
            Sigma = A.to(W.device, dtype=W.dtype)
            D = _matrix_sqrt_psd(Sigma)
            M = W @ D
        else:
            M = W.abs()
    
    else:
        M = W.abs()

    if use_lowrank:
        q = min(
            max(256, int(min(m, n) * 0.25)),
            min(m, n)
        )
        _, sv, _ = torch.svd_lowrank(M.float(), q=q, niter=niter)
    else:
        _, sv, _ = torch.linalg.svd(M.float(), full_matrices=False)

    if sv.numel() == 0:
        r = 1
    else:
        e = sv.square()
        total = e.sum()
        if total <= 0 or not torch.isfinite(total):
            r = 1
        else:
            cume = torch.cumsum(e, dim=0) / total
            target = max(1e-6, min(float(energy), 0.999999))
            r = int(torch.searchsorted(cume, torch.tensor(target, device=cume.device)).item()) + 1

    if round_to_128 and r > 0:
        r = ((r + 127) // 128) * 128

    r = max(1, min(r, min(m, n)))
    return r


def calculate_rank(m: int, n: int, round_to_128: bool = True) -> int:
    try:
        r = int((math.sqrt(m**2 + 6 * m * n + n**2) - (m + n)) / 2)
    except ValueError:
        r = min(m, n)

    if round_to_128 and r > 0:
        r = (r // 128) * 128

    r = max(1, min(r, min(m, n)))
    return r


def compute_cur_error(W: torch.Tensor, C: torch.Tensor, U: torch.Tensor, R: torch.Tensor) -> float:
    W_approx = C.float() @ U.float() @ R.float()
    error = torch.norm(W.float() - W_approx, p='fro') / torch.norm(W.float(), p='fro')
    return error.item()


def cur_decomposition_with_energy(
    W: torch.Tensor,
    energy: float = 0.9,
    min_rank: int = 32,
    max_rank: Optional[int] = None,
    aux_info: Optional[torch.Tensor] = None,
    aux_mode: str = 'wanda',
    cur_mode: str = 'deim',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int], int]:
    rank = energy_rank(W, aux_info, aux_mode=aux_mode, energy=energy, use_lowrank=True)
    
    rank = max(rank, min_rank)
    if max_rank is not None:
        rank = min(rank, max_rank)
    
    m, n = W.shape
    upper_bound = calculate_rank(m, n, round_to_128=False)
    if rank > upper_bound:
        rank = upper_bound
    
    C, U, R, row_indices, col_indices = cur_decomposition(
        W, aux_info, num_rows=rank, num_cols=rank,
        aux_mode=aux_mode, cur_mode=cur_mode, use_float64=True
    )
    
    return C, U, R, row_indices, col_indices, rank


def apply_cur_to_matrix(
    weight: torch.Tensor,
    aux_info: Optional[torch.Tensor],
    max_rank: Optional[int] = None,
    min_rank: Optional[int] = None,
    aux_mode: str = 'wanda',
    cur_mode: str = 'deim',
    energy: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, List[int], List[int]]:
    m, n = weight.shape
    
    if energy is not None:
        rank = energy_rank(
            weight, aux_info, aux_mode=aux_mode,
            energy=energy, use_lowrank=True
        )
        upper_bound_rank = calculate_rank(m, n)
        if rank > upper_bound_rank:
            raise ValueError(f"Energy {energy} requires rank {rank} > breakeven {upper_bound_rank}. No compression benefit.")
    else:
        rank = calculate_rank(m, n)
    
    if max_rank:
        rank = min(rank, int(max_rank))
    if min_rank:
        rank = max(rank, int(min_rank))
    
    C, U, R, row_indices, col_indices = cur_decomposition(
        weight, aux_info, num_rows=rank, num_cols=rank,
        aux_mode=aux_mode, cur_mode=cur_mode, use_float64=True
    )
    
    return C, U, R, rank, row_indices, col_indices
