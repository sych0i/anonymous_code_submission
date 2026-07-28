import torch
from torch import Tensor
from typing import Tuple, Callable
from .utils import normalize_weights

def stratified_resample(log_weights: Tensor):
    N = log_weights.shape[0]
    weights = normalize_weights(log_weights)
    cdf = torch.cumsum(weights, dim=0)

    # Stratified uniform samples
    u = (torch.arange(N, dtype=torch.float32, device=log_weights.device) + torch.rand(N, device=log_weights.device)) / N

    indices = torch.searchsorted(cdf, u, right=True)
    return indices

def systematic_resample(log_weights: Tensor, normalized=True):
    N = log_weights.shape[0]
    weights = normalize_weights(log_weights)
    cdf = torch.cumsum(weights, dim=0)

    # Systematic uniform samples
    u0 = torch.rand(1, device=log_weights.device) / N
    u = u0 + torch.arange(N, dtype=torch.float32, device=log_weights.device) / N

    indices = torch.searchsorted(cdf, u, right=True)
    return indices

def multinomial_resample(log_weights: Tensor, normalized=True):
    N = log_weights.shape[0]
    weights = normalize_weights(log_weights)
    resampled_indices = torch.multinomial(weights, N, replacement=True)
    return resampled_indices

def partial_resample(log_weights: torch.Tensor,
                     resample_fn: Callable[[Tensor], Tensor],
                     M: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform partial resampling on a set of particles using PyTorch.

    Args:
        log_weights (torch.Tensor): 1D tensor of shape (K,) containing log-weights.
        resample_fn (callable): function that takes log_weights and n_samples,
                                returning a tensor of shape (n_samples,) of sampled indices.
        M (int): total number of particles to resample.

    Returns:
        new_indices (torch.Tensor): 1D tensor of shape (K,) mapping each output slot to
                                    an original particle index.
        new_log_weights (torch.Tensor): 1D tensor of shape (K,) of updated log-weights.
    """
    K = log_weights.numel()

    # Convert log-weights to normalized weights
    weights = torch.softmax(log_weights, dim=0)

    # Determine how many high and low weights to resample
    M_hi = M // 2
    M_lo = M - M_hi

    # Get indices of highest and lowest weights
    _, hi_idx = torch.topk(weights, M_hi, largest=True)
    _, lo_idx = torch.topk(weights, M_lo, largest=False)
    I = torch.cat([hi_idx, lo_idx])  # indices selected for resampling

    # Perform multinomial resampling only on selected subset
    # resample_fn expects log-weights of the subset
    subset_logw = log_weights[I]
    local_sampled = resample_fn(subset_logw)  # indices in [0, len(I))
    # Map back to original indices
    sampled = I[local_sampled]

    # Build new index mapping: default to identity (retain original)
    new_indices = torch.arange(K, device=log_weights.device)
    new_indices[I] = sampled

    # Compute new uniform weight for resampled particles
    total_I_weight = weights[I].sum()
    uniform_weight = total_I_weight / M

    # Prepare new log-weights
    new_log_weight = torch.empty_like(log_weights)
    # For non-resampled, keep original log-weights
    mask = torch.ones(K, dtype=torch.bool, device=log_weights.device)
    mask[I] = False
    new_log_weight[mask] = log_weights[mask]
    # For resampled, assign uniform log-weight
    new_log_weight[I] = torch.log(uniform_weight)

    return new_indices, new_log_weight
