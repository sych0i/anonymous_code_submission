import torch

def compute_ess(w, dim=-1):
    ess = (w.sum(dim=dim))**2 / torch.sum(w**2, dim=dim)
    return ess

def compute_ess_from_log_w(log_w, dim=-1):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)

def normalize_weights(log_weights, dim=-1):
    return torch.exp(normalize_log_weights(log_weights, dim=dim))

def normalize_log_weights(log_weights, dim=-1):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - torch.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights

def lambda_schedule(num_timesteps: int, base: float = 2.0, gamma=None):
    """
    Exponential decay from 1 → 0 over num_timesteps:
      - λ(0) = 1
      - λ(T) = 0
    The curve is controlled by `base`:
      - base=2 is the original half-life style
      - larger base → steeper early drop
    """
    if gamma is not None:
        return [
            min((1 + gamma) ** (num_timesteps - t) - 1.0, 1.0)
            for t in range(num_timesteps + 1)
        ]
    
    # Solve (1+γ)^T = base  ⇒  γ = base^(1/T) - 1
    gamma = base ** (1 / num_timesteps) - 1.0

    # (1+γ)^(T - t) - 1 runs from base-1 → 0; divide by (base-1) to normalize to [1→0]
    return [
        ((1 + gamma) ** (num_timesteps - t) - 1.0) / (base - 1.0)
        for t in range(num_timesteps + 1)
    ]
