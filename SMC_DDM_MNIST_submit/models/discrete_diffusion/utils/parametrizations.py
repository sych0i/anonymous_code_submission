import torch
from torch import Tensor
import torch.nn.functional as F


def subs_parametrization(logits: Tensor, x: Tensor) -> Tensor:
    """
    logits: (B, L, num_categories)
    x: (B, L, num_categories)

    returns: (B, L, num_categories)
    """
    num_categories = logits.shape[-1]
    mask_index = num_categories - 1
    
    # Zero Masking Probabilities
    probs = torch.zeros_like(logits)
    probs[:, :, :-1] = torch.softmax(logits[:, :, :-1], dim=-1)

    # Carry-Over Unmasking
    masked_tokens = x[:, :, mask_index].bool().unsqueeze(-1).expand_as(probs)
    new_probs = torch.where(masked_tokens, probs, x)
    return new_probs


def subs_parametrization_continuous_v1(logits: Tensor, x: Tensor) -> Tensor:
    """
    logits: (B, L, num_categories)
    x: (B, L, num_categories)

    returns: (B, L, num_categories)
    """
    # Zero Masking Probabilities
    probs = torch.zeros_like(logits)
    probs[:, :, :-1] = torch.softmax(logits[:, :, :-1], dim=-1)

    # Carry-Over Unmasking
    # I am using the continuous formulation described in 
    # https://www.notion.so/Calculaing-the-gradients-for-the-denoising-model-1e12ec4bd55180ec8e84f77d7a490cec?pvs=4#1e12ec4bd5518089a9d2d321b13f29dc
    new_probs = torch.zeros_like(probs)
    new_probs[:, :, :-1] = (
        probs[:, :, :-1] * (1 - x[:, :, :-1].sum(dim=-1, keepdim=True))
        + x[:, :, :-1]
    )
    return new_probs

def subs_parametrization_continuous_v2(logits: Tensor, x: Tensor) -> Tensor:
    """
    logits: (B, L, num_categories)
    x: (B, L, num_categories)

    returns: (B, L, num_categories)
    """
    # Zero Masking Probabilities
    probs = torch.zeros_like(logits)
    probs[:, :, :-1] = torch.softmax(logits[:, :, :-1], dim=-1)

    # Carry-Over Unmasking
    # I am using the continuous formulation described in 
    # https://www.notion.so/Calculaing-the-gradients-for-the-denoising-model-1e12ec4bd55180ec8e84f77d7a490cec?pvs=4#1e12ec4bd5518089a9d2d321b13f29dc
    new_probs = torch.zeros_like(probs)
    new_probs[:, :, :-1] = (
        probs[:, :, :-1] * x[:, :, -1].unsqueeze(dim=-1)
        + x[:, :, :-1]
    )
    return new_probs

# Default version
subs_parametrization_continuous = subs_parametrization_continuous_v1
