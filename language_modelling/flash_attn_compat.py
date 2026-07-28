"""
If the flash-attn CUDA extension cannot load (e.g. glibc too old for prebuilt wheels),
register a minimal flash_attn-compatible API backed by torch SDPA.

Call ensure() before importing mdlm model code (dit / modeling_mdlm).
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from typing import Any

import torch
import torch.nn.functional as F


def _make_module(name: str, *, is_package: bool) -> Any:
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    mod = importlib.util.module_from_spec(spec)
    if is_package:
        mod.__path__ = []
    sys.modules[name] = mod
    return mod


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb_qkv_(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    q, k, v = qkv.unbind(2)
    cos = cos.to(device=qkv.device, dtype=qkv.dtype)
    sin = sin.to(device=qkv.device, dtype=qkv.dtype)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    q_out = q * cos_full + _rotate_half(q) * sin_full
    k_out = k * cos_full + _rotate_half(k) * sin_full
    qkv[:, :, 0].copy_(q_out)
    qkv[:, :, 1].copy_(k_out)
    return qkv


def _flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float,
    causal: bool = False,
) -> torch.Tensor:
    total, three, nheads, headdim = qkv.shape
    if three != 3:
        raise ValueError("expected qkvpacked with 3 in dim 1")
    batch_size = cu_seqlens.numel() - 1
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    if not (lens == max_seqlen).all():
        raise RuntimeError(
            "flash_attn SDPA fallback only supports fixed-length batches. "
            "Install a working flash-attn build for variable-length attention."
        )
    qkv_b = qkv.view(batch_size, max_seqlen, 3, nheads, headdim)
    q, k, v = qkv_b.unbind(2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    p = float(dropout_p) if dropout_p else 0.0
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=causal)
    return out.transpose(1, 2).contiguous().view(total, nheads, headdim)


def _install_shim() -> None:
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            del sys.modules[name]

    fa: Any = _make_module("flash_attn", is_package=True)
    layers: Any = _make_module("flash_attn.layers", is_package=True)
    rotary: Any = _make_module("flash_attn.layers.rotary", is_package=False)
    rotary.apply_rotary_emb_qkv_ = _apply_rotary_emb_qkv_
    iface: Any = _make_module("flash_attn.flash_attn_interface", is_package=False)
    iface.flash_attn_varlen_qkvpacked_func = _flash_attn_varlen_qkvpacked_func
    fa.layers = layers
    layers.rotary = rotary
    fa.flash_attn_interface = iface


def ensure() -> None:
    """Use real flash-attn when loadable; otherwise install SDPA shim."""
    try:
        from flash_attn.flash_attn_interface import (  # noqa: F401
            flash_attn_varlen_qkvpacked_func,
        )

        return
    except Exception:
        pass
    _install_shim()
