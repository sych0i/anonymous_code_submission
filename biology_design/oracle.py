"""ATAC classifier reward and evaluation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from grelu.lightning import LightningModel

import dataloader_gosai


class AtacRewardModel(torch.nn.Module):
    """Use log p(accessible | sequence), classifier channel 1, as the reward."""

    reward_type = "atac"

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        scale: float = 1.0,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        self.base_model = base_model
        self.scale = float(scale)
        self.eps = float(eps)

    def forward(self, onehot_channels_first: torch.Tensor) -> torch.Tensor:
        predictions = self.base_model(onehot_channels_first)
        while predictions.ndim > 2 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        if predictions.ndim == 1:
            if onehot_channels_first.shape[0] != 1:
                raise ValueError(
                    f"Unexpected ATAC output shape {tuple(predictions.shape)}"
                )
            predictions = predictions.unsqueeze(0)
        if predictions.ndim != 2 or predictions.shape[1] <= 1:
            raise ValueError(
                "ATAC classifier must return at least two channels; "
                f"got {tuple(predictions.shape)}"
            )
        return (
            predictions[:, 1].clamp(min=self.eps, max=1.0).log()
            * self.scale
        )


def _load_grelu_checkpoint(checkpoint_path: Path) -> LightningModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    model_params = dict(
        checkpoint.get("hyper_parameters", {}).get("model_params", {})
    )
    del checkpoint
    if model_params.get("model_type") == "EnformerPretrainedModel":
        model_params["model_type"] = "EnformerModel"
        return LightningModel.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model_params=model_params,
        )
    return LightningModel.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )


def get_atac_model(base_path: Optional[str] = None) -> LightningModel:
    root = Path(base_path or os.environ["DRAKES_DATA_ROOT"])
    model = _load_grelu_checkpoint(
        root / "mdlm/gosai_data/binary_atac_cell_lines.ckpt"
    )
    model.train_params["logger"] = None
    return model


def get_atac_reward_model(
    *,
    base_path: str,
    scale: float = 1.0,
    eps: float = 1.0e-8,
) -> AtacRewardModel:
    return AtacRewardModel(
        get_atac_model(base_path=base_path),
        scale=scale,
        eps=eps,
    )


def compute_reward_from_tokens(
    tokens: torch.Tensor,
    reward_model: torch.nn.Module,
) -> torch.Tensor:
    if tokens.ndim == 2:
        valid = (tokens >= 0) & (tokens < 4)
        safe_tokens = tokens.clamp(0, 3)
        onehot_tokens = F.one_hot(safe_tokens, num_classes=4).float()
        onehot_tokens = onehot_tokens * valid.unsqueeze(-1)
    elif tokens.ndim == 3:
        onehot_tokens = tokens[:, :, :4].float()
    else:
        raise ValueError("tokens must be two- or three-dimensional")

    rewards = reward_model(onehot_tokens.transpose(1, 2))
    return rewards.reshape(-1)


def cal_atac_pred_new(
    seqs: Sequence[str],
    *,
    model: Optional[torch.nn.Module] = None,
    base_path: Optional[str] = None,
) -> np.ndarray:
    model = model or get_atac_model(base_path=base_path)
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = model.device
    tokens = torch.as_tensor(
        dataloader_gosai.batch_dna_tokenize(seqs),
        dtype=torch.long,
        device=device,
    )
    onehot = F.one_hot(tokens, num_classes=4).float().transpose(1, 2)
    return model(onehot).detach().cpu().numpy().squeeze()
