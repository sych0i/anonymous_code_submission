"""Minimal pretrained MDLM runtime used by the SMC experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import models
import noise_schedule


class Diffusion(nn.Module):
    """Inference-only form of the pretrained Gosai diffusion model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = 5
        self.mask_index = 4
        self.parameterization = config.parameterization
        self.time_conditioning = config.time_conditioning
        self.neg_infinity = -1_000_000.0

        if config.backbone != "cnn":
            raise ValueError(f"Unknown backbone: {config.backbone}")
        self.backbone = models.dnaconv.CNNModel(
            config.model,
            alphabet_size=self.vocab_size,
            num_cls=3,
        )
        self.noise = noise_schedule.get_noise(config, dtype=torch.float32)
        if self.parameterization != "subs":
            raise ValueError("This submission supports only subs parameterization")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *, config):
        model = cls(config)
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location="cpu",
            mmap=True,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model

    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        if xt.ndim > 2 and xt.shape[-1] == self.vocab_size:
            xt = xt.argmax(dim=-1)
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        if sigma.ndim != 1:
            raise ValueError(f"sigma must be one-dimensional, got {sigma.shape}")
        return sigma

    def forward(self, x, sigma, return_raw_logits=False):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.backbone(x, sigma)
        if return_raw_logits:
            return logits
        return self._subs_parameterization(logits=logits, xt=x)

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def get_logits(self, x, t):
        sigma_t, _ = self.noise(t)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_t.ndim != 1:
            raise ValueError(f"sigma_t must be one-dimensional, got {sigma_t.shape}")
        logits = self.forward(x, sigma_t, return_raw_logits=True)
        logits[..., self.mask_index] = -torch.inf
        return logits.float()
