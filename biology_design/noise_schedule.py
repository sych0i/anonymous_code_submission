"""Log-linear continuous-time noise schedule used by pretrained MDLM."""

from __future__ import annotations

import torch
from torch import nn


class LogLinearNoise(nn.Module):
    def __init__(self, eps: float = 1.0e-3):
        super().__init__()
        self.eps = float(eps)

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)


def get_noise(config, dtype=torch.float32):
    del dtype
    if config.noise.type != "loglinear":
        raise ValueError("This submission supports only noise.type=loglinear")
    return LogLinearNoise()
