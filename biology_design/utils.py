"""Small runtime utilities for deterministic per-run seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, use_cuda: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"=> Seed of the run set to {seed}")
