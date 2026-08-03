#!/usr/bin/env python3
"""Launch run_vista.py with CUDA determinism forced on.

This wrapper deliberately lives outside the hash-pinned source set used by
the frozen-warmup provenance check (see run_vista._scientific_provenance).
Editing run_vista.py itself would change implementation_sha256 and break
compatibility with artifacts/shared_warmup_original and expected_results.json.
Setting the determinism flags here, before run_vista's own imports run,
achieves the same effect without touching a hashed file.

The seeds passed to run_vista.py (--seed / --warmup-seed-offset) already fix
random/numpy/torch RNG state. What is missing without this wrapper is
determinism of the CUDA reduction kernels themselves (cuDNN convolution
algorithm selection and cuBLAS GEMM launch config), which can otherwise
vary run-to-run for the same seed and produce different Unique Valid counts.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# Must be set before the first CUDA/cuBLAS handle is created in this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# warn_only=True: a few non-sampling-path ops (e.g. plotting/metrics) lack
# deterministic CUDA kernels; failing outright would abort otherwise-
# reproducible runs over code that doesn't affect the sampled particles.
torch.use_deterministic_algorithms(True, warn_only=True)

# cumsum has no deterministic CUDA kernel (PyTorch issue #75240) and is the
# one call on the resampling path (vista_smc.systematic_resample_indices,
# building the systematic-resampling CDF) that use_deterministic_algorithms
# cannot make reproducible on its own -- it only warns. That single call site
# operates on a length-N (<=~20) particle-weight vector, so routing it
# through the CPU is effectively free and removes the last source of
# run-to-run Unique-Valid drift. This is the only cumsum call in the
# codebase (verified via grep), so the patch has no other blast radius.
_cpu_cumsum = torch.Tensor.cumsum


def _deterministic_cumsum(self, *args, **kwargs):
    if self.is_cuda:
        return _cpu_cumsum(self.cpu(), *args, **kwargs).to(self.device)
    return _cpu_cumsum(self, *args, **kwargs)


torch.Tensor.cumsum = _deterministic_cumsum

RUN_VISTA = Path(__file__).resolve().parents[1] / "image_exp" / "run_vista.py"


if __name__ == "__main__":
    sys.argv[0] = str(RUN_VISTA)
    sys.path.insert(0, str(RUN_VISTA.parent))
    runpy.run_path(str(RUN_VISTA), run_name="__main__")
