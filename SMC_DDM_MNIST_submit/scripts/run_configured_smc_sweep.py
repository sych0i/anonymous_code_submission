#!/usr/bin/env python3
"""Run the SMC MNIST sweep using configs/smc_sweep.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

from run_smc_binarized_mnist_sweep import main


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parents[1] / "configs" / "smc_sweep.yaml"
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(default_config)])
    raise SystemExit(main())
