"""Validation for the two model assets used by the ATAC-only experiments."""

from __future__ import annotations

import os
from pathlib import Path


PRETRAINED_MODEL_PATH = Path("mdlm/outputs_gosai/pretrained.ckpt")
ATAC_MODEL_PATH = Path("mdlm/gosai_data/binary_atac_cell_lines.ckpt")


def validate_data_root() -> str:
    default_root = Path(__file__).resolve().parent / "model_weights/data_and_model"
    raw_root = os.environ.get("DRAKES_DATA_ROOT", str(default_root))
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        raise ValueError(
            "DRAKES_DATA_ROOT must be absolute because Hydra changes the "
            f"working directory; got {raw_root!r}"
        )
    root = root.resolve()
    missing = [
        relative
        for relative in (PRETRAINED_MODEL_PATH, ATAC_MODEL_PATH)
        if not (root / relative).is_file()
    ]
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing required model assets under {root}:\n{detail}"
        )
    os.environ["DRAKES_DATA_ROOT"] = str(root)
    return str(root)
