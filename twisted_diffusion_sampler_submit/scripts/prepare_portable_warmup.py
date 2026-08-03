#!/usr/bin/env python3
"""Build the reusable M=3 warmup summary at the current checkout path.

The frozen value curve is portable data.  The compatibility record is created
at runtime so it contains this checkout's absolute paths, which lets
``run_vista.py`` validate the warmup source after the folder is moved.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from image_exp import run_vista  # noqa: E402


SOURCE = ROOT / "artifacts" / "shared_warmup" / "value_means.json"
OUTPUT = ROOT / "artifacts" / "shared_warmup" / "summary.json"
EXPECTED = ROOT / "expected_results.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_args():
    parser = run_vista.build_parser()
    args = parser.parse_args(
        [
            "--model-config",
            str(ROOT / "image_exp/image_confs/mnist_model_and_diffusion_conf.yml"),
            "--task-config",
            str(ROOT / "image_exp/image_confs/task_class_cond_gen_conf.yml"),
            "--output-dir",
            str(ROOT / "outputs"),
            "--run-name",
            "portable-warmup-config",
            "--T",
            "100",
            "--T-prime",
            "12",
            "--N",
            "20",
            "--J",
            "1",
            "--M",
            "3",
            "--target",
            "4",
            "--alpha",
            "2",
            "--ess-threshold",
            "0.98",
            "--partial-resample",
            "20",
            "--value-estimation",
            "one_shot",
            "--final-transition",
            "legacy_deterministic",
            "--clip-twisted",
            "--no-clip-denoised",
            "--seed",
            "0",
            "--warmup-seed-offset",
            "100000",
            "--warmup-only",
            "--no-save-grids",
            "--no-plot-vhat",
        ]
    )
    return run_vista.validate_args(args)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare() -> Path:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not EXPECTED.is_file():
        raise FileNotFoundError(EXPECTED)

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    experiment = expected.get("experiment", {})
    total_steps = int(experiment.get("total_steps", 100))
    warmup_runs = int(experiment.get("warmup_runs", 3))
    schedule = source.get("schedule")
    values = np.asarray(source.get("value_means"), dtype=np.float64)
    if source.get("M") != warmup_runs:
        raise ValueError(f"warmup M mismatch: {source.get('M')} != {warmup_runs}")
    if schedule != list(range(total_steps)):
        raise ValueError("portable warmup schedule must be the full schedule")
    if values.shape != (total_steps + 1,) or not np.isfinite(values).all():
        raise ValueError(
            f"expected finite value_means with shape {(total_steps + 1,)}, "
            f"got {values.shape}"
        )

    args = build_args()
    warmup_config = run_vista._warmup_compatibility_config(args)
    expected_hash = expected["artifact_hashes"]["implementation_sha256"]
    actual_hash = warmup_config["provenance"]["implementation_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(
            "implementation hash does not match the reference: "
            f"{actual_hash} != {expected_hash}"
        )

    payload = {
        "M": warmup_runs,
        "artifacts": {"combined_npz": None, "value_curve_plot": None},
        "phase_wall_seconds": 0.0,
        "runs": [],
        "samples_excluded_from_policy_metrics": True,
        "sampling_reused": True,
        "sampling_seconds_total": 0.0,
        "schedule": schedule,
        "source": "frozen_m3_warmup_value_means",
        "value_means": values.tolist(),
        "warmup_config": warmup_config,
    }
    atomic_write_json(OUTPUT, payload)

    loaded = run_vista._load_warmup_source(OUTPUT, args=args)
    if not loaded["compatibility_verified"]:
        raise RuntimeError("portable warmup compatibility was not verified")
    if len(loaded["curves"]) != 1:
        raise RuntimeError("portable warmup should expose one frozen average curve")
    if not np.allclose(loaded["averaged_values"], values, rtol=1e-7, atol=1e-12):
        raise RuntimeError("portable warmup values changed during serialization")
    return OUTPUT


if __name__ == "__main__":
    print(prepare())
