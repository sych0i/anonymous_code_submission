"""Shared ATAC-only evaluation and schedule utilities."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import dataloader_gosai
import diffusion_gosai_update
import oracle
from eval_assets import validate_data_root
from smc.pipeline import Pipeline
from smc.resampling import resample
from smc.scheduler import MDLMScheduler
from smc.trace import (
    exp_reward_from_trace_row,
    mean_series_from_traces,
)
from smc.vista import (
    allstep_schedule,
    build_schedule_map,
    calculate_skipped_guidance_sum,
    compute_vista_schedule,
    validate_guidance_count,
    validate_mean_series_length,
)
from utils import set_seed


DEFAULT_EVAL_RUNS = 30
METRIC_NAMES = ("atac", "unique_atac_count")
DEFAULT_EVAL_METHODS = (
    "uniformsteps",
    "vista",
    "top_v",
    "top_dv",
    "interval1",
    "interval2",
    "interval3",
    "interval4",
    "interval5",
)


@dataclass
class EvalComponents:
    device: torch.device
    base_path: str
    diffusion_model: diffusion_gosai_update.Diffusion
    reward_model: oracle.AtacRewardModel
    pipe: Pipeline
    atac_model: torch.nn.Module
    unique_hamming_threshold: float
    atac_threshold: float


def cfg_select(config: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(config, key, default=default)


def configure_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(filename)s - %(asctime)s - %(levelname)s --> %(message)s"
    )
    for handler in (
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_device(config: DictConfig) -> torch.device:
    cuda_idx = cfg_select(config, "cuda_device", 0)
    if cuda_idx is None or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(cuda_idx)}")


def load_components(
    config: DictConfig,
    *,
    logger: Optional[logging.Logger] = None,
    reproduce_allstep_numerics: bool = False,
    unique_hamming_threshold: Optional[float] = None,
) -> EvalComponents:
    device = get_device(config)
    base_path = validate_data_root()
    checkpoint_path = (
        Path(base_path) / "mdlm/outputs_gosai/pretrained.ckpt"
    )
    if logger:
        logger.info("loading pretrained diffusion and ATAC classifier on %s", device)
    diffusion_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(
        checkpoint_path,
        config=config,
    ).to(device)
    diffusion_model.eval()
    reward_model = oracle.get_atac_reward_model(
        base_path=base_path,
        scale=1.0,
        eps=1.0e-8,
    ).to(device)
    reward_model.eval()
    scheduler = MDLMScheduler(
        model=diffusion_model,
        reproduce_allstep_numerics=reproduce_allstep_numerics,
    )
    pipe = Pipeline(
        diffusion_model,
        scheduler,
        device,
        model_dtype=torch.float,
    )
    return EvalComponents(
        device=device,
        base_path=base_path,
        diffusion_model=diffusion_model,
        reward_model=reward_model,
        pipe=pipe,
        atac_model=reward_model.base_model,
        unique_hamming_threshold=float(
            cfg_select(config, "eval.unique_hamming_threshold", 0.005)
            if unique_hamming_threshold is None
            else unique_hamming_threshold
        ),
        atac_threshold=float(cfg_select(config, "eval.atac_threshold", 0.5)),
    )


def paper_T(config: DictConfig) -> int:
    return int(cfg_select(config, "smc.num_inference_steps"))


def guidance_count(config: DictConfig) -> int:
    return int(cfg_select(config, "run_all.num_guidance_steps"))


def _natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"itr(\d+)_", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def find_allstep_traces(run_dir: Path) -> list[Path]:
    return sorted(
        run_dir.glob("reward_trace_itr*_allsteps.jsonl"),
        key=_natural_key,
    )


def save_mean_series(
    out_dir: Path,
    mean_series: Sequence[float],
) -> None:
    values = np.asarray(mean_series, dtype=float)
    np.save(out_dir / "mean_series.npy", values)
    with (out_dir / "mean_series.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", "mean_series"])
        writer.writeheader()
        for timestep, value in enumerate(values):
            writer.writerow({"t": timestep, "mean_series": float(value)})


def write_schedule_json(
    path: Path,
    *,
    name: str,
    T: int,
    T_prime: int,
    timesteps: Sequence[int],
    source_run_dir: Path,
    scores: dict[str, float],
    trace_files: Sequence[str],
    mean_series: Sequence[float],
) -> None:
    validate_mean_series_length(T, mean_series)
    payload = {
        "name": name,
        "T": int(T),
        "T_prime": int(T_prime),
        "timesteps": [int(timestep) for timestep in timesteps],
        "source_run_dir": os.path.relpath(source_run_dir, Path.cwd()),
        "scores": {key: float(value) for key, value in scores.items()},
        "metadata": {
            "warmup_mode": "prefix",
            "warmup_runs_used": len(trace_files),
            "gamma": 1.0,
            "particle_aggregation": "weighted",
            "trace_files": list(trace_files),
        },
        "mean_series": [float(value) for value in mean_series],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_schedule_scores_csv(
    path: Path,
    *,
    scores: dict[str, float],
    T: int,
    T_prime: int,
    trace_files: Sequence[str],
) -> None:
    fieldnames = [
        "method",
        "T",
        "T_prime",
        "schedule_score",
        "warmup_runs_used",
        "gamma",
        "particle_aggregation",
        "trace_files",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, score in scores.items():
            writer.writerow(
                {
                    "method": method,
                    "T": T,
                    "T_prime": T_prime,
                    "schedule_score": float(score),
                    "warmup_runs_used": len(trace_files),
                    "gamma": 1.0,
                    "particle_aggregation": "weighted",
                    "trace_files": json.dumps(list(trace_files)),
                }
            )


def plot_reward_trace(
    trace_path: Path,
    out_path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        timesteps: list[int] = []
        values: list[float] = []
        resampled_timesteps: list[int] = []
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                timestep = int(row["state_timestep"])
                timesteps.append(timestep)
                values.append(exp_reward_from_trace_row(row))
                if row.get("resampled", False):
                    resampled_timesteps.append(timestep)
        if not timesteps:
            return
        plt.figure(figsize=(10, 4))
        plt.plot(
            timesteps,
            values,
            label="particle-weighted mean exp(scale × reward)",
        )
        if resampled_timesteps:
            value_by_t = dict(zip(timesteps, values))
            plt.scatter(
                resampled_timesteps,
                [value_by_t[t] for t in resampled_timesteps],
                marker="x",
                s=40,
                linewidths=2,
                label="resampled",
            )
        plt.gca().invert_xaxis()
        plt.xlabel("State timestep t (128 = masked, 0 = clean)")
        plt.ylabel("weighted mean exp(scale × reward)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception as exc:
        if logger:
            logger.warning("failed to plot %s: %s", trace_path, exc)


def plot_value_trace(
    mean_series: Sequence[float],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(mean_series, dtype=float)
    timesteps = np.arange(len(values))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis in axes:
        axis.plot(timesteps, values, color="tab:blue", linewidth=2.0)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("value (linear)")
    threshold = max(1.0e-12, float(np.max(np.abs(values))) * 1.0e-5)
    axes[1].set_yscale("symlog", linthresh=threshold)
    axes[1].set_ylabel("value (symlog)")
    axes[1].set_xlabel("Forward timestep t")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_smc_once(
    components: EvalComponents,
    config: DictConfig,
    *,
    guidance_steps: set[int],
    tag: str,
    seed: int,
) -> tuple[torch.Tensor, float, Path]:
    os.environ["SMC_TRACE_TAG"] = tag
    set_seed(seed, use_cuda=torch.cuda.is_available())
    started = time.perf_counter()
    samples = components.pipe(
        resample_fn=lambda log_w: resample(
            log_w,
            ess_threshold=float(
                cfg_select(config, "smc.resampling.ess_threshold")
            ),
            partial=bool(cfg_select(config, "smc.resampling.partial")),
        ),
        reward_fn=lambda tokens: oracle.compute_reward_from_tokens(
            tokens,
            components.reward_model,
        ),
        batches=int(cfg_select(config, "smc.batches")),
        num_particles=int(cfg_select(config, "smc.num_particles")),
        batch_p=int(cfg_select(config, "smc.batch_p")),
        resample_frequency=int(
            cfg_select(config, "smc.resampling.frequency")
        ),
        num_inference_steps=int(
            cfg_select(config, "smc.num_inference_steps")
        ),
        guidance_steps=guidance_steps,
        use_continuous_formulation=bool(
            cfg_select(config, "smc.use_continuous_formulation")
        ),
        kl_weight=float(cfg_select(config, "smc.kl_weight")),
        phi=int(cfg_select(config, "smc.phi")),
        tau=float(cfg_select(config, "smc.tau")),
        final_strategy=str(cfg_select(config, "smc.final_strategy")),
        disable_progress_bar=bool(
            cfg_select(config, "smc.disable_progress_bar")
        ),
        verbose=bool(cfg_select(config, "smc.verbose")),
    )
    return (
        samples,
        time.perf_counter() - started,
        Path.cwd() / f"reward_trace_{tag}.jsonl",
    )


def save_samples(samples: torch.Tensor, tag: str, out_dir: Path) -> None:
    array = samples.detach().cpu().numpy()
    np.save(out_dir / f"samples_{tag}.npy", array)
    with (out_dir / f"samples_{tag}.txt").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for sequence in dataloader_gosai.batch_dna_detokenize(array):
            handle.write(f"{sequence}\n")


def append_timing_row(path: Path, row: dict[str, Any]) -> None:
    _append_csv_row(
        path,
        row,
        fieldnames=["tag", "method", "itr", "elapsed_s"],
    )


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    _append_csv_row(
        path,
        row,
        fieldnames=[
            "method",
            "itr",
            "elapsed_s",
            "atac",
            "unique_atac_count",
        ],
    )


def _append_csv_row(
    path: Path,
    row: dict[str, Any],
    *,
    fieldnames: Sequence[str],
) -> None:
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def hamming_threshold_clusters(
    seqs: Sequence[str],
    *,
    threshold: float,
) -> list[list[int]]:
    count = len(seqs)
    if count == 0:
        return []
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(count):
        for right in range(left + 1, count):
            first, second = seqs[left], seqs[right]
            denominator = max(len(first), len(second), 1)
            shared = min(len(first), len(second))
            mismatches = abs(len(first) - len(second)) + sum(
                first[index] != second[index] for index in range(shared)
            )
            if mismatches / denominator <= float(threshold):
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def count_unique_by_hamming_threshold(
    seqs: Sequence[str],
    *,
    threshold: float,
) -> int:
    if float(threshold) == 0.0:
        return len(set(seqs))
    return len(hamming_threshold_clusters(seqs, threshold=threshold))


def eval_get_metrics(
    detokenized_samples: Sequence[str],
    components: EvalComponents,
) -> dict[str, float]:
    sample_count = len(detokenized_samples)
    with torch.inference_mode():
        predictions = np.asarray(
            oracle.cal_atac_pred_new(
                detokenized_samples,
                model=components.atac_model,
            )
        )
    if predictions.ndim == 1:
        predictions = predictions[None, :]
    passing = predictions[:, 1] >= components.atac_threshold
    passing_sequences = [
        sequence
        for sequence, keep in zip(detokenized_samples, passing)
        if bool(keep)
    ]
    return {
        "atac": (
            float(passing.sum() / sample_count)
            if sample_count
            else float("nan")
        ),
        "unique_atac_count": float(
            count_unique_by_hamming_threshold(
                passing_sequences,
                threshold=components.unique_hamming_threshold,
            )
        ),
    }


def write_metrics_summary(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    output: list[dict[str, Any]] = []
    for method, method_rows in grouped.items():
        summary: dict[str, Any] = {
            "method": method,
            "n_runs": len(method_rows),
        }
        for metric in ("elapsed_s", "atac", "unique_atac_count"):
            values = np.asarray(
                [float(row[metric]) for row in method_rows],
                dtype=float,
            )
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values))
        output.append(summary)
    fieldnames = list(output[0]) if output else ["method", "n_runs"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
    return output


def parse_method_list(
    value: Any,
    default: Sequence[str] = DEFAULT_EVAL_METHODS,
) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item) for item in value]
