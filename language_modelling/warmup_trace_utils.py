from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import ListConfig


@dataclass(frozen=True)
class WarmupSample:
    sample_id: int
    prompt_idx: int | None
    run_idx: int | None
    repeat_idx: int | None
    tag: str
    trace_path: str
    text_path: str | None
    abc_path: str | None
    seconds: float | None
    series: list[float]


def as_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        value = list(value)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(v) for v in value]


def resolve_path(path: str | None, base_dir: str) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def flatten_floats(vals):
    if vals is None:
        return []
    if isinstance(vals, (int, float)):
        return [float(vals)]
    out = []
    for v in vals:
        if isinstance(v, (list, tuple)):
            out.extend(flatten_floats(v))
        else:
            out.append(float(v))
    return out


def mean_exp_alpha_r_series_from_trace_jsonl(trace_path: str) -> list[float]:
    """
    Per SMC step, return the particle mean of the rollout value estimates.

    New traces store Algorithm 5's Monte Carlo estimates directly in
    ``value_estimates``. Legacy traces only contain rollout log rewards, so they
    retain the old exp(scale * reward) conversion for backwards compatibility.
    """
    series: list[float] = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            values = flatten_floats(row.get("value_estimates"))
            if values:
                series.append(sum(values) / len(values))
                continue
            scale_cur = float(row.get("scale_cur", 5.0))
            flat = flatten_floats(row.get("reward_aggregated", []))
            if not flat:
                series.append(float("nan"))
            else:
                series.append(sum(math.exp(scale_cur * v) for v in flat) / len(flat))
    return series[::-1]


def ensure_boundary_series(series, T: int) -> list[float]:
    vals = [float(x) for x in series]
    if len(vals) == T + 1:
        return vals
    if len(vals) == T:
        # Legacy traces recorded timesteps 0..T-1 only.  Reuse the first
        # high-noise measurement as the missing boundary value V_T.
        return vals + [vals[-1]]
    raise ValueError(
        f"Warmup series must have length T or T+1 for T={T}; got length {len(vals)}"
    )


def load_timing_by_tag(timing_log_path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not os.path.isfile(timing_log_path):
        return out
    with open(timing_log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tag = str(row.get("tag", ""))
            if tag:
                out[tag] = float(row.get("seconds", 0.0))
    return out


def _path_from_record(record: dict[str, Any], key: str, manifest_dir: str) -> str | None:
    path = record.get(key)
    if not path:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(manifest_dir, path)


def _sample_from_manifest_record(
    record: dict[str, Any],
    sample_id: int,
    manifest_dir: str,
) -> WarmupSample:
    trace_path = _path_from_record(record, "trace_path", manifest_dir)
    if trace_path is None:
        raise ValueError(f"Manifest sample {sample_id} has no trace_path")
    series = record.get("series")
    return WarmupSample(
        sample_id=int(record.get("sample_id", sample_id)),
        prompt_idx=(
            None
            if record.get("prompt_idx") is None
            else int(record.get("prompt_idx"))
        ),
        run_idx=None if record.get("run_idx") is None else int(record.get("run_idx")),
        repeat_idx=(
            None
            if record.get("repeat_idx") is None
            else int(record.get("repeat_idx"))
        ),
        tag=str(record.get("tag", os.path.basename(trace_path))),
        trace_path=trace_path,
        text_path=_path_from_record(record, "text_path", manifest_dir),
        abc_path=_path_from_record(record, "abc_path", manifest_dir),
        seconds=None if record.get("seconds") is None else float(record.get("seconds")),
        series=[float(x) for x in series]
        if series is not None
        else mean_exp_alpha_r_series_from_trace_jsonl(trace_path),
    )


def load_samples_from_manifest(manifest_path: str) -> list[WarmupSample]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest.get("samples", manifest)
    if not isinstance(records, list):
        raise ValueError(f"{manifest_path} must contain a list or a top-level 'samples' list")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    return [
        _sample_from_manifest_record(record, sample_id, manifest_dir)
        for sample_id, record in enumerate(records)
    ]


def load_samples_from_trace_dir(trace_dir: str, trace_glob: str) -> list[WarmupSample]:
    pattern = os.path.join(trace_dir, trace_glob)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No warmup trace files matched {pattern}")

    timing_by_tag = load_timing_by_tag(os.path.join(trace_dir, "inference_timing.jsonl"))
    samples: list[WarmupSample] = []
    for sample_id, path in enumerate(paths):
        basename = os.path.basename(path)
        tag = basename.removeprefix("reward_trace_").removesuffix(".jsonl")
        samples.append(
            WarmupSample(
                sample_id=sample_id,
                prompt_idx=None,
                run_idx=None,
                repeat_idx=None,
                tag=tag,
                trace_path=path,
                text_path=os.path.join(trace_dir, f"text_samples_{tag}.jsonl"),
                abc_path=os.path.join(trace_dir, f"abc_ssdlm_gen_{tag}.jsonl"),
                seconds=timing_by_tag.get(tag),
                series=mean_exp_alpha_r_series_from_trace_jsonl(path),
            )
        )
    return samples


def load_warmup_samples(
    *,
    trace_manifest: str | None,
    trace_dir: str | None,
    trace_glob: str,
) -> list[WarmupSample]:
    if trace_manifest:
        samples = load_samples_from_manifest(trace_manifest)
    elif trace_dir:
        samples = load_samples_from_trace_dir(trace_dir, trace_glob)
    else:
        raise ValueError(
            "No saved all-step traces were configured. Run sample_allstep_traces.py first, "
            "then pass run_all.warmup_trace_manifest=... or run_all.warmup_trace_dir=...."
        )

    for sample in samples:
        with open(sample.trace_path, encoding="utf-8") as f:
            first_row = next(
                (json.loads(line) for line in f if line.strip()),
                None,
            )
        if not first_row or "value_estimates" not in first_row:
            raise ValueError(
                f"{sample.trace_path} is a legacy trace without rollout "
                "value_estimates. Regenerate it with the current "
                "sample_allstep_traces.py."
            )
    return samples


def dp_timestep_list_vista(mean_series: np.ndarray, T: int, T_: int) -> list[int]:
    """VISTA DP schedule with t_start chosen by argmax_s dp[s][T'] + s V_s."""
    if T_ <= 0 or T_ > T:
        raise ValueError(f"T_ must satisfy 1 <= T_ <= T, got T_={T_}, T={T}")
    if len(mean_series) <= T:
        raise ValueError(f"mean_series must contain index T={T}, got length {len(mean_series)}")

    dp = [[0.0] * (T_ + 1) for _ in range(T)]
    parent = [[0] * (T_ + 1) for _ in range(T)]

    for s in range(T_ - 1, T):
        dp[s][1] = (T - 1 - s) * mean_series[T]

    for i in range(2, T_ + 1):
        for s in range(T_ - i, T - i + 1):
            for s_next in range(s + 1, T - i + 2):
                c = dp[s_next][i - 1] + (s_next - s - 1) * mean_series[s_next]
                if c > dp[s][i] or s_next == s + 1:
                    dp[s][i] = c
                    parent[s][i] = s_next

    t_start = max(
        range(T - T_ + 1),
        key=lambda s: dp[s][T_] + s * mean_series[s],
    )

    timestep = [t_start]
    curr_s = t_start
    for i in range(T_, 1, -1):
        curr_s = parent[curr_s][i]
        timestep.append(curr_s)
    return timestep


def dp_timestep_list(mean_series: np.ndarray, T: int, T_: int) -> list[int]:
    """Backward-compatible alias for the VISTA DP schedule."""
    return dp_timestep_list_vista(mean_series, T, T_)


def uniform_steps(T: int, T_: int) -> list[int]:
    if T_ <= 0 or T_ > T:
        raise ValueError(f"T_ must satisfy 1 <= T_ <= T, got T_={T_}, T={T}")
    if T_ == 1:
        return [T - 1]
    return sorted({round((T - 1) * i / (T_ - 1)) for i in range(T_)})


def interval_steps(
    T: int,
    T_: int,
    interval: int,
    num_intervals: int = 5,
) -> list[int]:
    """A contiguous T'-step window at one of evenly spaced trajectory positions."""
    if T_ <= 0 or T_ > T:
        raise ValueError(f"T_ must satisfy 1 <= T_ <= T, got T_={T_}, T={T}")
    if num_intervals < 2:
        raise ValueError("num_intervals must be at least 2")
    if interval <= 0 or interval > num_intervals:
        raise ValueError(
            f"interval must be in [1, {num_intervals}], got {interval}"
        )
    start = round(
        (num_intervals - interval) * (T - T_) / (num_intervals - 1)
    )
    return list(range(start, start + T_))


def _top_steps(scores, T: int, T_: int) -> list[int]:
    if T_ <= 0 or T_ > T:
        raise ValueError(f"T_ must satisfy 1 <= T_ <= T, got T_={T_}, T={T}")
    vals = [float(score) for score in scores]
    if len(vals) != T:
        raise ValueError(f"scores must have length T={T}, got {len(vals)}")
    if not all(math.isfinite(value) for value in vals):
        raise ValueError("scores must contain only finite values")
    selected = sorted(range(T), key=lambda t: (-vals[t], t))[:T_]
    return sorted(selected)


def top_value_steps(mean_series, T: int, T_: int) -> list[int]:
    """Select the T' timesteps with largest V_hat_t, for t=0,...,T-1."""
    if len(mean_series) <= T:
        raise ValueError(
            f"mean_series must contain indices 0..T for T={T}, "
            f"got length {len(mean_series)}"
        )
    return _top_steps(mean_series[:T], T, T_)


def top_delta_value_steps(mean_series, T: int, T_: int) -> list[int]:
    """Select the T' largest V_hat_t - V_hat_{t+1} values."""
    if len(mean_series) <= T:
        raise ValueError(
            f"mean_series must contain indices 0..T for T={T}, "
            f"got length {len(mean_series)}"
        )
    deltas = [
        float(mean_series[t]) - float(mean_series[t + 1])
        for t in range(T)
    ]
    return _top_steps(deltas, T, T_)


def baseline_schedules(
    T: int,
    T_: int,
    reference_steps: list[int],
    mean_series=None,
) -> dict[str, list[int]]:
    schedules = {
        "reference_dp": sorted(reference_steps),
        "uniform": uniform_steps(T, T_),
        **{
            f"interval_{interval}": interval_steps(T, T_, interval)
            for interval in range(1, 6)
        },
    }
    if mean_series is not None:
        schedules["top_v"] = top_value_steps(mean_series, T, T_)
        schedules["top_dv"] = top_delta_value_steps(mean_series, T, T_)
    return schedules


def calculate_skipped_guidance_sum(T: int, T_prime, series) -> float:
    total_sum = 0.0
    T_prime_cup_T = sorted(list(T_prime) + [T])
    skipped_timesteps = set(range(T)) - set(T_prime)
    for t in skipped_timesteps:
        ceil_t = min(tp for tp in T_prime_cup_T if tp >= t)
        total_sum += float(series[ceil_t])
    return total_sum
