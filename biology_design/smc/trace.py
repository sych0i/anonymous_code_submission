from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TRACE_SCHEMA_VERSION = 2


def flatten_floats(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [float(values)]
    out: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(flatten_floats(value))
        else:
            out.append(float(value))
    return out


def exp_reward_from_trace_row(
    row: dict[str, Any],
    *,
    gamma: float = 1.0,
) -> float:
    """
    Estimate ``E[exp(scale * reward)]`` using residual normalized SMC weights.
    """
    rewards = flatten_floats(row.get("reward_aggregated", []))
    if not rewards:
        raise ValueError("trace row has no reward_aggregated values")

    num_batches = int(row.get("num_batches", 1))
    num_particles = int(row.get("num_particles", len(rewards)))
    if num_batches < 1 or num_particles < 1:
        raise ValueError("num_batches and num_particles must be positive")
    if num_batches * num_particles != len(rewards):
        raise ValueError(
            "trace shape metadata does not match reward count: "
            f"{num_batches} * {num_particles} != {len(rewards)}"
        )

    scale = float(row.get("scale_cur", 1.0)) * float(gamma)
    values = []
    for reward in rewards:
        try:
            values.append(math.exp(scale * reward))
        except OverflowError:
            values.append(float("inf"))
    exp_values = np.asarray(values, dtype=float)
    weights = flatten_floats(row.get("normalized_weights", []))
    if len(weights) != len(rewards):
        raise ValueError(
            "trace row must contain one normalized weight per reward; "
            f"got {len(weights)} weights and {len(rewards)} rewards"
        )
    batch_estimates: list[float] = []
    for batch_index in range(num_batches):
        start = batch_index * num_particles
        stop = start + num_particles
        batch_weights = np.asarray(weights[start:stop], dtype=float)
        if not np.all(np.isfinite(batch_weights)) or np.any(batch_weights < 0):
            raise ValueError("trace contains invalid normalized weights")
        weight_sum = float(batch_weights.sum())
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("trace normalized weights must have positive finite mass")
        batch_weights = batch_weights / weight_sum
        batch_estimates.append(
            float(np.dot(batch_weights, exp_values[start:stop]))
        )
    return float(np.mean(np.asarray(batch_estimates, dtype=float)))


def mean_exp_reward_series_from_trace_jsonl(
    trace_path: Path,
    *,
    gamma: float = 1.0,
) -> list[float]:
    """
    Read a schema-v2 full-guidance trace as ``[V_0, ..., V_T]``.

    Legacy traces are rejected because their timestep labels describe the
    transition destination while their rewards were evaluated at the source.
    They must be regenerated after the SMC ordering fix.
    """
    values_by_timestep: dict[int, float] = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("schema_version", 0)) < TRACE_SCHEMA_VERSION:
                raise ValueError(
                    f"{trace_path}:{line_number} is a legacy reward trace. "
                    "Rerun eval_allstep.py to generate schema-v2 V_0..V_T values."
                )
            if "state_timestep" not in row:
                raise ValueError(
                    f"{trace_path}:{line_number} is missing state_timestep"
                )
            state_timestep = int(row["state_timestep"])
            if state_timestep < 0:
                raise ValueError(
                    f"{trace_path}:{line_number} has negative state_timestep"
                )
            if state_timestep in values_by_timestep:
                raise ValueError(
                    f"{trace_path} contains duplicate state_timestep={state_timestep}"
                )
            values_by_timestep[state_timestep] = (
                exp_reward_from_trace_row(
                    row,
                    gamma=gamma,
                )
            )

    if not values_by_timestep:
        raise ValueError(f"{trace_path} contains no reward trace rows")

    terminal_timestep = max(values_by_timestep)
    expected = set(range(terminal_timestep + 1))
    missing = sorted(expected - set(values_by_timestep))
    if missing:
        raise ValueError(
            f"{trace_path} is not a full-guidance trace; missing state timesteps {missing}"
        )
    return [
        values_by_timestep[timestep]
        for timestep in range(terminal_timestep + 1)
    ]


def mean_series_from_traces(
    trace_paths: Sequence[Path],
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    return np.mean(
        trace_series_matrix_from_traces(
            trace_paths,
            gamma=gamma,
        ),
        axis=0,
    )


def trace_series_matrix_from_traces(
    trace_paths: Sequence[Path],
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    """Return one ``[V_0, ..., V_T]`` row per full-guidance trace."""
    if not trace_paths:
        raise ValueError("no allstep reward traces found")
    series = [
        mean_exp_reward_series_from_trace_jsonl(
            path,
            gamma=gamma,
        )
        for path in trace_paths
    ]
    lengths = {len(item) for item in series}
    if len(lengths) != 1:
        detail = ", ".join(
            f"{path.name}:{len(item)}"
            for path, item in zip(trace_paths, series)
        )
        raise ValueError(f"trace series lengths differ: {detail}")
    return np.asarray(series, dtype=float)
