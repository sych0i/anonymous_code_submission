from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence


def validate_guidance_count(T: int, T_prime: int) -> None:
    if not (1 <= T_prime <= T):
        raise ValueError(
            f"num_guidance_steps must satisfy 1 <= T' <= T={T}, got {T_prime}"
        )


def allstep_schedule(T: int) -> set[int]:
    return set(range(T))


def interval_schedule(T: int, T_prime: int, interval_index: int) -> set[int]:
    """
    Return the kth contiguous interval from first to last denoising steps.

    Reverse diffusion runs from high to low timestep, so interval 1 starts at
    ``T - T_prime`` (first executed steps) and interval 5 starts at zero (last
    executed steps).
    """
    validate_guidance_count(T, T_prime)
    if interval_index not in range(1, 6):
        raise ValueError(
            f"interval_index must be one of 1, 2, 3, 4, 5; got {interval_index}"
        )
    start = round((T - T_prime) * (5 - interval_index) / 4)
    return set(range(start, start + T_prime))


def _top_timesteps(
    scores: Sequence[float],
    T_prime: int,
    *,
    label: str,
) -> set[int]:
    invalid = [
        timestep
        for timestep, score in enumerate(scores)
        if not math.isfinite(float(score))
    ]
    if invalid:
        raise ValueError(f"{label} contains non-finite values at timesteps {invalid}")
    # A smaller timestep wins an exact tie, making the policy deterministic.
    ranked = sorted(
        range(len(scores)),
        key=lambda timestep: (-float(scores[timestep]), timestep),
    )
    return set(ranked[:T_prime])


def top_value_schedule(
    T: int,
    T_prime: int,
    mean_series: Sequence[float],
) -> set[int]:
    validate_guidance_count(T, T_prime)
    validate_mean_series_length(T, mean_series)
    return _top_timesteps(
        [float(mean_series[timestep]) for timestep in range(T)],
        T_prime,
        label="V_t",
    )


def top_delta_value_schedule(
    T: int,
    T_prime: int,
    mean_series: Sequence[float],
) -> set[int]:
    validate_guidance_count(T, T_prime)
    validate_mean_series_length(T, mean_series)
    return _top_timesteps(
        [
            float(mean_series[timestep])
            - float(mean_series[timestep + 1])
            for timestep in range(T)
        ],
        T_prime,
        label="V_t - V_{t+1}",
    )


def heuristic_schedules(
    T: int,
    T_prime: int,
    mean_series: Optional[Sequence[float]] = None,
) -> dict[str, set[int]]:
    """Build all non-VISTA sparse comparison policies."""
    validate_guidance_count(T, T_prime)
    schedules = {
        "uniformsteps": (
            {0}
            if T_prime == 1
            else {
                round((T - 1) * i / (T_prime - 1))
                for i in range(T_prime)
            }
        ),
    }
    if mean_series is not None:
        validate_mean_series_length(T, mean_series)
        schedules.update(
            {
                "top_v": top_value_schedule(T, T_prime, mean_series),
                "top_dv": top_delta_value_schedule(T, T_prime, mean_series),
            }
        )
    schedules.update(
        {
            f"interval{interval_index}": interval_schedule(
                T,
                T_prime,
                interval_index,
            )
            for interval_index in range(1, 6)
        }
    )
    return schedules


def validate_mean_series_length(
    T: int,
    mean_series: Sequence[float],
    *,
    label: str = "mean_series",
) -> None:
    length = len(mean_series)
    if length != T + 1:
        raise ValueError(
            f"{label} length={length}; expected T+1={T + 1} explicit "
            "state values [V_0, ..., V_T]. Regenerate legacy all-step traces."
        )


def mean_series_value(
    mean_series: Sequence[float],
    index: int,
    T: int,
) -> float:
    if 0 <= index <= T and index < len(mean_series):
        return float(mean_series[index])
    raise IndexError(
        f"mean_series has length {len(mean_series)}; cannot read index {index}"
    )


def calculate_skipped_guidance_sum(
    T: int,
    T_prime_steps: Iterable[int],
    series: Sequence[float],
) -> float:
    validate_mean_series_length(T, series)
    steps = {int(timestep) for timestep in T_prime_steps}
    invalid = sorted(timestep for timestep in steps if timestep < 0 or timestep >= T)
    if invalid:
        raise ValueError(f"guidance timesteps must be in [0, {T - 1}], got {invalid}")

    active_with_terminal = sorted([*steps, T])
    total = 0.0
    for timestep in set(range(T)) - steps:
        ceiling = min(
            active for active in active_with_terminal if active >= timestep
        )
        total += mean_series_value(series, ceiling, T)
    return total


def compute_vista_schedule(
    T: int,
    T_prime: int,
    mean_series: Sequence[float],
) -> list[int]:
    validate_guidance_count(T, T_prime)
    validate_mean_series_length(T, mean_series)
    terminal_value = mean_series_value(mean_series, T, T)

    dp = [[0.0] * (T_prime + 1) for _ in range(T)]
    parent = [[0] * (T_prime + 1) for _ in range(T)]

    for start in range(T):
        dp[start][1] = (T - 1 - start) * terminal_value

    for count in range(2, T_prime + 1):
        for start in range(T - count + 1):
            for next_start in range(start + 1, T - count + 2):
                candidate = (
                    dp[next_start][count - 1]
                    + (next_start - start - 1)
                    * float(mean_series[next_start])
                )
                if candidate > dp[start][count] or next_start == start + 1:
                    dp[start][count] = candidate
                    parent[start][count] = next_start

    current = max(
        range(T - T_prime + 1),
        key=lambda start: (
            dp[start][T_prime] + start * float(mean_series[start])
        ),
    )
    timesteps = [current]
    for count in range(T_prime, 1, -1):
        current = parent[current][count]
        timesteps.append(current)
    return timesteps


def build_schedule_map(
    T: int,
    T_prime: int,
    mean_series: Optional[Sequence[float]] = None,
) -> dict[str, set[int]]:
    schedules = heuristic_schedules(T, T_prime, mean_series)
    if mean_series is not None:
        schedules = {
            "vista": set(compute_vista_schedule(T, T_prime, mean_series)),
            **schedules,
        }
    return schedules
