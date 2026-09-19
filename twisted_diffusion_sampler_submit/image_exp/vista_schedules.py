"""Guidance schedules in the paper's reverse-time convention.

The paper indexes states as ``x_0, ..., x_T`` and reverse transitions as
``p_{t|t+1}``, so a schedule contains target-state indices
``t in {0, ..., T - 1}``.  ``t=T-1`` is the first denoising transition
executed from the prior and ``t=0`` is the last transition into a clean image.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set


# Exponents k offered for the (1 - t/T)^k weighted-VISTA objective.
WEIGHT_POWERS = (1, 2, 3, 20)
VALUE_POLICIES = (
    "vista",
    "top-V",
    "top-dV",
    "weighted-vista",
) + tuple(f"weighted-vista-{power}" for power in WEIGHT_POWERS if power != 1)
HEURISTIC_POLICIES = (
    "interval-1",
    "interval-2",
    "interval-3",
    "interval-4",
    "interval-5",
    "uniform",
)
SPARSE_POLICIES = HEURISTIC_POLICIES + VALUE_POLICIES
POLICIES = ("full",) + SPARSE_POLICIES


def _validate_budget(T: int, T_prime: int) -> None:
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if not 1 <= T_prime <= T:
        raise ValueError(
            f"T_prime must satisfy 1 <= T_prime <= T={T}, got {T_prime}"
        )


def _validate_value_series(T: int, values: Sequence[float]) -> None:
    if len(values) != T + 1:
        raise ValueError(
            f"values must contain [V_0, ..., V_T] (length {T + 1}); "
            f"got length {len(values)}"
        )
    invalid = [
        index
        for index, value in enumerate(values)
        if not math.isfinite(float(value))
    ]
    if invalid:
        raise ValueError(f"values contain non-finite entries at {invalid}")


def normalize_policy_name(policy: str) -> str:
    """Normalize CLI-friendly aliases while keeping one canonical spelling."""
    key = policy.strip()
    compact = key.lower().replace("_", "-").replace(" ", "")
    aliases = {
        "full": "full",
        "all": "full",
        "allstep": "full",
        "allsteps": "full",
        "uniform": "uniform",
        "uniformstep": "uniform",
        "uniformsteps": "uniform",
        "vista": "vista",
        "top-v": "top-V",
        "topv": "top-V",
        "top-dv": "top-dV",
        "topdv": "top-dV",
        "weighted-vista": "weighted-vista",
        "weightedvista": "weighted-vista",
    }
    for power in WEIGHT_POWERS:
        if power == 1:
            continue
        aliases[f"weighted-vista-{power}"] = f"weighted-vista-{power}"
        aliases[f"weightedvista{power}"] = f"weighted-vista-{power}"
    for k in range(1, 6):
        aliases[f"interval{k}"] = f"interval-{k}"
        aliases[f"interval-{k}"] = f"interval-{k}"
    canonical = aliases.get(compact, key)
    if canonical not in POLICIES:
        raise ValueError(
            f"unknown policy {policy!r}; choose one of {', '.join(POLICIES)}"
        )
    return canonical


def interval_schedule(T: int, T_prime: int, k: int) -> Set[int]:
    """Return ``range(s_k, s_k + T')`` exactly as specified by the user."""
    _validate_budget(T, T_prime)
    if k not in range(1, 6):
        raise ValueError(f"k must be one of 1, 2, 3, 4, 5; got {k}")
    s_k = round((T - T_prime) * (5 - k) / 4)
    return set(range(s_k, s_k + T_prime))


def uniform_schedule(T: int, T_prime: int) -> Set[int]:
    """Uniform schedule with both endpoints, using Python's ``round``."""
    _validate_budget(T, T_prime)
    if T_prime == 1:
        # The formula has a zero denominator.  The clean endpoint is the
        # deterministic convention used throughout the VISTA codebases.
        return {0}
    return {
        round((T - 1) / (T_prime - 1) * k)
        for k in range(T_prime)
    }


def _top_indices(scores: Sequence[float], count: int) -> Set[int]:
    # Smaller paper-t wins exact ties, which makes schedules reproducible.
    ranked = sorted(
        range(len(scores)),
        key=lambda timestep: (-float(scores[timestep]), timestep),
    )
    return set(ranked[:count])


def top_value_schedule(
    T: int, T_prime: int, values: Sequence[float]
) -> Set[int]:
    _validate_budget(T, T_prime)
    _validate_value_series(T, values)
    return _top_indices(values[:T], T_prime)


def top_delta_value_schedule(
    T: int, T_prime: int, values: Sequence[float]
) -> Set[int]:
    """Select largest ``V_hat_t - V_hat_{t+1}`` for ``t=0,...,T-1``."""
    _validate_budget(T, T_prime)
    _validate_value_series(T, values)
    deltas = [
        float(values[timestep]) - float(values[timestep + 1])
        for timestep in range(T)
    ]
    return _top_indices(deltas, T_prime)


def vista_schedule(
    T: int, T_prime: int, values: Sequence[float]
) -> Set[int]:
    """Solve Eq. (12) with Algorithm 6 from the VISTA paper.

    This implements the line-16 start-state maximization and does not force
    either endpoint into the schedule.
    """
    _validate_budget(T, T_prime)
    _validate_value_series(T, values)
    if T_prime == T:
        return set(range(T))

    negative_infinity = float("-inf")
    # dp[count][start] represents paper dp[start][count].
    dp = [
        [negative_infinity for _ in range(T)]
        for _ in range(T_prime + 1)
    ]
    parent: List[List[int | None]] = [
        [None for _ in range(T)]
        for _ in range(T_prime + 1)
    ]

    for start in range(T):
        dp[1][start] = (T - 1 - start) * float(values[T])

    for count in range(2, T_prime + 1):
        for start in range(0, T - count + 1):
            best_value = negative_infinity
            best_next = None
            for next_start in range(start + 1, T - count + 2):
                candidate = (
                    dp[count - 1][next_start]
                    + (next_start - start - 1)
                    * float(values[next_start])
                )
                # Strict comparison matches Algorithm 6. Iteration in
                # ascending order gives deterministic smaller-t tie-breaking.
                if candidate > best_value:
                    best_value = candidate
                    best_next = next_start
            dp[count][start] = best_value
            parent[count][start] = best_next

    best_objective = negative_infinity
    first_t = None
    for start in range(0, T - T_prime + 1):
        objective = (
            dp[T_prime][start] + start * float(values[start])
        )
        if objective > best_objective:
            best_objective = objective
            first_t = start

    if first_t is None:
        raise RuntimeError("VISTA dynamic programming found no schedule")

    selected = [first_t]
    current = first_t
    for count in range(T_prime, 1, -1):
        next_t = parent[count][current]
        if next_t is None:
            raise RuntimeError(
                "VISTA dynamic-programming parent table is incomplete"
            )
        selected.append(next_t)
        current = next_t
    if len(selected) != T_prime or len(set(selected)) != T_prime:
        raise RuntimeError(f"invalid VISTA schedule reconstructed: {selected}")
    return set(selected)


def policy_weight_power(policy: str) -> int | None:
    """Return k for weighted-VISTA policies, otherwise ``None``."""
    policy = normalize_policy_name(policy)
    if policy == "weighted-vista":
        return 1
    if policy.startswith("weighted-vista-"):
        return int(policy.rsplit("-", 1)[1])
    return None


def timestep_weights(T: int, power: int | None = None) -> List[float]:
    """Weights for omitted paper-time transitions.

    Paper time runs from ``T-1`` (first/noisiest reverse transition) to ``0``
    (last/cleanest transition).  Thus ``(1 - t/T)^k`` deliberately gives the
    largest weight to the late, clean-image portion of the reverse path.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if power is None:
        return [1.0] * T
    if power not in WEIGHT_POWERS:
        raise ValueError(
            f"power must be one of {', '.join(map(str, WEIGHT_POWERS))}; "
            f"got {power}"
        )
    return [(1.0 - timestep / T) ** power for timestep in range(T)]


def weighted_vista_schedule(
    T: int, T_prime: int, values: Sequence[float], power: int
) -> Set[int]:
    """Maximize ``sum_{t not in G} lambda(t) V_hat_{ceil_G(t)}`` exactly.

    Unlike multiplying the value curve by ``lambda``, this applies the weight
    to the *omitted timestep* t, as required by weighted total deviation.
    """
    _validate_budget(T, T_prime)
    _validate_value_series(T, values)
    if T_prime == T:
        return set(range(T))
    weights = timestep_weights(T, power)
    prefix = [0.0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    def weight_sum(start: int, stop: int) -> float:
        return prefix[stop] - prefix[start]

    negative_infinity = float("-inf")
    dp = [[negative_infinity for _ in range(T)] for _ in range(T_prime + 1)]
    parent: List[List[int | None]] = [[None for _ in range(T)] for _ in range(T_prime + 1)]
    for start in range(T):
        # After the final guided transition, ceil_G(t)=T.
        dp[1][start] = weight_sum(start + 1, T) * float(values[T])
    for count in range(2, T_prime + 1):
        for start in range(0, T - count + 1):
            for next_start in range(start + 1, T - count + 2):
                candidate = (
                    dp[count - 1][next_start]
                    + weight_sum(start + 1, next_start) * float(values[next_start])
                )
                if candidate > dp[count][start]:
                    dp[count][start] = candidate
                    parent[count][start] = next_start
    first_t = max(
        range(0, T - T_prime + 1),
        key=lambda start: dp[T_prime][start] + weight_sum(0, start) * float(values[start]),
    )
    selected = [first_t]
    current = first_t
    for count in range(T_prime, 1, -1):
        next_t = parent[count][current]
        if next_t is None:
            raise RuntimeError("weighted VISTA dynamic-programming parent table is incomplete")
        selected.append(next_t)
        current = next_t
    return set(selected)


def schedule_objective(
    T: int,
    schedule: Iterable[int],
    values: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Return ``sum_{t not in G} w(t) V_hat_{ceil_G(t)}``."""
    _validate_value_series(T, values)
    if weights is None:
        weights = timestep_weights(T)
    if len(weights) != T or not all(math.isfinite(float(w)) for w in weights):
        raise ValueError("weights must contain T finite entries")
    guided = _validated_schedule(T, schedule)
    active_with_terminal = sorted(guided | {T})
    total = 0.0
    for timestep in range(T):
        if timestep in guided:
            continue
        ceiling = min(active for active in active_with_terminal if active >= timestep)
        total += float(weights[timestep]) * float(values[ceiling])
    return total


def _validated_schedule(T: int, schedule: Iterable[int]) -> Set[int]:
    guided: Set[int] = set()
    for raw_timestep in schedule:
        if isinstance(raw_timestep, bool) or int(raw_timestep) != raw_timestep:
            raise TypeError(
                f"schedule entries must be integers, got {raw_timestep!r}"
            )
        guided.add(int(raw_timestep))
    invalid = sorted(t for t in guided if t < 0 or t >= T)
    if invalid:
        raise ValueError(f"schedule entries must lie in [0, {T}); got {invalid}")
    return guided


def empirical_utd(
    T: int,
    schedule: Iterable[int],
    values: Sequence[float],
    reward_max: float = 1.0,
    weights: Sequence[float] | None = None,
) -> float:
    """Evaluate the empirical UTD upper bound from paper Eq. (9).

    For the MNIST reward ``R(x_0)=p(y|x_0)`` with ``alpha=1``,
    ``reward_max=1``.
    """
    if not math.isfinite(float(reward_max)) or reward_max <= 0:
        raise ValueError(
            f"reward_max must be positive and finite, got {reward_max}"
        )
    _validate_value_series(T, values)
    if weights is None:
        weights = timestep_weights(T)
    if len(weights) != T or not all(math.isfinite(float(w)) for w in weights):
        raise ValueError("weights must contain T finite entries")
    guided = _validated_schedule(T, schedule)
    active_with_terminal = sorted(guided | {T})
    bound_sum = 0.0
    for timestep in range(T):
        if timestep in guided:
            continue
        ceiling = min(active for active in active_with_terminal if active >= timestep)
        bound_sum += float(weights[timestep]) * (
            1.0 - float(values[ceiling]) / float(reward_max)
        )
    return 2.0 * bound_sum / T


def build_schedule(
    policy: str,
    T: int,
    T_prime: int,
    values: Sequence[float] | None = None,
) -> Set[int]:
    policy = normalize_policy_name(policy)
    _validate_budget(T, T_prime)
    if policy == "full":
        return set(range(T))
    if policy == "uniform":
        return uniform_schedule(T, T_prime)
    if policy.startswith("interval-"):
        return interval_schedule(T, T_prime, int(policy.rsplit("-", 1)[1]))
    if values is None:
        raise ValueError(f"policy {policy} requires warmup value estimates")
    if policy == "vista":
        return vista_schedule(T, T_prime, values)
    power = policy_weight_power(policy)
    if power is not None:
        return weighted_vista_schedule(T, T_prime, values, power)
    if policy == "top-V":
        return top_value_schedule(T, T_prime, values)
    if policy == "top-dV":
        return top_delta_value_schedule(T, T_prime, values)
    raise AssertionError(f"unhandled canonical policy {policy}")


def build_schedule_map(
    T: int,
    T_prime: int,
    values: Sequence[float],
    policies: Iterable[str] = POLICIES,
) -> Dict[str, Set[int]]:
    result: Dict[str, Set[int]] = {}
    for raw_policy in policies:
        policy = normalize_policy_name(raw_policy)
        if policy in result:
            raise ValueError(f"duplicate policy after normalization: {policy}")
        result[policy] = build_schedule(
            policy=policy,
            T=T,
            T_prime=T_prime,
            values=values,
        )
    return result


def execution_order(schedule: Iterable[int]) -> List[int]:
    """Return paper-t entries in the order reverse diffusion executes them."""
    return sorted((int(t) for t in schedule), reverse=True)


def paper_t_to_execution_index(timestep: int, T: int) -> int:
    """Map paper ``t`` to chronological zero-based denoising-step index."""
    if timestep < 0 or timestep >= T:
        raise ValueError(f"timestep must lie in [0, {T}); got {timestep}")
    return T - 1 - timestep
