'''
Sparse-guidance schedule generators, producing a guided-step set G subset of {0, ..., T-1} with
|G| <= budget for use with SMC_Base.inference(guided_set=...): Section 5's heuristic
Interval/Uniform baselines, VISTA-SMC's learned schedule (Section 4), and the warmup-value
heuristics Top-dV/Top-V.

Indexing matches Definition 4.1 / Algorithm 2: t indexes the *destination* of each reverse
step (the reverse loop runs t = T-1, T-2, ..., 0), so guiding high t means guiding the steps
executed first (closest to T, coarse/global steering while most of the sequence is still
masked); guiding low t means guiding the steps executed last (closest to 0, fine-grained
correction). "interval1" guides the top of the range (formerly called "First"); "interval5"
guides the bottom (formerly "Last"); interval2-4 slide the guided window linearly in between
(interval3 landing near the middle, formerly "Middle").
'''
import functools

import numpy as np

from .vista import solve_schedule_dp

NUM_INTERVALS = 5


def interval_schedule(T, budget, k, num_intervals=NUM_INTERVALS):
    '''
    Interval-k of `num_intervals` (default 5): a `budget`-sized contiguous window of
    {0, ..., T-1}, its start position slid linearly from the top (k=1: range(T-budget, T), the
    old "First") to the bottom (k=num_intervals: range(0, budget), the old "Last") in
    num_intervals-1 equal steps.
    '''
    if not (1 <= k <= num_intervals):
        raise ValueError(f"k must be in [1, {num_intervals}], got {k}")
    budget = max(0, min(budget, T))
    span = T - budget
    start = round(span * (num_intervals - k) / (num_intervals - 1)) if num_intervals > 1 else 0
    return set(range(start, start + budget))


def uniform_schedule(T, budget):
    '''
    Evenly-spaced steps across {0, ..., T-1}. Rounding can occasionally collapse two nearby
    fractional positions onto the same integer, so the returned set may have fewer than
    `budget` elements for small T/budget ratios -- matches how this baseline is commonly
    implemented elsewhere (e.g. simple np.linspace(...).round() sparse-guidance baselines).
    '''
    budget = max(0, min(budget, T))
    if budget <= 0:
        return set()
    if budget == 1:
        return {T // 2}
    idx = np.linspace(0, T - 1, budget)
    return set(int(round(i)) for i in idx)


def make_vista_schedule(V_hat, budget, lam=None):
    '''
    VISTA-SMC (Algorithm 6 / Eq. 12): given a precomputed warmup value estimate `V_hat`
    (length T+1, e.g. from `sampling.vista.estimate_warmup_values`), solves for the
    approximation-optimal guided-step set at this `budget` via `solve_schedule_dp`.

    `lam`, if given, is a length-T time-weighting lambda(t) (e.g. from
    `sampling.vista.time_weighted_lambda`) applied to Eq. (12)'s per-step term; `lam=None`
    (default) recovers the original unweighted objective -- see `solve_schedule_dp`.

    Unlike the heuristics above, VISTA needs data (V_hat), not just (T, budget) -- computing
    that data means running full-guidance warmup passes against an actual model/reward, which
    isn't something a schedule generator here can do on its own (it isn't a pure function of T
    and budget the way Interval/Uniform are). So this function's signature takes V_hat in place
    of T (T is recovered as len(V_hat) - 1); callers first get V_hat from
    `sampling.vista.estimate_warmup_values(algo, T, ...)`, once, then reuse it across every
    budget they want a schedule for -- see `build_schedule(..., V_hat=...)` below and
    scripts/sparse_schedule_experiment*.py's `build_schedule_for` for the full warmup-then-solve
    flow.
    '''
    T = len(V_hat) - 1
    return solve_schedule_dp(V_hat, T, budget, lam=lam)


def make_top_dv_schedule(V_hat, budget):
    '''
    Top-dV: given the same warmup value estimate `V_hat` VISTA uses (length T+1, from
    `sampling.vista.estimate_warmup_values` after M full-guidance warmup runs), rank each
    reverse step t in {0, ..., T-1} by its average value increment

        dv(t) := Vhat[t] - Vhat[t+1] ~= E[V_t(x_t) - V_{t+1}(x_{t+1})]

    -- exact by linearity of expectation, since Vhat[t] is already the Eq. (11) particle average
    of V_t(x_t), so no separate paired-sample rollout is needed beyond the warmup VISTA already
    runs -- and guide the `budget` steps with the largest dv(t).
    '''
    T = len(V_hat) - 1
    budget = max(0, min(budget, T))
    dv = [float(V_hat[t]) - float(V_hat[t + 1]) for t in range(T)]
    order = sorted(range(T), key=lambda t: dv[t], reverse=True)
    return set(order[:budget])


def make_top_v_schedule(V_hat, budget):
    '''
    Top-V: guide the `budget` steps t in {0, ..., T-1} with the largest warmup value estimate
    Vhat[t] itself (Eq. 11) -- unlike Top-dV (which ranks by the *change* in value) or VISTA
    (which solves the budgeted Shat(G) trade-off, see `make_vista_schedule`).
    '''
    T = len(V_hat) - 1
    budget = max(0, min(budget, T))
    order = sorted(range(T), key=lambda t: float(V_hat[t]), reverse=True)
    return set(order[:budget])


SCHEDULES = {
    "uniform": uniform_schedule,
    **{f"interval{k}": functools.partial(interval_schedule, k=k)
       for k in range(1, NUM_INTERVALS + 1)},
}

# Schedules that, like "vista", need a precomputed warmup V_hat rather than being a pure
# function of (T, budget) -- see `make_vista_schedule`'s docstring for why.
WARMUP_SCHEDULES = {
    "top_dv": make_top_dv_schedule,
    "top_v": make_top_v_schedule,
}


def build_schedule(name, T, budget, V_hat=None, lam=None):
    if name == "full":
        return set(range(T))
    if name == "vista" or name in WARMUP_SCHEDULES:
        if V_hat is None:
            raise ValueError(f"build_schedule({name!r}, ...) requires V_hat (length T+1) -- see "
                              "sampling.vista.estimate_warmup_values / make_vista_schedule")
        if len(V_hat) != T + 1:
            raise ValueError(f"V_hat must have length T+1={T + 1}, got {len(V_hat)}")
        if name == "vista":
            return make_vista_schedule(V_hat, budget, lam=lam)
        return WARMUP_SCHEDULES[name](V_hat, budget)
    if name not in SCHEDULES:
        raise ValueError(f"Unknown schedule '{name}', expected one of "
                          f"{list(SCHEDULES) + ['full', 'vista'] + list(WARMUP_SCHEDULES)}")
    return SCHEDULES[name](T, budget)
