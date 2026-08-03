'''
VISTA-SMC (Value-Informed Schedule Trimming for Accelerated SMC guidance), Section 4 of the
paper. Builds a budgeted guided-step set G exactly like the heuristics in `schedules.py`
(First/Last/Middle/Uniform), except G is chosen from a few full-guidance warmup runs instead of
a fixed hand-designed rule.

Usage, drop-in next to `sampling.schedules.build_schedule`:

    from sampling.vista import build_vista_schedule
    guided_set, Vhat = build_vista_schedule(algo, T, budget, num_samples=N, num_warmup_runs=3)
    algo.inference(num_samples=N, guided_set=guided_set, ...)   # same call as any other schedule

`algo` is any `SMC_Base` or `SMC_Grad` instance (already constructed with its net/reward/alpha/etc,
same object you'd otherwise call `.inference(guided_set=build_schedule(...))` on). The warmup runs
use that same instance under full guidance (`guided_set=None`), so they use the same proposal
family, reward model, and alpha as the eventual sparse runs -- required for the schedule to be
valid for those runs (Section 4.3 / the "same model, reward, budget, and proposal family" caveat).
'''
import math
import operator

import torch


@torch.no_grad()
def estimate_warmup_values(algo, T, num_samples, num_warmup_runs=3, verbose=False):
    '''
    Run `num_warmup_runs` independent full-guidance generations (Algorithm 4) with `algo` and
    average the value estimates V_t(x_t^(i)) seen at every timestep over all particles and runs
    (Eq. 11): Vhat_t ~= E_{x_t ~ pi*_t}[V_t(x_t)].

    Full guidance (`guided_set=None`) computes V_next at every t_next = T-1, ..., 0 plus V_prev at
    t = T up front, so a single pass already visits every t in {0, ..., T} -- `value_callback`
    (see SMC_Base.inference / SMC_Grad.inference) just taps those values as they're computed, no
    extra rollout cost beyond running full guidance `num_warmup_runs` times.

    Returns a tensor `Vhat` of shape (T + 1,), Vhat[t] = \\hat V_t for t = 0, ..., T.
    '''
    sums = torch.zeros(T + 1, device=algo.device)
    counts = torch.zeros(T + 1, device=algo.device)

    def callback(t, V, w):
        del w
        # Eq. (11) is the literal unweighted average over the M warmup runs and their N
        # particles: Vhat_t = (MN)^-1 sum_m sum_i Vhat_t^(i,m).  In particular, do not replace
        # it with an importance-weighted estimator when adaptive/partial resampling leaves
        # non-uniform internal weights; that would optimize a different schedule statistic.
        sums[t] += V.detach().sum()
        counts[t] += V.numel()

    for _ in range(num_warmup_runs):
        algo.inference(num_samples=num_samples, verbose=verbose, guided_set=None,
                       value_callback=callback)

    missing = (counts == 0).nonzero().flatten().tolist()
    if missing:
        raise RuntimeError(f"warmup never visited timesteps {missing}; is T consistent with "
                            f"algo.net.timestep?")
    return sums / counts


def time_weighted_lambda(T, k):
    '''
    lambda(t) = (1 - t/T)^k for t = 0, ..., T-1.

    Weights the per-step total-deviation term TD(G) := (1/T) sum_t lambda(t) ||pi_t* - pi_t^G||_1
    (and its dp surrogate Shat(G), see `solve_schedule_dp`/`evaluate_schedule`) so that steps near
    t=0 -- the *last* steps executed by the reverse loop (fine-grained correction, see the t
    convention in sampling/schedules.py's module docstring) -- matter more than steps near t=T
    (executed first, coarse/global steering). For ReMDM/UDLM/continuous-time guidance, matching
    the back half of the reverse path was found to matter more than matching the front half,
    unlike (absorbing) MDLM where the unweighted Eq. (12) (lam=None, equivalent to k=0 here) is
    used. k in {1, 2, 3} controls how sharply the weight concentrates near t=0; larger k discounts
    early steps more aggressively. k=0 reduces to lambda(t) = 1 for all t, i.e. the original
    unweighted objective.
    '''
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return [(1.0 - t / T) ** k for t in range(T)]


def solve_schedule_dp(Vhat, T, budget, lam=None):
    '''
    Solve Eq. (12) exactly by dynamic programming (Algorithm 6):
    find G subset {0, ..., T-1} with |G| = budget maximizing

        Shat(G) = sum_{t in {0,...,T-1} \\ G} lambda(t) * Vhat[ceil(t)],
        ceil(t) := min{s in G union {T} | s >= t}.

    `lam`, if given, is a length-T sequence lambda(t) for t = 0, ..., T-1 (e.g. from
    `time_weighted_lambda`); `lam=None` (default) uses lambda(t) = 1 for all t, recovering the
    paper's original unweighted Eq. (12).

    Eq. (12) optimizes over every size-``budget`` subset of ``{0, ..., T-1}``; it does not impose
    ``0 in G``.  Letting ``s=min(G)``, the unguided steps ``0, ..., s-1`` contribute
    ``LambdaSum(0, s-1) * Vhat[s]`` (``= s * Vhat[s]`` when lam is uniform).  Algorithm 6 therefore
    selects its smallest active point by maximizing ``dp[budget][s] + LambdaSum(0, s-1)*Vhat[s]``,
    exactly as implemented below.

    Vhat: length-(T+1) sequence, Vhat[t] for t = 0, ..., T (e.g. from `estimate_warmup_values`).
    Returns a `set[int]` schedule G -- pass straight as `guided_set=` to
    `SMC_Base.inference` / `SMC_Grad.inference`.
    '''
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    budget = max(1, min(int(budget), T))
    Vhat = [float(v) for v in Vhat]
    if len(Vhat) != T + 1:
        raise ValueError(f"Vhat must have length T+1={T + 1}, got {len(Vhat)}")

    lam = [1.0] * T if lam is None else [float(x) for x in lam]
    if len(lam) != T:
        raise ValueError(f"lam must have length T={T}, got {len(lam)}")
    # cum[i] = sum_{t=0}^{i-1} lam[t], so LambdaSum(a, b) := sum_{t=a}^{b} lam[t]
    # = cum[b+1] - cum[a] for a <= b, else 0 (empty range).
    cum = [0.0] * (T + 1)
    for t in range(T):
        cum[t + 1] = cum[t] + lam[t]

    def lambda_sum(a, b):
        return cum[b + 1] - cum[a] if a <= b else 0.0

    NEG_INF = float("-inf")
    # dp[i][s]: best achievable sum_{t>s} lambda(t) * Vhat[ceil(t)] given a chain of i active
    # points whose smallest element is s (s = t_i in the paper's G' = {s = t_i < ... < t_1 < T}).
    dp = [[NEG_INF] * T for _ in range(budget + 1)]
    parent = [[None] * T for _ in range(budget + 1)]

    for s in range(T):
        dp[1][s] = lambda_sum(s + 1, T - 1) * Vhat[T]

    for i in range(2, budget + 1):
        for s in range(0, T - i + 1):
            best_val, best_s2 = NEG_INF, None
            for s2 in range(s + 1, T - i + 2):
                if dp[i - 1][s2] == NEG_INF:
                    continue
                c = dp[i - 1][s2] + lambda_sum(s + 1, s2 - 1) * Vhat[s2]
                if c > best_val:
                    best_val, best_s2 = c, s2
            dp[i][s] = best_val
            parent[i][s] = best_s2

    # Algorithm 6 line 16: t_{T'} <- argmax_s (dp[s][T'] + LambdaSum(0, s-1) * Vhat[s]).
    best_val, best_s = NEG_INF, None
    for s in range(0, T - budget + 1):
        if dp[budget][s] == NEG_INF:
            continue
        val = dp[budget][s] + lambda_sum(0, s - 1) * Vhat[s]
        if val > best_val:
            best_val, best_s = val, s

    if best_s is None:
        raise ValueError(f"no valid schedule with budget={budget} for T={T}")

    schedule = [best_s]
    s, i = best_s, budget
    while i > 1:
        s2 = parent[i][s]
        schedule.append(s2)
        s, i = s2, i - 1

    return set(schedule)


def _as_finite_float_sequence(values, expected_length, name):
    try:
        values = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must have length {expected_length}, got {len(values)}")
    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain scalar numeric values") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _normalize_guided_set(G, T):
    try:
        raw_steps = list(G)
    except TypeError as exc:
        raise TypeError("G must be an iterable of integer timesteps") from exc

    guided_set = set()
    for step in raw_steps:
        if _is_boolean_scalar(step):
            raise TypeError("G must contain integer timesteps, not booleans")
        try:
            step = operator.index(step)
        except TypeError as exc:
            raise TypeError(
                f"G must contain integer timesteps, got {step!r}") from exc
        if not 0 <= step < T:
            raise ValueError(f"each timestep in G must be in [0, {T - 1}], got {step}")
        guided_set.add(step)
    return guided_set


def _is_boolean_scalar(value):
    if isinstance(value, bool):
        return True
    dtype = getattr(value, "dtype", None)
    return dtype is not None and str(dtype).rsplit(".", 1)[-1] == "bool"


def compute_shat(Vhat, T, G, lam=None):
    '''
    Compute the empirical VISTA objective from Eq. (12):

        Shat(G) = sum_{t in {0,...,T-1} \\ G} Vhat[ceil(t)],
        ceil(t) := min{s in G union {T} | s >= t}.

    ``Vhat`` must contain the linear-space (exponentiated) value estimates from Eq. (11), not
    raw rewards or log-values, and therefore has length ``T + 1``. The sentinel value ``Vhat[T]``
    is used after the largest active timestep; ``T`` itself must not be included in ``G``.

    For sorted active steps g_1 < ... < g_K, with g_0 = -1 and g_{K+1} = T, the implementation
    uses the equivalent gap form

        sum_j (g_j - g_{j-1} - 1) * Vhat[g_j],

    so every skipped timestep is counted exactly once without repeatedly searching ``G``.
    ``lam`` optionally supplies the repository's time-weighted extension; ``lam=None`` is the
    paper's unweighted Eq. (12). The return value is a Python float.
    '''
    if _is_boolean_scalar(T):
        raise TypeError("T must be an integer")
    try:
        T = operator.index(T)
    except TypeError as exc:
        raise TypeError("T must be an integer") from exc
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")

    Vhat = _as_finite_float_sequence(Vhat, T + 1, "Vhat")
    G = _normalize_guided_set(G, T)

    if lam is None:
        lam = [1.0] * T
    else:
        lam = _as_finite_float_sequence(lam, T, "lam")

    # For an active boundary s following previous_s, the skipped interval is
    # {previous_s + 1, ..., s - 1} and all its terms use Vhat[s]. Sum the weighted terms inside
    # each disjoint gap: global prefix differences can erase small gaps, while factoring out
    # Vhat[s] can overflow the raw weight sum even when every weighted term is representable.
    terms = []
    previous_s = -1
    for s in sorted(G) + [T]:
        terms.append(math.fsum(
            weight * Vhat[s] for weight in lam[previous_s + 1:s]))
        previous_s = s
    return math.fsum(terms)


def evaluate_schedule(Vhat, T, G, lam=None):
    '''Backward-compatible name for :func:`compute_shat`.'''
    return compute_shat(Vhat, T, G, lam=lam)


def build_vista_schedule(algo, T, budget, num_samples, num_warmup_runs=3, verbose=False):
    '''
    End-to-end VISTA-SMC schedule construction (Section 4.3): run the warmup, estimate
    \\hat V_t (Eq. 11), then solve the budgeted DP (Eq. 12 / Algorithm 6).

    Returns `(guided_set, Vhat)`. `guided_set` is a `set[int]`, ready to use exactly like
    `sampling.schedules.build_schedule(...)`'s return value:

        guided_set, _ = build_vista_schedule(algo, T, budget, num_samples=N)
        algo.inference(num_samples=N, guided_set=guided_set)

    The warmup cost (num_warmup_runs full-guidance generations) is a one-time cost, amortized
    across all later sparse generations reusing the same `guided_set` (Appendix A.6).
    '''
    Vhat = estimate_warmup_values(algo, T, num_samples, num_warmup_runs=num_warmup_runs, verbose=verbose)
    guided_set = solve_schedule_dp(Vhat, T, budget)
    return guided_set, Vhat
