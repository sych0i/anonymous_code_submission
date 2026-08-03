"""Continuous-DDPM VISTA sparse Sequential Monte Carlo sampler.

The paper index convention used throughout this module is

    x_T -> x_{T-1} -> ... -> x_0,

where a schedule entry ``t`` controls the transition
``p_{t|t+1}(x_t | x_{t+1})``.  Consequently, sampling executes paper
timesteps ``T - 1, ..., 0``.  This convention is deliberately kept separate
from the common DDPM convention in which a loop variable names the source
state.

At a guided step, the proposal has the same diagonal covariance as the
unconditional model.  The repository's variance-preserving TDS implementation
first shifts the clean prediction,

    x0_twisted = x0_pred + (1-alpha_bar_{t+1})/sqrt(alpha_bar_{t+1})
                              * grad log p(y | x_{t+1}),

clips that prediction to ``[-1, 1]`` by default, and recomputes the DDPM
posterior mean.  Without clipping this is algebraically
``p_mean + beta/sqrt(alpha) * grad log p(y|x)``.  Reusing the learned reverse
variance as the score-shift coefficient, or omitting the configured clean-state
clipping, changes the upstream proposal family whose schedule VISTA trims.

At an unguided step the sampler uses the unconditional transition exactly.
The incremental guided weight is

    log w_t = log w_anchor + log(p/q) + log(V_t/V_anchor).

The initial state x_T is also value weighted, as in Algorithms 4--5 of the
VISTA paper.  Resampling is only performed at the initial state and active
steps, and uses the configured adaptive partial rule even at ``x_0``; there is
    no extra unconditional full resampling after Algorithm 5.  The returned value
    statistic is computed on the post-resampling particle system, using its
    normalized weights (a plain mean when the weights are equal), to estimate
    ``E_{pi_t^*}[V_t]``.  In a sparse run, inactive entries are NaN; a full guidance
    warmup produces all entries ``V_0, ..., V_T``.

The proposal potential and the value estimator are deliberately separate.
The proposal follows the repository's original TDS construction: the clean
MNIST classifier is evaluated on the unconditional model's differentiable
``pred_xstart(x_{t+1})`` estimate.  This is the project's deterministic
approximation to ``log p(y | x_{t+1})`` in the proposal requested above.

Two explicitly named value estimators are supported for the importance weights
and warmup ``V_hat`` measurements:

``one_shot``
    The paper-structured continuous TDS estimator used by default.  Make one
    denoiser call at the current noisy state, treat ``pred_xstart`` as a delta
    approximation to the model's unavailable direct ``p(x_0 | x_t)`` output,
    and apply the clean classifier.  The discrete paper samples directly from
    its denoiser's categorical ``x0_probs``; it does not run a reverse chain.
    Since this DDPM exposes only a point prediction, ``J`` is fixed to one and
    the discrete paper's exact UTD guarantee does not transfer unchanged.

``one_shot_gaussian``
    An explicit stochastic continuous extension.  Make one denoiser call,
    use ``pred_xstart`` as the clean-state mean, and draw
    ``J=num_rollouts`` direct clean candidates with variance
    ``1 - alpha_cumprod_t``.  This is the posterior variance obtained from a
    unit-Gaussian prior approximation to the forward process.  It preserves
    the paper's one-shot sampling and ``J`` reward-evaluation structure, but
    it is not an exact DDPM ``p(x_0 | x_t)`` distribution or a proven VISTA
    bound.  The TDS proposal itself remains the deterministic classifier
    potential on ``pred_xstart``.

``ancestral``
    An optional, computationally expensive diagnostic ablation.  Draw
    ``J=num_rollouts`` full reverse trajectories from the current state to
    x_0.  This estimates a chain-defined conditional value, but it is not the
    one-shot estimator or cost model used by the VISTA paper.

The final DDPM transition is deterministic by default, matching the reference
sampler's x_1 -> x_0 convention.  ``final_transition='stochastic'`` remains
available only for explicit ablations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional

import torch
import torch.nn.functional as F


Tensor = torch.Tensor

_ONE_SHOT_ESTIMATORS = frozenset({"one_shot", "one_shot_gaussian"})
_VALUE_ESTIMATORS = _ONE_SHOT_ESTIMATORS | frozenset({"ancestral"})
_FINAL_TRANSITIONS = frozenset({"stochastic", "legacy_deterministic"})


def _sum_nonbatch(value: Tensor) -> Tensor:
    """Sum all dimensions except the leading particle dimension."""

    if value.ndim < 1:
        raise ValueError("particle tensors must have a leading dimension")
    if value.ndim == 1:
        return value
    return value.flatten(start_dim=1).sum(dim=1)


def _normalize_log_weights(log_weights: Tensor) -> Tensor:
    """Normalize finite one-dimensional log weights."""

    if log_weights.ndim != 1:
        raise ValueError(
            f"log_weights must be one-dimensional, got {log_weights.shape}"
        )
    if torch.isnan(log_weights).any() or torch.isposinf(log_weights).any():
        raise FloatingPointError("log weights contain NaN or +Inf")
    log_normalizer = torch.logsumexp(log_weights, dim=0)
    if not torch.isfinite(log_normalizer):
        raise FloatingPointError("all particle weights are zero or non-finite")
    return log_weights - log_normalizer


def effective_sample_size(log_weights: Tensor) -> Tensor:
    """Return the standard ESS of a one-dimensional log-weight vector."""

    normalized = _normalize_log_weights(log_weights).exp()
    return normalized.square().sum().reciprocal()


def weighted_value_mean(log_values: Tensor, log_weights: Tensor) -> Tensor:
    """Compute ``sum_i normalized_weight_i * exp(log_value_i)`` stably."""

    if log_values.shape != log_weights.shape:
        raise ValueError(
            "log_values and log_weights must have the same shape, got "
            f"{log_values.shape} and {log_weights.shape}"
        )
    normalized = _normalize_log_weights(log_weights)
    return torch.logsumexp(normalized + log_values, dim=0).exp()


def gaussian_log_p_over_q(
    sample: Tensor,
    p_mean: Tensor,
    q_mean: Tensor,
    variance: Tensor,
) -> Tensor:
    """Exact per-particle ``log p(sample) - log q(sample)``.

    ``p`` and ``q`` are diagonal Gaussians with the same variance.  The
    delta form below avoids subtracting two large squared norms.  Dimensions
    with exactly zero variance are permitted only when both means coincide;
    they then represent the same deterministic kernel and contribute zero.
    """

    try:
        sample, p_mean, q_mean, variance = torch.broadcast_tensors(
            sample, p_mean, q_mean, variance
        )
    except RuntimeError as exc:
        raise ValueError("Gaussian parameters are not broadcast compatible") from exc
    if sample.ndim < 1:
        raise ValueError("Gaussian inputs need a particle dimension")
    if not (
        torch.isfinite(sample).all()
        and torch.isfinite(p_mean).all()
        and torch.isfinite(q_mean).all()
        and torch.isfinite(variance).all()
    ):
        raise FloatingPointError("Gaussian parameters contain non-finite values")
    if (variance < 0).any():
        raise ValueError("Gaussian variance must be non-negative")

    positive = variance > 0
    mean_delta = q_mean - p_mean
    singular_mismatch = (~positive) & (mean_delta != 0)
    if singular_mismatch.any():
        raise ValueError(
            "p and q have different means along a zero-variance dimension"
        )

    safe_variance = torch.where(positive, variance, torch.ones_like(variance))
    # log p - log q =
    #   -(x - mu_p) delta / var + 0.5 delta^2 / var.
    elementwise = (
        -(sample - p_mean) * mean_delta / safe_variance
        + 0.5 * mean_delta.square() / safe_variance
    )
    elementwise = torch.where(positive, elementwise, torch.zeros_like(elementwise))
    return _sum_nonbatch(elementwise)


def _randn_like(value: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    return torch.randn(
        value.shape,
        dtype=value.dtype,
        device=value.device,
        generator=generator,
    )


def systematic_resample_indices(
    normalized_weights: Tensor,
    num_samples: Optional[int] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Draw ordered ancestor indices by systematic resampling."""

    if normalized_weights.ndim != 1:
        raise ValueError("systematic resampling expects one weight vector")
    if torch.isnan(normalized_weights).any() or (normalized_weights < 0).any():
        raise ValueError("resampling weights must be finite and non-negative")
    total = normalized_weights.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("resampling weights must have positive finite mass")
    weights = normalized_weights / total
    count = len(weights) if num_samples is None else int(num_samples)
    if count <= 0:
        raise ValueError(f"num_samples must be positive, got {count}")

    offset = torch.rand(
        (),
        dtype=weights.dtype,
        device=weights.device,
        generator=generator,
    )
    positions = (
        offset + torch.arange(count, device=weights.device, dtype=weights.dtype)
    ) / count
    cdf = weights.cumsum(dim=0)
    cdf[-1] = 1.0
    return torch.searchsorted(cdf, positions, right=True).clamp_max(
        len(weights) - 1
    )


def adaptive_systematic_partial_resample(
    particles: Tensor,
    log_weights: Tensor,
    last_guided_log_values: Tensor,
    *,
    ess_threshold: float,
    partial_resample: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, bool, Tensor]:
    """Apply the adaptive partial-resampling rule used by VISTA experiments.

    If ESS is below ``ess_threshold * N``, the selected subset consists of
    ``floor(partial_resample / 2)`` highest-weight particles and the remaining
    lowest-weight particles, matching the Binary-MNIST VISTA reference.
    Ancestors are sampled systematically within this subset.  The selected
    subset's total normalized mass is split uniformly among its output slots;
    unselected particles retain their normalized weights.  Values follow
    exactly the same ancestor map.
    """

    if particles.ndim < 1:
        raise ValueError("particles need a leading particle dimension")
    num_particles = particles.shape[0]
    if log_weights.shape != (num_particles,):
        raise ValueError(
            f"expected {num_particles} log weights, got {log_weights.shape}"
        )
    if last_guided_log_values.shape != (num_particles,):
        raise ValueError(
            "last_guided_log_values must have shape "
            f"({num_particles},), got {last_guided_log_values.shape}"
        )
    if ess_threshold < 0:
        raise ValueError("ess_threshold must be non-negative")
    if partial_resample < 0:
        raise ValueError("partial_resample must be non-negative")

    normalized_log_weights = _normalize_log_weights(log_weights)
    ess = effective_sample_size(normalized_log_weights)
    identity = torch.arange(num_particles, device=particles.device)
    should_resample = (
        partial_resample > 0
        and ess < float(ess_threshold) * num_particles
    )
    if not bool(should_resample):
        return (
            particles,
            normalized_log_weights,
            last_guided_log_values,
            identity,
            False,
            ess,
        )

    subset_size = min(int(partial_resample), num_particles)
    high_count = subset_size // 2
    low_count = subset_size - high_count
    highest = torch.topk(
        normalized_log_weights, high_count, largest=True, sorted=True
    ).indices
    # Exclude selected high-weight entries before choosing the low-weight half.
    # This also keeps the subset unique for M=N and in exact-tie cases.
    eligible = normalized_log_weights.clone()
    eligible[highest] = torch.inf
    lowest = torch.topk(
        eligible, low_count, largest=False, sorted=True
    ).indices
    subset = torch.cat((highest, lowest))

    subset_log_weights = normalized_log_weights[subset]
    # Normalize within the subset in log space. Directly slicing exp(log W)
    # can turn every very-low-weight member into exact zero in float32.
    subset_weights = torch.softmax(subset_log_weights, dim=0)
    local_ancestors = systematic_resample_indices(
        subset_weights,
        subset_size,
        generator=generator,
    )
    ancestors = subset[local_ancestors]
    ancestor_map = identity.clone()
    ancestor_map[subset] = ancestors

    new_particles = particles[ancestor_map]
    new_values = last_guided_log_values[ancestor_map]

    log_subset_mass = torch.logsumexp(subset_log_weights, dim=0)
    uniform_subset_log_weight = log_subset_mass - math.log(subset_size)
    new_log_weights = normalized_log_weights.clone()
    new_log_weights[subset] = uniform_subset_log_weight
    new_log_weights = _normalize_log_weights(new_log_weights)
    return (
        new_particles,
        new_log_weights,
        new_values,
        ancestor_map,
        True,
        ess,
    )


def full_systematic_resample(
    particles: Tensor,
    log_weights: Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return an equal-weight terminal particle set."""

    if particles.ndim < 1:
        raise ValueError("particles need a leading particle dimension")
    num_particles = particles.shape[0]
    if log_weights.shape != (num_particles,):
        raise ValueError(
            f"expected {num_particles} log weights, got {log_weights.shape}"
        )
    weights = _normalize_log_weights(log_weights).exp()
    ancestors = systematic_resample_indices(
        weights, num_particles, generator=generator
    )
    equal_log_weights = torch.full_like(
        log_weights, -math.log(num_particles)
    )
    return particles[ancestors], equal_log_weights, ancestors


def _repeat_particle_kwargs(
    value: Any, num_particles: int, repeats: int
) -> Any:
    """Repeat particle-indexed model kwargs for ancestral J rollouts."""

    if isinstance(value, Tensor):
        if value.ndim > 0 and value.shape[0] == num_particles:
            return value.repeat_interleave(repeats, dim=0)
        return value
    if isinstance(value, Mapping):
        return {
            key: _repeat_particle_kwargs(item, num_particles, repeats)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _repeat_particle_kwargs(item, num_particles, repeats)
            for item in value
        )
    if isinstance(value, list):
        return [
            _repeat_particle_kwargs(item, num_particles, repeats)
            for item in value
        ]
    return value


@dataclass(frozen=True)
class _Transition:
    mean: Tensor
    variance: Tensor
    pred_xstart: Tensor


@dataclass(frozen=True)
class _PreparedOneShotStep:
    """Detached model output and proposal data for one noisy state.

    ``paper_t`` names the destination of the cached edge.  No autograd graph
    is retained: the cache can safely follow a particle ancestor map after
    adaptive resampling without keeping model or classifier parameters alive.
    """

    paper_t: int
    transition: _Transition
    log_value: Optional[Tensor]
    q_shift: Optional[Tensor]


class ContinuousVistaSMCSampler:
    """VISTA sparse SMC sampler over the repository's ``SMCDiffusion`` API."""

    def __init__(
        self,
        diffusion: Any,
        model: Callable[..., Tensor],
        classifier: Callable[[Tensor], Tensor],
        target_class: int,
        num_particles: int,
        guidance_schedule: Iterable[int],
        *,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        device: Optional[torch.device | str] = None,
        value_estimation: str = "one_shot",
        num_rollouts: int = 1,
        collect_value_statistics: bool = True,
        alpha: float = 1.0,
        ess_threshold: float = 0.95,
        partial_resample: Optional[int] = None,
        final_transition: str = "legacy_deterministic",
        clip_denoised: bool = False,
        clip_twisted: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        if not hasattr(diffusion, "T") or not hasattr(
            diffusion, "p_trans_model"
        ):
            raise TypeError(
                "diffusion must expose T and p_trans_model, as SMCDiffusion does"
            )
        self.diffusion = diffusion
        # ``SMCDiffusion.p_mean_variance`` consults this attribute even when
        # truncation is disabled.  The legacy runner injects it dynamically;
        # a standalone sampler must provide the disabled default itself.
        if not hasattr(self.diffusion, "t_truncate"):
            self.diffusion.t_truncate = 0
        self.model = model
        self.classifier = classifier
        self.target_class = int(target_class)
        self.num_particles = int(num_particles)
        self.T = int(diffusion.T)
        if self.T <= 0:
            raise ValueError(f"diffusion.T must be positive, got {self.T}")
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")

        schedule: set[int] = set()
        for raw_t in guidance_schedule:
            if isinstance(raw_t, bool) or int(raw_t) != raw_t:
                raise TypeError(f"schedule entries must be integers, got {raw_t!r}")
            paper_t = int(raw_t)
            if not 0 <= paper_t < self.T:
                raise ValueError(
                    f"paper timestep {paper_t} is outside [0, {self.T - 1}]"
                )
            schedule.add(paper_t)
        self.guidance_schedule = frozenset(schedule)

        if value_estimation not in _VALUE_ESTIMATORS:
            raise ValueError(
                f"value_estimation must be one of {sorted(_VALUE_ESTIMATORS)}, "
                f"got {value_estimation!r}"
            )
        if final_transition not in _FINAL_TRANSITIONS:
            raise ValueError(
                f"final_transition must be one of {sorted(_FINAL_TRANSITIONS)}, "
                f"got {final_transition!r}"
            )
        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")
        if value_estimation == "one_shot" and int(num_rollouts) != 1:
            raise ValueError(
                "continuous one_shot uses the DDPM's single pred_xstart point "
                "estimate, so independent J samples are unavailable; pass "
                "num_rollouts=1"
            )
        if value_estimation == "one_shot_gaussian" and int(num_rollouts) < 3:
            raise ValueError(
                "continuous one_shot_gaussian requires at least three direct "
                "clean-state samples; pass num_rollouts>=3"
            )
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if ess_threshold < 0:
            raise ValueError("ess_threshold must be non-negative")
        if partial_resample is None:
            partial_resample = self.num_particles // 2
        if not 0 <= int(partial_resample) <= self.num_particles:
            raise ValueError(
                "partial_resample must lie in [0, num_particles], got "
                f"{partial_resample}"
            )

        self.value_estimation = value_estimation
        self.num_rollouts = int(num_rollouts)
        self.collect_value_statistics = bool(collect_value_statistics)
        self.alpha = float(alpha)
        self.ess_threshold = float(ess_threshold)
        self.partial_resample = int(partial_resample)
        self.final_transition = final_transition
        self.clip_denoised = bool(clip_denoised)
        self.clip_twisted = bool(clip_twisted)
        self.generator = generator
        self.model_kwargs: Mapping[str, Any] = (
            {} if model_kwargs is None else dict(model_kwargs)
        )
        self.device = (
            torch.device(device) if device is not None else self._infer_device()
        )

    def _infer_device(self) -> torch.device:
        for candidate in (self.model, self.classifier):
            parameters = getattr(candidate, "parameters", None)
            if callable(parameters):
                try:
                    return next(parameters()).device
                except (StopIteration, TypeError):
                    pass
        return torch.device("cpu")

    def _classifier_log_reward(self, x0: Tensor) -> Tensor:
        """Return log R(x0) = log p(target|x0) / alpha."""

        output = self.classifier(x0)
        if not isinstance(output, Tensor):
            raise TypeError("classifier must return a torch.Tensor")
        if output.ndim == 1:
            if output.shape[0] != x0.shape[0]:
                raise ValueError(
                    "one-dimensional classifier output must have one selected "
                    "log probability per particle"
                )
            selected_log_probability = output
        elif output.ndim == 2:
            if output.shape[0] != x0.shape[0]:
                raise ValueError("classifier batch dimension does not match x0")
            if not 0 <= self.target_class < output.shape[1]:
                raise ValueError(
                    f"target class {self.target_class} is invalid for "
                    f"{output.shape[1]} classifier classes"
                )
            selected_log_probability = F.log_softmax(output, dim=-1)[
                :, self.target_class
            ]
        else:
            raise ValueError(
                "classifier must return selected log probabilities [N] or "
                f"logits [N,C], got {output.shape}"
            )
        log_reward = selected_log_probability / self.alpha
        if not torch.isfinite(log_reward).all():
            raise FloatingPointError("classifier returned a non-finite log reward")
        # Do not clamp small probabilities: a hard floor would make the
        # proposal gradient exactly zero for sufficiently unlikely particles.
        return log_reward

    def _one_shot_gaussian_variance(self, paper_t: int) -> float:
        """Return the explicit unit-prior approximation Var[x0 | x_t]."""

        alphas_cumprod = getattr(self.diffusion, "alphas_cumprod", None)
        if alphas_cumprod is None:
            raise TypeError(
                "one_shot_gaussian requires diffusion.alphas_cumprod"
            )
        try:
            alpha_bar = float(alphas_cumprod[paper_t])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid alpha_cumprod entry for paper timestep {paper_t}"
            ) from exc
        if not math.isfinite(alpha_bar) or not 0.0 < alpha_bar <= 1.0:
            raise ValueError(
                "alpha_cumprod must be finite and lie in (0, 1], got "
                f"{alpha_bar} at paper timestep {paper_t}"
            )
        return 1.0 - alpha_bar

    def _one_shot_log_value_from_prediction(
        self,
        pred_xstart: Tensor,
        paper_t: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Evaluate a direct clean-state value without a reverse rollout."""

        if self.value_estimation == "one_shot":
            return self._classifier_log_reward(pred_xstart)
        if self.value_estimation != "one_shot_gaussian":
            raise AssertionError("direct clean-state value used by ancestral mode")

        repeats = self.num_rollouts
        expanded_mean = pred_xstart.unsqueeze(1).expand(
            -1, repeats, *pred_xstart.shape[1:]
        )
        variance = self._one_shot_gaussian_variance(paper_t)
        generator = self.generator if generator is None else generator
        clean_samples = expanded_mean + math.sqrt(variance) * _randn_like(
            expanded_mean, generator
        )
        if self.clip_denoised:
            clean_samples = clean_samples.clamp(-1.0, 1.0)
        flat_samples = clean_samples.flatten(0, 1)
        log_rewards = self._classifier_log_reward(flat_samples).view(
            pred_xstart.shape[0], repeats
        )
        return torch.logsumexp(log_rewards, dim=1) - math.log(repeats)

    def _transition(
        self,
        source: Tensor,
        paper_t: int,
        *,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> _Transition:
        """Evaluate unconditional p_{t|t+1}; ``paper_t`` names destination."""

        kwargs = self.model_kwargs if model_kwargs is None else model_kwargs
        out = self.diffusion.p_trans_model(
            xtp1=source,
            t=paper_t,
            model=self.model,
            clip_denoised=self.clip_denoised,
            model_kwargs=kwargs,
        )
        required = ("mean_untwisted", "var_untwisted", "pred_xstart")
        missing = [key for key in required if key not in out]
        if missing:
            raise KeyError(f"p_trans_model output is missing {missing}")
        mean = out["mean_untwisted"]
        variance = out["var_untwisted"]
        pred_xstart = out["pred_xstart"]
        if mean.shape != source.shape:
            raise ValueError(
                f"p mean shape {mean.shape} does not match source {source.shape}"
            )
        try:
            variance = torch.broadcast_to(variance, mean.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"p variance shape {variance.shape} cannot broadcast to {mean.shape}"
            ) from exc
        if pred_xstart.shape != source.shape:
            raise ValueError(
                "pred_xstart must match the particle shape, got "
                f"{pred_xstart.shape} and {source.shape}"
            )
        if not (
            torch.isfinite(mean).all()
            and torch.isfinite(variance).all()
            and torch.isfinite(pred_xstart).all()
        ):
            raise FloatingPointError("p_trans_model returned non-finite values")
        if (variance < 0).any():
            raise ValueError("p_trans_model returned a negative variance")
        return _Transition(mean, variance, pred_xstart)

    def _sample_from_p(
        self,
        transition: _Transition,
        paper_t: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if paper_t == 0 and self.final_transition == "legacy_deterministic":
            return transition.mean
        generator = self.generator if generator is None else generator
        return transition.mean + transition.variance.sqrt() * _randn_like(
            transition.mean, generator
        )

    def _tds_vp_mean_shift(
        self,
        score: Tensor,
        source: Tensor,
        pred_xstart: Tensor,
        p_mean: Tensor,
        paper_t: int,
    ) -> Tensor:
        """Return the upstream VP-TDS mean shift for ``x_{t+1} -> x_t``.

        ``TwistedDDPM._compute_twisted_step_helper`` shifts ``pred_xstart``,
        optionally clamps it, and passes it through the posterior-mean formula.
        The familiar beta/sqrt(alpha) shift is only the unclipped algebraic
        simplification.  ``paper_t`` is the destination index and therefore the
        correct index into the respaced DDPM arrays.
        """

        if not 0 <= paper_t < self.T:
            raise ValueError(f"paper timestep must lie in [0, {self.T})")
        alphas_cumprod = getattr(self.diffusion, "alphas_cumprod", None)
        posterior = getattr(self.diffusion, "q_posterior_mean_variance", None)
        if alphas_cumprod is None or not callable(posterior):
            raise TypeError(
                "upstream VP-TDS proposals require alphas_cumprod and "
                "q_posterior_mean_variance"
            )
        alpha_bar = float(alphas_cumprod[paper_t])
        if not math.isfinite(alpha_bar) or not 0.0 < alpha_bar <= 1.0:
            raise ValueError(
                f"invalid alpha_cumprod at paper timestep {paper_t}: "
                f"{alpha_bar}"
            )
        twisted_xstart = pred_xstart + (
            (1.0 - alpha_bar) / math.sqrt(alpha_bar)
        ) * score
        if self.clip_twisted:
            twisted_xstart = twisted_xstart.clamp(-1.0, 1.0)
        q_mean, _, _ = posterior(
            x_start=twisted_xstart,
            x_t=source,
            t=paper_t,
        )
        if q_mean.shape != p_mean.shape or not torch.isfinite(q_mean).all():
            raise FloatingPointError(
                "q_posterior_mean_variance returned an invalid TDS mean"
            )
        return q_mean - p_mean

    def _estimate_log_value(
        self,
        state: Tensor,
        state_t: int,
        *,
        track_grad: bool,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Estimate per-particle log V_state_t(state).

        ``one_shot`` mirrors the paper's direct clean-state prediction but uses
        a delta distribution at this DDPM's ``pred_xstart``.  ``ancestral`` is
        retained only as a full-chain diagnostic ablation.
        """

        if not 0 <= state_t <= self.T:
            raise ValueError(f"state timestep must lie in [0, {self.T}]")
        context = torch.enable_grad() if track_grad else torch.no_grad()
        with context:
            if state_t == 0:
                return self._classifier_log_reward(state)
            if self.value_estimation in _ONE_SHOT_ESTIMATORS:
                transition = self._transition(state, state_t - 1)
                return self._one_shot_log_value_from_prediction(
                    transition.pred_xstart,
                    state_t - 1,
                    generator=generator,
                )

            num_particles = state.shape[0]
            repeats = self.num_rollouts
            rollout = state.repeat_interleave(repeats, dim=0)
            rollout_kwargs = _repeat_particle_kwargs(
                self.model_kwargs, num_particles, repeats
            )
            for paper_t in range(state_t - 1, -1, -1):
                transition = self._transition(
                    rollout, paper_t, model_kwargs=rollout_kwargs
                )
                rollout = self._sample_from_p(
                    transition, paper_t, generator=generator
                )
            log_rewards = self._classifier_log_reward(rollout).view(
                num_particles, repeats
            )
            return torch.logsumexp(log_rewards, dim=1) - math.log(repeats)

    def _guided_transition(
        self, source: Tensor, paper_t: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample q and return ``(x_t, log(p/q), mean_shift_norm)``."""

        if paper_t == 0 and self.final_transition == "legacy_deterministic":
            with torch.no_grad():
                transition = self._transition(source, paper_t)
            zeros = source.new_zeros(self.num_particles)
            return transition.mean.detach(), zeros, zeros

        source_for_grad = source.detach().requires_grad_(True)
        with torch.enable_grad():
            transition = self._transition(source_for_grad, paper_t)
            # q is the repository's VP-TDS proposal: shift differentiable
            # pred_xstart by its source-state classifier score, apply the
            # configured clean-state clipping, then recompute the posterior
            # mean. The proposal remains deterministic in this potential even
            # when ancestral rollouts estimate the importance-weight value.
            source_log_value = self._classifier_log_reward(
                transition.pred_xstart
            )
            if source_log_value.requires_grad:
                score = torch.autograd.grad(
                    source_log_value.sum(),
                    source_for_grad,
                    allow_unused=True,
                )[0]
            else:
                score = None
        if score is None:
            score = torch.zeros_like(source_for_grad)
        if not torch.isfinite(score).all():
            raise FloatingPointError("classifier value gradient is non-finite")

        p_mean = transition.mean.detach()
        variance = transition.variance.detach()
        mean_shift = self._tds_vp_mean_shift(
            score.detach(),
            source.detach(),
            transition.pred_xstart.detach(),
            p_mean,
            paper_t,
        )
        q_mean = p_mean + mean_shift
        sample = q_mean + variance.sqrt() * _randn_like(q_mean, self.generator)
        log_p_over_q = gaussian_log_p_over_q(
            sample, p_mean, q_mean, variance
        )
        shift_norm = mean_shift.flatten(start_dim=1).norm(dim=1)
        return sample.detach(), log_p_over_q.detach(), shift_norm.detach()

    def _unguided_transition(self, source: Tensor, paper_t: int) -> Tensor:
        with torch.no_grad():
            transition = self._transition(source, paper_t)
            return self._sample_from_p(transition, paper_t).detach()

    def _prepare_one_shot_step(
        self,
        source: Tensor,
        state_t: int,
        *,
        need_value: bool,
        guided_outgoing: bool,
    ) -> _PreparedOneShotStep:
        """Evaluate the one edge leaving noisy state ``x_state_t`` once.

        The same ``pred_xstart`` supplies both the Tweedie value and, when
        requested, the TDS proposal gradient.  Only detached tensors enter
        the returned cache.  This is what makes a full one-shot run require
        exactly the ``T`` calls for edges ``T-1, ..., 0``.
        """

        if self.value_estimation not in _ONE_SHOT_ESTIMATORS:
            raise AssertionError("one-shot step cache used by ancestral mode")
        if not 1 <= state_t <= self.T:
            raise ValueError(
                f"noisy state timestep must lie in [1, {self.T}], got {state_t}"
            )

        paper_t = state_t - 1
        legacy_final = (
            paper_t == 0
            and self.final_transition == "legacy_deterministic"
        )
        need_q_shift = bool(guided_outgoing and not legacy_final)
        source_for_model = source.detach()
        if need_q_shift:
            source_for_model = source_for_model.requires_grad_(True)
        context = torch.enable_grad() if need_q_shift else torch.no_grad()
        if self.value_estimation == "one_shot":
            need_reward = bool(need_value or need_q_shift)
            with context:
                transition = self._transition(source_for_model, paper_t)
                shared_log_value = (
                    self._classifier_log_reward(transition.pred_xstart)
                    if need_reward
                    else None
                )
                if need_q_shift:
                    if shared_log_value is None:
                        raise AssertionError(
                            "guided cache is missing its potential"
                        )
                    if shared_log_value.requires_grad:
                        score = torch.autograd.grad(
                            shared_log_value.sum(),
                            source_for_model,
                            allow_unused=True,
                        )[0]
                    else:
                        score = None
                else:
                    score = None
            log_value = shared_log_value if need_value else None
        else:
            # The stochastic J-sample value is used only in SMC weights.  The
            # proposal remains the repository's deterministic TDS potential
            # on pred_xstart, so random clean draws never randomize q.
            with context:
                transition = self._transition(source_for_model, paper_t)
                proposal_log_value = (
                    self._classifier_log_reward(transition.pred_xstart)
                    if need_q_shift
                    else None
                )
                if (
                    proposal_log_value is not None
                    and proposal_log_value.requires_grad
                ):
                    score = torch.autograd.grad(
                        proposal_log_value.sum(),
                        source_for_model,
                        allow_unused=True,
                    )[0]
                else:
                    score = None
            with torch.no_grad():
                log_value = (
                    self._one_shot_log_value_from_prediction(
                        transition.pred_xstart.detach(),
                        paper_t,
                        generator=self.generator,
                    )
                    if need_value
                    else None
                )

        if need_q_shift and score is None:
            score = torch.zeros_like(source_for_model)
        if score is not None and not torch.isfinite(score).all():
            raise FloatingPointError("classifier value gradient is non-finite")

        detached_transition = _Transition(
            mean=transition.mean.detach(),
            variance=transition.variance.detach(),
            pred_xstart=transition.pred_xstart.detach(),
        )
        q_shift = (
            self._tds_vp_mean_shift(
                score.detach(),
                source.detach(),
                transition.pred_xstart.detach(),
                detached_transition.mean,
                paper_t,
            )
            if score is not None
            else None
        )
        return _PreparedOneShotStep(
            paper_t=paper_t,
            transition=detached_transition,
            log_value=None if log_value is None else log_value.detach(),
            q_shift=q_shift,
        )

    @staticmethod
    def _index_prepared_one_shot_step(
        prepared: _PreparedOneShotStep,
        ancestors: Tensor,
    ) -> _PreparedOneShotStep:
        """Apply the particle resampling ancestor map to every cached field."""

        transition = prepared.transition
        return _PreparedOneShotStep(
            paper_t=prepared.paper_t,
            transition=_Transition(
                mean=transition.mean[ancestors].detach(),
                variance=transition.variance[ancestors].detach(),
                pred_xstart=transition.pred_xstart[ancestors].detach(),
            ),
            log_value=(
                None
                if prepared.log_value is None
                else prepared.log_value[ancestors].detach()
            ),
            q_shift=(
                None
                if prepared.q_shift is None
                else prepared.q_shift[ancestors].detach()
            ),
        )

    def _sample_prepared_one_shot_step(
        self,
        prepared: _PreparedOneShotStep,
        *,
        active: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Consume one cached edge and return sample, log ratio, q shift norm."""

        paper_t = prepared.paper_t
        transition = prepared.transition
        zeros = transition.mean.new_zeros(self.num_particles)
        if paper_t == 0 and self.final_transition == "legacy_deterministic":
            return transition.mean.detach(), zeros, zeros
        if not active:
            return self._sample_from_p(transition, paper_t).detach(), zeros, zeros
        if prepared.q_shift is None:
            raise AssertionError("guided outgoing edge has no cached q shift")

        p_mean = transition.mean
        variance = transition.variance
        q_mean = p_mean + prepared.q_shift
        sample = q_mean + variance.sqrt() * _randn_like(
            q_mean, self.generator
        )
        log_p_over_q = gaussian_log_p_over_q(
            sample, p_mean, q_mean, variance
        )
        shift_norm = prepared.q_shift.flatten(start_dim=1).norm(dim=1)
        return (
            sample.detach(),
            log_p_over_q.detach(),
            shift_norm.detach(),
        )

    def _initial_particles(self, initial_particles: Optional[Tensor]) -> Tensor:
        if initial_particles is not None:
            if initial_particles.shape[0] != self.num_particles:
                raise ValueError(
                    "initial_particles has the wrong leading dimension: "
                    f"{initial_particles.shape[0]} != {self.num_particles}"
                )
            return initial_particles.to(self.device).detach().clone()

        if self.generator is not None and hasattr(
            self.diffusion, "particle_base_shape"
        ):
            # SMCDiffusion.ref_sample is a standard normal but does not accept
            # a generator.  Sampling it here makes the entire run controlled
            # by the supplied generator.
            return torch.randn(
                (self.num_particles, *self.diffusion.particle_base_shape),
                device=self.device,
                generator=self.generator,
            )
        return self.diffusion.ref_sample(
            self.num_particles, device=self.device
        ).detach()

    def _sample_one_shot(
        self, initial_particles: Optional[Tensor]
    ) -> tuple[Tensor, dict[str, Any], Tensor]:
        """Run a direct-clean one-shot estimator with one denoiser call/state."""

        particles = self._initial_particles(initial_particles)
        if particles.shape[0] != self.num_particles:
            raise AssertionError("internal particle-count mismatch")
        value_means = torch.full(
            (self.T + 1,),
            torch.nan,
            dtype=particles.dtype,
            device=particles.device,
        )
        ess_trace = torch.full_like(value_means, torch.nan)
        resampled_trace = torch.zeros(
            self.T + 1, dtype=torch.bool, device=particles.device
        )
        mean_log_p_over_q = torch.full_like(value_means, torch.nan)
        q_shift_norm = torch.full_like(value_means, torch.nan)
        ancestor_maps: MutableMapping[int, Tensor] = {}

        # Prepare x_T before its value correction.  If partial resampling
        # changes its ancestry, the exact same map is applied to the cached
        # p transition and proposal shift before the edge is consumed.
        prepared = self._prepare_one_shot_step(
            particles,
            self.T,
            need_value=True,
            guided_outgoing=(self.T - 1 in self.guidance_schedule),
        )
        if prepared.log_value is None:
            raise AssertionError("x_T cache is missing its prior value")
        prior_log_values = prepared.log_value
        log_weights = _normalize_log_weights(prior_log_values)
        last_guided_log_values = prior_log_values
        (
            particles,
            log_weights,
            last_guided_log_values,
            ancestors,
            was_resampled,
            prior_ess,
        ) = adaptive_systematic_partial_resample(
            particles,
            log_weights,
            last_guided_log_values,
            ess_threshold=self.ess_threshold,
            partial_resample=self.partial_resample,
            generator=self.generator,
        )
        prepared = self._index_prepared_one_shot_step(
            prepared, ancestors
        )
        ess_trace[self.T] = prior_ess
        resampled_trace[self.T] = was_resampled
        ancestor_maps[self.T] = ancestors.detach().cpu()
        if self.collect_value_statistics:
            # Equation (11) averages particles representing pi*_t. Adaptive
            # partial resampling can leave non-uniform weights, so retain the
            # normalized-weight estimator (it becomes a plain mean after full
            # resampling). The apparent V^2 under the reference law is exactly
            # E_{pi*_t}[V_t] = E_{p_t}[V_t^2] / Z.
            value_means[self.T] = weighted_value_mean(
                last_guided_log_values, log_weights
            )

        transition_kinds: list[str] = []
        q_transition_count = 0
        for paper_t in range(self.T - 1, -1, -1):
            if prepared.paper_t != paper_t:
                raise AssertionError(
                    "prepared one-shot edge is out of timestep order: "
                    f"{prepared.paper_t} != {paper_t}"
                )
            active = paper_t in self.guidance_schedule
            legacy_final = (
                paper_t == 0
                and self.final_transition == "legacy_deterministic"
            )
            particles, log_ratio, shift_norm = (
                self._sample_prepared_one_shot_step(
                    prepared, active=active
                )
            )
            transition_kinds.append(
                "p_legacy" if legacy_final else ("q" if active else "p")
            )
            if active and not legacy_final:
                q_transition_count += 1

            # At a noisy destination x_t, prepare edge t-1 now.  Its Tweedie
            # prediction is simultaneously V_t when this destination is an
            # active correction and the q potential when edge t-1 is active.
            next_prepared: Optional[_PreparedOneShotStep]
            if paper_t > 0:
                next_prepared = self._prepare_one_shot_step(
                    particles,
                    paper_t,
                    need_value=active,
                    guided_outgoing=(
                        paper_t - 1 in self.guidance_schedule
                    ),
                )
                current_log_values = next_prepared.log_value
            else:
                next_prepared = None
                with torch.no_grad():
                    current_log_values = (
                        self._classifier_log_reward(particles).detach()
                        if active
                        else None
                    )

            if active:
                if current_log_values is None:
                    raise AssertionError(
                        f"active destination x_{paper_t} has no value"
                    )
                log_weights = _normalize_log_weights(
                    log_weights
                    + log_ratio
                    + current_log_values
                    - last_guided_log_values
                )
                last_guided_log_values = current_log_values
                mean_log_p_over_q[paper_t] = log_ratio.mean()
                q_shift_norm[paper_t] = shift_norm.mean()
                (
                    particles,
                    log_weights,
                    last_guided_log_values,
                    ancestors,
                    was_resampled,
                    current_ess,
                ) = adaptive_systematic_partial_resample(
                    particles,
                    log_weights,
                    last_guided_log_values,
                    ess_threshold=self.ess_threshold,
                    partial_resample=self.partial_resample,
                    generator=self.generator,
                )
                if next_prepared is not None:
                    next_prepared = self._index_prepared_one_shot_step(
                        next_prepared, ancestors
                    )
                ess_trace[paper_t] = current_ess
                resampled_trace[paper_t] = was_resampled
                ancestor_maps[paper_t] = ancestors.detach().cpu()
                if self.collect_value_statistics:
                    value_means[paper_t] = weighted_value_mean(
                        last_guided_log_values, log_weights
                    )
            else:
                ess_trace[paper_t] = effective_sample_size(log_weights)
                ancestor_maps[paper_t] = torch.arange(self.num_particles)

            if next_prepared is not None:
                prepared = next_prepared

        pre_final_log_weights = _normalize_log_weights(log_weights)
        pre_final_ess = effective_sample_size(pre_final_log_weights)
        final_ancestors = torch.arange(
            self.num_particles, device=particles.device
        )

        gaussian_one_shot = self.value_estimation == "one_shot_gaussian"
        stats: dict[str, Any] = {
            "paper_timesteps": tuple(range(self.T - 1, -1, -1)),
            "guided_steps": tuple(sorted(self.guidance_schedule)),
            "guided_execution_order": tuple(
                t
                for t in range(self.T - 1, -1, -1)
                if t in self.guidance_schedule
            ),
            "transition_kinds": tuple(transition_kinds),
            "active_count": len(self.guidance_schedule),
            "skipped_count": self.T - len(self.guidance_schedule),
            "q_transition_count": q_transition_count,
            "ess_before_resampling": ess_trace.detach().cpu(),
            "resampled": resampled_trace.detach().cpu(),
            "mean_log_p_over_q": mean_log_p_over_q.detach().cpu(),
            "mean_q_shift_norm": q_shift_norm.detach().cpu(),
            "ancestor_maps": dict(ancestor_maps),
            "pre_final_log_weights": pre_final_log_weights.detach().cpu(),
            "pre_final_ess": float(pre_final_ess.detach().cpu()),
            "final_log_weights": pre_final_log_weights.detach().cpu(),
            "final_ancestor_indices": final_ancestors.detach().cpu(),
            "terminal_full_resampled": False,
            "terminal_resampling": "scheduled_adaptive_partial_only",
            "value_estimation": self.value_estimation,
            "value_estimator_definition": (
                "monte_carlo_mean_reward_over_one_shot_gaussian_x0"
                if gaussian_one_shot
                else "classifier_on_one_shot_pred_xstart_delta"
            ),
            "vista_value_role": (
                "continuous_gaussian_p0_given_xt_approximation"
                if gaussian_one_shot
                else "continuous_delta_p0_given_xt_approximation"
            ),
            "paper_value_sampling_structure": "one_shot",
            "paper_cost_model_aligned": True,
            "continuous_tds_implementation_aligned": (
                not gaussian_one_shot and self.clip_twisted
            ),
            "proposal_implementation_aligned": self.clip_twisted,
            "continuous_extension": gaussian_one_shot,
            "direct_p0_given_xt_distribution": (
                "gaussian_pred_xstart_mean_unit_prior_posterior_variance"
                if gaussian_one_shot
                else "delta_at_pred_xstart"
            ),
            "one_shot_clean_variance": (
                "1_minus_alpha_cumprod_at_noisy_state"
                if gaussian_one_shot
                else "zero"
            ),
            "exact_p0_given_xt_available": False,
            "supports_theoretical_utd": False,
            "uses_ancestral_rollouts": False,
            "effective_ancestral_rollouts": 0,
            "effective_one_shot_samples": (
                self.num_rollouts if gaussian_one_shot else 1
            ),
            "denoiser_calls_per_nonterminal_value_query": 1,
            "denoiser_calls_per_noisy_state": 1,
            "state_kernel_cache": "resampling_ancestor_indexed",
            "proposal_potential": "classifier_on_pred_xstart",
            "proposal_family": (
                "upstream_vp_tds_twisted_xstart_posterior"
                if self.clip_twisted
                else "vp_tds_twisted_xstart_posterior_unclipped_ablation"
            ),
            "clip_twisted": self.clip_twisted,
            "num_rollouts": self.num_rollouts,
            "value_statistics_collected": self.collect_value_statistics,
            "value_statistics_estimator": (
                (
                    "post_resampling_normalized_weight_mean_of_same_j_sample_estimate"
                    if gaussian_one_shot
                    else "post_resampling_normalized_weight_mean_of_tweedie_value"
                )
                if self.collect_value_statistics
                else "not_collected"
            ),
            "value_statistics_independent_of_weight_estimate": False,
            "finite_j_self_weighting_bias": (
                "possible_same_estimate_used_for_weight_and_statistic"
                if gaussian_one_shot
                else "not_applicable_deterministic_delta"
            ),
            "ancestral_rollouts_per_weight_query": 0,
            "ancestral_rollouts_per_value_statistic_query": 0,
            "final_transition": self.final_transition,
            "partial_resample": self.partial_resample,
            "ess_threshold": self.ess_threshold,
        }
        return particles.detach(), stats, value_means.detach()

    def sample(
        self, initial_particles: Optional[Tensor] = None
    ) -> tuple[Tensor, dict[str, Any], Tensor]:
        """Run sparse SMC and return ``(samples, stats, value_means)``.

        ``value_means`` has length ``T+1`` in paper order: array index ``t``
        stores the estimate for V_t.  Inactive sparse timesteps are NaN.  When
        ``collect_value_statistics`` is false, the entire array is NaN while
        the value estimates required by the SMC weights are still computed.
        """

        if self.value_estimation in _ONE_SHOT_ESTIMATORS:
            return self._sample_one_shot(initial_particles)

        particles = self._initial_particles(initial_particles)
        if particles.shape[0] != self.num_particles:
            raise AssertionError("internal particle-count mismatch")
        value_means = torch.full(
            (self.T + 1,),
            torch.nan,
            dtype=particles.dtype,
            device=particles.device,
        )
        ess_trace = torch.full_like(value_means, torch.nan)
        resampled_trace = torch.zeros(
            self.T + 1, dtype=torch.bool, device=particles.device
        )
        mean_log_p_over_q = torch.full_like(value_means, torch.nan)
        q_shift_norm = torch.full_like(value_means, torch.nan)
        ancestor_maps: MutableMapping[int, Tensor] = {}

        # Algorithm 5 boundary t=T: q_T=p_T for SMCDiffusion's normal prior.
        prior_log_values = self._estimate_log_value(
            particles, self.T, track_grad=False
        ).detach()
        log_weights = _normalize_log_weights(prior_log_values)
        last_guided_log_values = prior_log_values
        (
            particles,
            log_weights,
            last_guided_log_values,
            ancestors,
            was_resampled,
            prior_ess,
        ) = adaptive_systematic_partial_resample(
            particles,
            log_weights,
            last_guided_log_values,
            ess_threshold=self.ess_threshold,
            partial_resample=self.partial_resample,
            generator=self.generator,
        )
        ess_trace[self.T] = prior_ess
        resampled_trace[self.T] = was_resampled
        ancestor_maps[self.T] = ancestors.detach().cpu()
        if self.collect_value_statistics:
            value_means[self.T] = weighted_value_mean(
                last_guided_log_values, log_weights
            )

        transition_kinds: list[str] = []
        q_transition_count = 0
        for paper_t in range(self.T - 1, -1, -1):
            active = paper_t in self.guidance_schedule
            legacy_final = (
                paper_t == 0
                and self.final_transition == "legacy_deterministic"
            )
            if active:
                particles, log_ratio, shift_norm = self._guided_transition(
                    particles, paper_t
                )
                transition_kinds.append("p_legacy" if legacy_final else "q")
                if not legacy_final:
                    q_transition_count += 1

                current_log_values = self._estimate_log_value(
                    particles, paper_t, track_grad=False
                ).detach()
                log_weights = _normalize_log_weights(
                    log_weights
                    + log_ratio
                    + current_log_values
                    - last_guided_log_values
                )
                last_guided_log_values = current_log_values
                mean_log_p_over_q[paper_t] = log_ratio.mean()
                q_shift_norm[paper_t] = shift_norm.mean()
                (
                    particles,
                    log_weights,
                    last_guided_log_values,
                    ancestors,
                    was_resampled,
                    current_ess,
                ) = adaptive_systematic_partial_resample(
                    particles,
                    log_weights,
                    last_guided_log_values,
                    ess_threshold=self.ess_threshold,
                    partial_resample=self.partial_resample,
                    generator=self.generator,
                )
                ess_trace[paper_t] = current_ess
                resampled_trace[paper_t] = was_resampled
                ancestor_maps[paper_t] = ancestors.detach().cpu()
                if self.collect_value_statistics:
                    value_means[paper_t] = weighted_value_mean(
                        last_guided_log_values, log_weights
                    )
            else:
                particles = self._unguided_transition(particles, paper_t)
                transition_kinds.append(
                    "p_legacy" if legacy_final else "p"
                )
                ess_trace[paper_t] = effective_sample_size(log_weights)
                ancestor_maps[paper_t] = torch.arange(
                    self.num_particles
                )

        pre_final_log_weights = _normalize_log_weights(log_weights)
        pre_final_ess = effective_sample_size(pre_final_log_weights)
        final_ancestors = torch.arange(
            self.num_particles, device=particles.device
        )

        stats: dict[str, Any] = {
            "paper_timesteps": tuple(range(self.T - 1, -1, -1)),
            "guided_steps": tuple(sorted(self.guidance_schedule)),
            "guided_execution_order": tuple(
                t
                for t in range(self.T - 1, -1, -1)
                if t in self.guidance_schedule
            ),
            "transition_kinds": tuple(transition_kinds),
            "active_count": len(self.guidance_schedule),
            "skipped_count": self.T - len(self.guidance_schedule),
            "q_transition_count": q_transition_count,
            "ess_before_resampling": ess_trace.detach().cpu(),
            "resampled": resampled_trace.detach().cpu(),
            "mean_log_p_over_q": mean_log_p_over_q.detach().cpu(),
            "mean_q_shift_norm": q_shift_norm.detach().cpu(),
            "ancestor_maps": dict(ancestor_maps),
            "pre_final_log_weights": pre_final_log_weights.detach().cpu(),
            "pre_final_ess": float(pre_final_ess.detach().cpu()),
            "final_log_weights": pre_final_log_weights.detach().cpu(),
            "final_ancestor_indices": final_ancestors.detach().cpu(),
            "terminal_full_resampled": False,
            "terminal_resampling": "scheduled_adaptive_partial_only",
            "value_estimation": self.value_estimation,
            "value_estimator_definition": (
                "monte_carlo_mean_reward_over_p_reverse_paths"
            ),
            "paper_value_sampling_structure": "full_reverse_chain_ablation",
            "paper_cost_model_aligned": False,
            "vista_value_role": "monte_carlo_conditional_expectation",
            "supports_theoretical_utd": True,
            "uses_ancestral_rollouts": True,
            "effective_ancestral_rollouts": self.num_rollouts,
            "denoiser_calls_per_nonterminal_value_query": None,
            "proposal_potential": "classifier_on_pred_xstart",
            "proposal_implementation_aligned": self.clip_twisted,
            "proposal_family": (
                "upstream_vp_tds_twisted_xstart_posterior"
                if self.clip_twisted
                else "vp_tds_twisted_xstart_posterior_unclipped_ablation"
            ),
            "clip_twisted": self.clip_twisted,
            "num_rollouts": self.num_rollouts,
            "value_statistics_collected": self.collect_value_statistics,
            "value_statistics_estimator": (
                "post_resampling_normalized_weight_mean_of_weight_rollout_estimates"
                if self.collect_value_statistics
                else "not_collected"
            ),
            "value_statistics_independent_of_weight_estimate": False,
            "ancestral_rollouts_per_weight_query": self.num_rollouts,
            "ancestral_rollouts_per_value_statistic_query": (
                self.num_rollouts if self.collect_value_statistics else 0
            ),
            "final_transition": self.final_transition,
            "partial_resample": self.partial_resample,
            "ess_threshold": self.ess_threshold,
        }
        return particles.detach(), stats, value_means.detach()


def sample_vista_smc(
    diffusion: Any,
    model: Callable[..., Tensor],
    classifier: Callable[[Tensor], Tensor],
    target_class: int,
    num_particles: int,
    guidance_schedule: Iterable[int],
    *,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device | str] = None,
    value_estimation: str = "one_shot",
    num_rollouts: int = 1,
    collect_value_statistics: bool = True,
    alpha: float = 1.0,
    ess_threshold: float = 0.95,
    partial_resample: Optional[int] = None,
    final_transition: str = "legacy_deterministic",
    clip_denoised: bool = False,
    clip_twisted: bool = True,
    generator: Optional[torch.Generator] = None,
    initial_particles: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Any], Tensor]:
    """Functional wrapper for :class:`ContinuousVistaSMCSampler`."""

    sampler = ContinuousVistaSMCSampler(
        diffusion=diffusion,
        model=model,
        classifier=classifier,
        target_class=target_class,
        num_particles=num_particles,
        guidance_schedule=guidance_schedule,
        model_kwargs=model_kwargs,
        device=device,
        value_estimation=value_estimation,
        num_rollouts=num_rollouts,
        collect_value_statistics=collect_value_statistics,
        alpha=alpha,
        ess_threshold=ess_threshold,
        partial_resample=partial_resample,
        final_transition=final_transition,
        clip_denoised=clip_denoised,
        clip_twisted=clip_twisted,
        generator=generator,
    )
    return sampler.sample(initial_particles=initial_particles)


__all__ = [
    "ContinuousVistaSMCSampler",
    "adaptive_systematic_partial_resample",
    "effective_sample_size",
    "full_systematic_resample",
    "gaussian_log_p_over_q",
    "sample_vista_smc",
    "systematic_resample_indices",
    "weighted_value_mean",
]
