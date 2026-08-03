import contextlib
import math

import torch
import torch.nn.functional as F
import tqdm

from .base import Algo


@contextlib.contextmanager
def backbone_eval_mode(net):
    '''
    Temporarily force the diffusion backbone into eval() mode (dropout off) for the duration of
    an SMC run, restoring whatever mode it was in before on exit.

    Nothing in this repo's samplers ever puts the backbone in eval mode (checkpoints are loaded
    via `MDLMDiffusion.load_from_checkpoint`, which doesn't touch train/eval state, and the only
    `.eval()`/`.train()` calls on the backbone live in PyTorch-Lightning hooks that only fire
    inside an actual `Trainer.fit()` loop -- never during standalone sampling). With the
    checkpoints' configured `dropout: 0.1`, that means every `net.model.forward` call during
    sampling is dropout-perturbed: calling it twice on the identical (x, t) gives measurably
    different logits (checked empirically: max abs difference ~6 in log-prob space).

    This matters more for SMC than for the other samplers here because SMC's importance weight
    (V_t(x_t) / V_{t+1}(x_{t+1})) is only a valid correction to the base kernel if p_{0|t} and
    p_{t|t+1} are genuinely fixed functions of (x, t); with active dropout they're rolls of the
    dice on every call, which silently corrupts the weight math with uncontrolled extra variance
    that Algorithm 4/5 don't account for -- rather than J independent draws from a single
    p_{0|t}(.|x_t), you get J draws from J different noisy realizations of it.

    Scoped as a context manager around the SMC inference calls specifically (not a permanent
    `.eval()` in `__init__` or a repo-wide change to `MDLM`) so every other sampler in this repo
    keeps its current dropout-during-sampling behavior unless/until that's addressed separately.
    '''
    model = getattr(net, 'model', None)
    if model is None or not hasattr(model, 'eval'):
        yield
        return
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def systematic_resample(w: torch.Tensor) -> torch.LongTensor:
    '''
    Standard systematic resampling (Algorithm 3 in the VISTA-SMC paper, deterministic-spacing
    variant). Draws N ancestor indices whose empirical distribution matches the normalized
    weights `w` (shape (N,), must sum to 1) at N -> infinity.
    '''
    N = w.shape[0]
    positions = (torch.arange(N, device=w.device, dtype=w.dtype) + torch.rand(1, device=w.device)) / N
    cumsum = torch.cumsum(w, dim=0)
    cumsum[-1] = 1.0  # guard against floating point drift
    idx = torch.searchsorted(cumsum, positions, right=True)
    return idx.clamp(max=N - 1)


class SMC_Base(Algo):
    '''
    SMC for the SMC-base proposal family (q_{t|t+1} = p_{t|t+1}, i.e. the pretrained MDLM
    reverse transition, so the importance-weight update reduces to the value ratio
    V_t(x_t) / V_{t+1}(x_{t+1})). Implements both:
      - Algorithm 4 (full guidance): every reverse step is guided. Default, i.e.
        `inference(..., guided_set=None)`.
      - Algorithm 2 (Reduced/sparse SMC guidance): only timesteps in `guided_set` (a subset of
        {0, ..., T-1}, budget T' = len(guided_set)) are guided; the rest just propagate through
        the plain reverse transition with no value estimation, reweighting, or resampling. The
        weight ratio at the next guided step is computed against V_{ceil(t)} from the *last*
        guided step (Definition 4.1's ceil(t) = min{s in G union {T} | s >= t}), i.e. V_prev is
        simply left untouched across unguided steps -- see `inference`.
    See `sampling/schedules.py` for the heuristic First/Last/Middle/Uniform schedule generators
    used to build `guided_set` at a given budget.

    At each guided timestep the value function
        V_t(x_t) := E_{x0 ~ p_{0|t}(.|x_t)}[R(x0)],  R(x0) = exp(r(x0) / alpha)
    is estimated by drawing `num_rollout_samples` hard samples directly from the denoiser's
    single-call p_theta(x0 | xt), as in the paper's experiments, then scoring them with
    `forward_op`.  A step-by-step reverse-chain rollout is retained only as an explicit
    `rollout_mode="chain"` diagnostic; it is not the experiment default.

    Initialization follows Algorithm 4/5 literally. The model's `get_start` distribution is
    treated as q_T = p_T (the interface does not expose separate p_T/q_T densities), so the
    initial importance weights reduce to w_T proportional to V_T(x_T). The resulting particles
    are always systematically resampled once before reverse propagation. This V_T correction is
    necessary even when p_T is a non-degenerate prior; q_T = p_T only cancels the density ratio,
    not the value factor.

    Resampling after guided reverse steps is adaptive (ESS-triggered).  With
    `partial_resample=False` it is an ordinary full systematic resample.  With
    `partial_resample=True` it is the exact Biology-experiment rule from the reference code:
    select the largest-weight particle and the lowest-weight N/2-1 particles, resample only
    inside that subset, and assign its preserved total mass uniformly to the resampled slots.
    The mandatory t=T correction and the final conversion to an unweighted sample remain full
    systematic resamples.

    Two further fixes on top of the above (found by comparing against the sibling implementation
    in SGPO-main_2's `sampling/sgpo.py`):
      - `inference` threads a single `log_p_x0` per timestep through both the direct value sample
        and the actual reverse transition (`net.compute_log_p_x0` /
        `net.p_sample(..., log_p_x0=...)`).  Thus the default experiment path pays one backbone
        call per evaluated state, independent of the number of Monte Carlo samples J.
      - Value estimation now reduces over rollout samples in log-space (`compute_log_value`, via
        `logsumexp`) instead of averaging `exp(r/alpha)` in linear space -- avoids intermediate
        overflow when `r/alpha` is large across many rollout samples. `compute_value` remains as
        a public linear-space compatibility wrapper.
      - `inference` now defaults to an unconditional resample at the very end
        (`final_resample=True`), so the returned particles are always an unweighted sample of
        the target distribution even if the last guided step's ESS never dropped below
        threshold -- previously, any importance weight left over at the end was silently
        discarded (the old `inference` never returned `logw` at all).
    '''

    def __init__(self, net, data_config, forward_op=None, alpha=1.0,
                 num_rollout_samples=1, ess_threshold=0.98, final_resample=True,
                 rollout_mode='direct', partial_resample=False,
                 n_max_mutations=None, device='cuda'):
        super().__init__(net=net, forward_op=forward_op, data_config=data_config,
                          n_max_mutations=n_max_mutations, device=device)
        self.alpha = alpha
        self.num_rollout_samples = num_rollout_samples
        self.ess_threshold = ess_threshold
        self.final_resample = final_resample
        self.rollout_mode = rollout_mode
        self.partial_resample = partial_resample
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.num_rollout_samples < 1:
            raise ValueError(
                "num_rollout_samples must be at least 1, got "
                f"{self.num_rollout_samples}")
        if not 0.0 <= self.ess_threshold <= 1.0:
            raise ValueError(
                f"ess_threshold must lie in [0, 1], got {self.ess_threshold}")
        if self.rollout_mode not in {'direct', 'chain'}:
            raise ValueError(
                "rollout_mode must be 'direct' (paper experiment) or 'chain', got "
                f"{self.rollout_mode!r}")

    def update_model(self, forward_op):
        self.forward_op = forward_op

    def project(self, x):
        return x * self.mask + self.full_seq * (1 - self.mask)

    def evaluate_reward(self, x0):
        '''r(x0) for a batch of clean (fully unmasked) token sequences, shape (B,).'''
        x0 = self.project(x0)
        strings = [self.net.tokenizer.untokenize(row) for row in x0]
        strings = self.project_sequences(strings)
        combos = [''.join([s[r] for r in self.residues]) for s in strings]
        r = self.forward_op(combos)
        return r.to(x0.device)

    def rollout_to_x0(self, x, t, log_p_x0=None):
        '''
        Sample x0 ~ p_theta(x0 | xt).  The paper experiment uses a single denoiser call followed
        by a hard Gumbel-Softmax categorical draw (`rollout_mode='direct'`).  The historical
        step-by-step reverse-chain simulation remains available as `rollout_mode='chain'` for
        diagnostic comparisons.

        `log_p_x0`, when supplied, is the cached denoiser output for this exact `(x, t)`.
        '''
        if int(t) == 0:
            return x

        if self.rollout_mode == 'direct':
            if log_p_x0 is None and hasattr(self.net, 'compute_log_p_x0'):
                log_p_x0 = self.net.compute_log_p_x0(x, t)
            if log_p_x0 is not None:
                # `hard=True` is straight categorical sampling; tau does not change the hard
                # argmax distribution.  Using the reference implementation's Gumbel draw also
                # lets all J samples share one cached denoiser evaluation.
                return F.gumbel_softmax(
                    log_p_x0, tau=1.0, hard=True, dim=-1).argmax(dim=-1)
            # Generic nets without an exposed p(x0|xt) cache can still request the direct
            # transition through their sampler API.
            t_zero = torch.as_tensor(0, dtype=torch.long)
            return self.net.p_sample(x, t, t_zero, hard=True)

        cur_t = int(t)
        while cur_t > 0:
            t_tensor = torch.as_tensor(cur_t, dtype=torch.long)
            t_next_tensor = torch.as_tensor(cur_t - 1, dtype=torch.long)
            if log_p_x0 is not None:
                x = self.net.p_sample(x, t_tensor, t_next_tensor, hard=True, log_p_x0=log_p_x0)
            else:
                x = self.net.p_sample(x, t_tensor, t_next_tensor, hard=True)
            log_p_x0 = None  # only the first step's cache is valid; later steps recompute
            cur_t -= 1
        return x

    @torch.no_grad()
    def compute_log_value(self, xt, t, log_p_x0=None):
        '''
        MC estimate of log V_t(xt) = log E_{x0 ~ p_{0|t}(.|xt)}[exp(r(x0)/alpha)], shape (N,).
        Reduces over the `num_rollout_samples` rollouts via `logsumexp` rather than averaging
        `exp(r/alpha)` in linear space, so a large `r/alpha` on even one sample can't silently
        overflow the estimate for the whole particle before the reduction happens.
        For t == 0, xt is already clean, so no rollout is needed.

        `log_p_x0`, if given, is `net.compute_log_p_x0(xt, t)` -- reused (after repeating along
        the rollout dimension) instead of recomputed, since the J rollout draws share the same
        xt and t and would otherwise force `rollout_to_x0` to forward an identical input through
        the backbone J times just to get J independent categorical samples from the same logits.
        '''
        N = xt.shape[0]
        if t == 0:
            r = self.evaluate_reward(xt)
            return r / self.alpha

        J = self.num_rollout_samples
        x_roll = xt.repeat_interleave(J, dim=0)
        log_p_x0_roll = log_p_x0.repeat_interleave(J, dim=0) if log_p_x0 is not None else None
        x0_roll = self.rollout_to_x0(x_roll, t, log_p_x0=log_p_x0_roll)
        r = self.evaluate_reward(x0_roll)
        log_R = (r / self.alpha).view(N, J)
        return torch.logsumexp(log_R, dim=1) - math.log(J)

    @torch.no_grad()
    def compute_value(self, xt, t, log_p_x0=None):
        '''Linear-space V_t(xt) = exp(compute_log_value(...)), retained as a public compatibility
        wrapper for callers that need the paper's original (rather than internal log-space) units.'''
        return self.compute_log_value(xt, t, log_p_x0=log_p_x0).exp()

    def maybe_resample(self, x, logw, value, log_p_x0=None):
        '''
        logw is assumed already self-normalized (logsumexp(logw) == 0) on entry.
        Adaptive: only resamples when ESS/N drops below `ess_threshold`.  The selected rule is a
        full systematic resample, or the paper experiment's mass-preserving partial resample when
        `self.partial_resample` is enabled.

        `value` (log- or linear-space V, the caller's choice -- only reindexed here, never used
        arithmetically) and the optional cached `log_p_x0` are carried along and reindexed
        identically to `x`, so a caller reusing `log_p_x0` for the next step's transition still
        gets the right per-particle tensor after ancestors are duplicated/dropped.
        '''
        N = x.shape[0]
        w = logw.exp()
        ess = 1.0 / (w.pow(2).sum() + 1e-12)

        if (ess / N).item() >= self.ess_threshold:
            return x, logw, value, log_p_x0, False

        if self.partial_resample:
            x, logw, value, log_p_x0 = self.partial_resample_particles(
                x, logw, value, log_p_x0)
        else:
            x, logw, value, log_p_x0 = self.resample_particles(
                x, logw, value, log_p_x0)
        return x, logw, value, log_p_x0, True

    @staticmethod
    def _index_particle_data(data, idx):
        '''Recursively reindex any particle-aligned tensor cache.'''
        if data is None:
            return None
        if torch.is_tensor(data):
            return data[idx]
        if isinstance(data, dict):
            return {key: SMC_Base._index_particle_data(value, idx)
                    for key, value in data.items()}
        if isinstance(data, tuple):
            return tuple(SMC_Base._index_particle_data(value, idx) for value in data)
        if isinstance(data, list):
            return [SMC_Base._index_particle_data(value, idx) for value in data]
        raise TypeError(f"unsupported particle cache type: {type(data).__name__}")

    @staticmethod
    def resample_particles(x, logw, value, log_p_x0=None):
        '''Systematically resample a particle-aligned state and reset its weights.

        `value` and `log_p_x0` are not interpreted here; they are reindexed by exactly the same
        ancestor indices as `x`. Keeping this operation in one helper is important at t=T, where
        Algorithm 4/5 requires the value anchor and optional backbone cache to remain aligned with
        the newly resampled particles.
        '''
        N = x.shape[0]
        idx = systematic_resample(logw.exp())
        x = x[idx]
        value = value[idx]
        log_p_x0 = SMC_Base._index_particle_data(log_p_x0, idx)
        logw = torch.full_like(logw, -math.log(N))
        return x, logw, value, log_p_x0

    @staticmethod
    def partial_resample_particles(x, logw, value, log_p_x0=None):
        '''Paper Biology partial resampling with exact subset-mass preservation.

        For N particles, M=floor(N/2).  The selected subset contains the single largest weight
        and the M-1 smallest weights.  Systematic resampling is performed only within that
        subset; its total probability mass is then split uniformly over its M output slots.
        All non-selected particles and weights are retained unchanged.
        '''
        N = x.shape[0]
        M = N // 2
        if M < 1:
            return SMC_Base.resample_particles(x, logw, value, log_p_x0)

        normalized_logw = logw - torch.logsumexp(logw, dim=0)
        weights = normalized_logw.exp()
        high = torch.topk(weights, 1, largest=True).indices
        if M > 1:
            low = torch.topk(weights, M - 1, largest=False).indices
            subset = torch.cat([high, low])
        else:
            subset = high

        subset_weights = weights[subset]
        subset_weights = subset_weights / subset_weights.sum()
        local_idx = systematic_resample(subset_weights)
        idx = torch.arange(N, device=x.device)
        idx[subset] = subset[local_idx]

        new_logw = normalized_logw.clone()
        subset_mass = weights[subset].sum()
        new_logw[subset] = torch.log(subset_mass / M)
        # Preserve normalization in finite precision without changing relative weights.
        new_logw = new_logw - torch.logsumexp(new_logw, dim=0)

        x = x[idx]
        value = value[idx]
        log_p_x0 = SMC_Base._index_particle_data(log_p_x0, idx)
        return x, new_logw, value, log_p_x0

    def inference(self, num_samples=1, verbose=True, detokenize=False, inpaint=False, guided_set=None,
                  value_callback=None, return_logw=False):
        '''
        guided_set: iterable of ints in {0, ..., T-1} -- timesteps to guide (Algorithm 2's G).
            None (default) guides every step, i.e. full guidance (Algorithm 4).
        value_callback: optional callable(t: int, V: Tensor of shape (N,),
            w: Tensor of shape (N,)) invoked every time a value estimate V_t(x_t) is computed
            (t = T once up front, then t_next at each guided step). `w` contains the current
            normalized importance weights after the timestep's resampling decision (the
            mandatory initial resample at T, or adaptive resampling at a reverse step). Under
            full guidance (guided_set=None) this fires for every t = 0, ..., T, which is exactly
            the warmup VISTA-SMC needs (see `sampling/vista.py`) -- no extra rollout cost. `V` is
            in linear space, matching the paper's units. `w` is exposed for diagnostics and
            custom estimators; VISTA itself follows Eq. (11)'s unweighted M-by-N particle mean.
        return_logw: if True, also return the (self-normalized) final importance-weight log_w
            alongside x -- mostly useful with final_resample=False, where the returned particles
            are still weighted and a caller wants to do its own resampling/reweighting downstream.
            With the default final_resample=True, logw is uniform by construction. Setting
            final_resample=False while leaving return_logw=False is rejected rather than silently
            presenting weighted particles as an ordinary unweighted sample.

        Backbone-forward reuse: at each timestep this computes `log_p_x0 = net.compute_log_p_x0(x,
        t)` once (when the net supports it) and reuses it both for the guided value-rollout at t
        and for the reverse-transition call from t to t_next -- see the class docstring. Nets
        without `compute_log_p_x0` fall back to the original, net-agnostic (but ~2x costlier
        under full guidance) behavior automatically.

        The whole run happens with the backbone forced into eval mode (see `backbone_eval_mode`)
        -- scoped to this call only, restored on exit, so nothing else in the repo that reuses
        the same net object is affected.
        '''
        with backbone_eval_mode(self.net):
            return self._inference(num_samples=num_samples, verbose=verbose, detokenize=detokenize,
                                    inpaint=inpaint, guided_set=guided_set,
                                    value_callback=value_callback, return_logw=return_logw)

    def _inference(self, num_samples=1, verbose=True, detokenize=False, inpaint=False, guided_set=None,
                    value_callback=None, return_logw=False):
        if not self.final_resample and not return_logw:
            raise ValueError(
                "final_resample=False may leave non-uniform importance weights; pass "
                "return_logw=True and consume the returned weights, or enable final_resample")

        N = num_samples
        T = self.net.timestep
        use_cache = hasattr(self.net, 'compute_log_p_x0')

        guided_set = set(range(T)) if guided_set is None else set(int(s) for s in guided_set)

        x = self.net.get_start(N)
        cached_log_p_x0 = self.net.compute_log_p_x0(x, T) if use_cache else None
        log_V_prev = self.compute_log_value(x, T, log_p_x0=cached_log_p_x0)

        # Algorithm 4/5, lines 2--7. The sampler interface supplies x_T via `get_start` but does
        # not expose separate p_T and q_T densities, so q_T = p_T is the explicit default model
        # contract. Consequently p_T/q_T cancels while V_T does not: w_T is proportional to
        # V_T(x_T), followed by the paper's mandatory (not ESS-gated) initial resampling.
        logw = log_V_prev - torch.logsumexp(log_V_prev, dim=0)
        x, logw, log_V_prev, cached_log_p_x0 = self.resample_particles(
            x, logw, log_V_prev, cached_log_p_x0)
        if value_callback is not None:
            value_callback(T, log_V_prev.exp(), logw.exp())

        timesteps = torch.linspace(T, 0, T + 1, dtype=torch.long)
        steps = list(zip(timesteps[:-1], timesteps[1:]))
        pbar = tqdm.tqdm(steps, desc="SMC-base") if verbose else steps

        n_guided_resamples = 0
        n_guided = 0
        for t, t_next in pbar:
            # Proposal q_{t|t+1} = p_{t|t+1} either way -- guidance only changes whether we
            # also reweight/resample afterward (Algorithm 2, lines 3-9).
            if use_cache:
                if cached_log_p_x0 is None:
                    cached_log_p_x0 = self.net.compute_log_p_x0(x, t)
                x_next = self.net.p_sample(x, t, t_next, hard=True, log_p_x0=cached_log_p_x0)
            else:
                x_next = self.net.p_sample(x, t, t_next, hard=True)
            if inpaint:
                x_next = self.project(x_next)
            cached_log_p_x0 = None

            t_next_int = int(t_next.item())
            if t_next_int in guided_set:
                n_guided += 1
                log_p_x0_next = (self.net.compute_log_p_x0(x_next, t_next_int)
                                  if use_cache and t_next_int != 0 else None)
                log_V_next = self.compute_log_value(x_next, t_next_int, log_p_x0=log_p_x0_next)
                logw = logw + (log_V_next - log_V_prev)
                logw = logw - torch.logsumexp(logw, dim=0)
                x_next, logw, log_V_next, log_p_x0_next, resampled = self.maybe_resample(
                    x_next, logw, log_V_next, log_p_x0_next)
                n_guided_resamples += int(resampled)
                # Fired *after* maybe_resample.  The callback receives both the particle values
                # and current weights; sampling.vista deliberately uses the former's unweighted
                # mean because that is the statistic specified by Eq. (11).
                if value_callback is not None:
                    value_callback(t_next_int, log_V_next.exp(), logw.exp())
                log_V_prev = log_V_next
                cached_log_p_x0 = log_p_x0_next  # reused as next iteration's transition input
            # else: unguided step -- just propagate x, leave logw/V_prev untouched until the
            # next guided step (they then compare against V_{ceil(t)}, Definition 4.1).

            x = x_next

        did_final_resample = False
        if self.final_resample:
            idx = systematic_resample(logw.exp())
            x = x[idx]
            logw = torch.full_like(logw, -math.log(N))
            did_final_resample = True

        if verbose:
            print(f"SMC-base: guided {n_guided}/{T} steps, "
                  f"adaptive guided resamples={n_guided_resamples}/{n_guided} "
                  f"(initial resample=yes, final resample={did_final_resample})")

        x = self.project(x)

        if detokenize:
            detokenized = [self.net.tokenizer.untokenize(s) for s in x]
            detokenized = self.project_sequences(detokenized)
            return (x, detokenized, logw) if return_logw else (x, detokenized)
        return (x, logw) if return_logw else x
