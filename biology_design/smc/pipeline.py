from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Set, TYPE_CHECKING
import json
import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from smc.resampling import (
    compute_ess_from_log_w,
    normalize_weights,
    systematic_resample_population,
)
from smc.scheduler import BaseScheduler

if TYPE_CHECKING:
    import diffusion_gosai_update


def logmeanexp(x, dim=None, keepdim=False):
    """Numerically stable log-mean-exp."""
    if dim is None:
        x = x.view(-1)
        dim = 0
    lse = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    return lse - math.log(x.size(dim))


@dataclass
class _ValueEstimate:
    """Monte Carlo estimate of a value function at one diffusion state."""

    logits: Optional[torch.Tensor]
    rewards: torch.Tensor
    log_twist: torch.Tensor
    reward_grad: Optional[torch.Tensor] = None
    logits_are_from_base_model: bool = False

    def index_select(self, indices: torch.Tensor) -> "_ValueEstimate":
        return _ValueEstimate(
            logits=None if self.logits is None else self.logits[indices],
            rewards=self.rewards[indices],
            log_twist=self.log_twist[indices],
            reward_grad=None if self.reward_grad is None else self.reward_grad[indices],
            logits_are_from_base_model=self.logits_are_from_base_model,
        )


class Pipeline:
    model: "diffusion_gosai_update.Diffusion"

    def __init__(
        self,
        model: "diffusion_gosai_update.Diffusion",
        scheduler: BaseScheduler,
        device=torch.device("cuda"),
        model_dtype: torch.dtype = torch.float,
    ):
        self.model = model
        self.scheduler = scheduler
        self._execution_device = device
        self.model_dtype = model_dtype

    @torch.no_grad()
    def __call__(
        self,
        reward_fn: Callable,
        resample_fn: Callable,
        resample_frequency: int = 1,
        kl_weight: float = 1.0,
        lambdas: Optional[torch.Tensor] = None,
        num_inference_steps: int = 48,
        batches: int = 1,
        num_particles: int = 1,
        batch_p: int = 1,
        phi: int = 1,
        tau: float = 1.0,
        guidance_steps: Optional[Set[int]] = None,
        use_continuous_formulation: bool = False,
        disable_progress_bar: bool = False,
        final_strategy: str = "argmax_rewards",
        verbose: bool = True,
    ):
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if batches < 1 or num_particles < 1:
            raise ValueError("batches and num_particles must be >= 1")
        if resample_frequency < 1:
            raise ValueError("resample_frequency must be >= 1")
        if kl_weight <= 0:
            raise ValueError("kl_weight must be > 0")
        if lambdas is None:
            lambdas = torch.ones(num_inference_steps + 1)
        if len(lambdas) != num_inference_steps + 1:
            raise ValueError(f"lambdas must have length {num_inference_steps + 1}")
        lambdas = lambdas.clamp_min(0.001).to(self._execution_device)

        total_particles = batches * num_particles
        batch_p = min(batch_p, total_particles)
        length = self.model.config.model.length
        vocab_size = self.model.vocab_size

        latents = self.model._sample_prior(total_particles, length).to(self._execution_device)
        rewards = torch.zeros(total_particles, device=self._execution_device)
        log_twist = torch.zeros(total_particles, device=self._execution_device)
        log_w = torch.zeros(total_particles, device=self._execution_device)
        reward_trace: list[dict[str, Any]] = []

        one = torch.ones(vocab_size, device=self._execution_device).float()
        mask = F.one_hot(
            torch.tensor(self.model.mask_index, device=self._execution_device),
            num_classes=vocab_size,
        ).float()

        eps = 1e-5
        continuous_times = torch.linspace(
            1, eps, num_inference_steps + 1, device=self._execution_device
        )
        dt = (1 - eps) / num_inference_steps

        active_steps = (
            set(range(num_inference_steps))
            if guidance_steps is None
            else {int(step) for step in guidance_steps}
        )

        invalid_steps = sorted(
            step for step in active_steps if step < 0 or step >= num_inference_steps
        )
        if invalid_steps:
            raise ValueError(
                f"guidance_steps must be in [0, {num_inference_steps - 1}], got {invalid_steps}"
            )

        def time_tensor(value: torch.Tensor, size: int) -> torch.Tensor:
            return value * torch.ones(size, 1, device=self._execution_device)

        def get_logits(model, current_latents: torch.Tensor, state_time: torch.Tensor):
            output = torch.zeros(
                (*current_latents.shape, vocab_size), device=self._execution_device
            )
            for start in range(0, total_particles, batch_p):
                stop = min(start + batch_p, total_particles)
                batch_latents = current_latents[start:stop]
                batch_time = time_tensor(state_time, stop - start)
                with torch.no_grad():
                    output[start:stop] = model.get_logits(batch_latents, batch_time).detach()
            return output

        def estimate_value(
            current_latents: torch.Tensor,
            state_time: torch.Tensor,
            state_scale: torch.Tensor,
            *,
            need_grad: bool,
            exact: bool = False,
        ) -> _ValueEstimate:
            if exact:
                with torch.no_grad():
                    exact_rewards = reward_fn(current_latents).detach()
                return _ValueEstimate(
                    logits=None,
                    rewards=exact_rewards,
                    log_twist=exact_rewards * state_scale,
                )

            estimate_logits = torch.zeros(
                (*current_latents.shape, vocab_size), device=self._execution_device
            )
            estimate_rewards = torch.zeros(
                total_particles, device=self._execution_device
            )
            estimate_grad = (
                torch.zeros(
                    (*current_latents.shape, vocab_size),
                    device=self._execution_device,
                )
                if need_grad
                else None
            )

            for start in range(0, total_particles, batch_p):
                stop = min(start + batch_p, total_particles)
                batch_latents = current_latents[start:stop]
                batch_time = time_tensor(state_time, stop - start)

                grad_context = (
                    torch.enable_grad() if need_grad else torch.no_grad()
                )
                with grad_context:
                    one_hot = F.one_hot(
                        batch_latents, num_classes=vocab_size
                    ).to(dtype=self.model_dtype).requires_grad_(need_grad)
                    batch_logits = self.model.get_logits(one_hot, batch_time)
                    rollout_rewards = torch.zeros(
                        stop - start, phi, device=self._execution_device
                    )
                    carry_mask = 1 - ((one - mask) * one_hot).sum(
                        dim=-1, keepdim=True
                    )
                    for rollout in range(phi):
                        sample = F.gumbel_softmax(
                            batch_logits, tau=tau, hard=True
                        )
                        if use_continuous_formulation:
                            sample = (
                                carry_mask * sample
                                + (one - mask) * one_hot
                            )
                        rollout_rewards[:, rollout] = reward_fn(sample)
                    batch_rewards = (
                        logmeanexp(rollout_rewards * state_scale, dim=-1)
                        / state_scale
                    )
                    if need_grad:
                        batch_grad = torch.autograd.grad(
                            outputs=batch_rewards,
                            inputs=one_hot,
                            grad_outputs=torch.ones_like(batch_rewards),
                        )[0].detach()
                if need_grad:
                    assert estimate_grad is not None
                    estimate_grad[start:stop] = batch_grad

                estimate_logits[start:stop] = batch_logits.detach()
                estimate_rewards[start:stop] = batch_rewards.detach()

            return _ValueEstimate(
                logits=estimate_logits,
                rewards=estimate_rewards,
                log_twist=estimate_rewards * state_scale,
                reward_grad=estimate_grad,
                logits_are_from_base_model=True,
            )

        def maybe_resample(
            estimate: _ValueEstimate,
            *,
            enabled: bool,
        ) -> tuple[_ValueEstimate, bool]:
            nonlocal latents, rewards, log_twist, log_w
            if not enabled:
                return estimate, False

            matrix_log_w = log_w.reshape(batches, num_particles)
            batch_indices = []
            batch_log_w = []
            any_resampled = False
            for batch_index in range(batches):
                indices, was_resampled, new_log_w = resample_fn(
                    matrix_log_w[batch_index]
                )
                batch_indices.append(indices + batch_index * num_particles)
                batch_log_w.append(new_log_w)
                any_resampled = any_resampled or bool(was_resampled)

            flat_indices = torch.cat(batch_indices, dim=0)
            log_w = torch.cat(batch_log_w, dim=0)
            if any_resampled:
                latents = latents[flat_indices]
                rewards = rewards[flat_indices]
                log_twist = log_twist[flat_indices]
                estimate = estimate.index_select(flat_indices)
            return estimate, any_resampled

        def correct_current_state(
            estimate: _ValueEstimate,
            transition_log_ratio: torch.Tensor,
            *,
            resample_now: bool,
        ) -> tuple[_ValueEstimate, bool]:
            nonlocal rewards, log_twist, log_w
            incremental_log_w = (
                transition_log_ratio + estimate.log_twist - log_twist
            )
            log_w = log_w + incremental_log_w
            rewards = estimate.rewards
            log_twist = estimate.log_twist

            if verbose:
                print("Rewards: ", rewards)
                print("Incremental log weights: ", incremental_log_w)
                print("Log weights: ", log_w.reshape(batches, num_particles))
                print(
                    "Normalized weights: ",
                    normalize_weights(
                        log_w.reshape(batches, num_particles), dim=-1
                    ),
                )
                print(
                    "ESS: ",
                    compute_ess_from_log_w(
                        log_w.reshape(batches, num_particles), dim=-1
                    ),
                )

            return maybe_resample(estimate, enabled=resample_now)

        def append_trace(
            *,
            state_timestep: int,
            state_scale: torch.Tensor,
            estimate: _ValueEstimate,
            resampled: bool,
        ) -> None:
            normalized = normalize_weights(
                log_w.reshape(batches, num_particles), dim=-1
            )
            reward_trace.append(
                {
                    "schema_version": 2,
                    "step": int(num_inference_steps - state_timestep),
                    "timestep": int(state_timestep),
                    "state_timestep": int(state_timestep),
                    "scale_cur": float(state_scale),
                    "phi": int(phi),
                    "num_batches": int(batches),
                    "num_particles": int(num_particles),
                    "resampled": bool(resampled),
                    "reward_aggregated": (
                        estimate.rewards.detach().float().cpu().reshape(-1).tolist()
                    ),
                    "normalized_weights": (
                        normalized.detach().float().cpu().reshape(-1).tolist()
                    ),
                }
            )

        def remove_remaining_noise(current_latents: torch.Tensor) -> torch.Tensor:
            if not self.model.config.sampling.noise_removal:
                return current_latents
            clean_logits = torch.zeros(
                (*current_latents.shape, vocab_size),
                device=self._execution_device,
            )
            with torch.no_grad():
                for start in range(0, total_particles, batch_p):
                    stop = min(start + batch_p, total_particles)
                    batch_latents = current_latents[start:stop]
                    batch_time = time_tensor(
                        continuous_times[-1], stop - start
                    )
                    conditioning = self.model.noise(batch_time)[0]
                    clean_logits[start:stop] = self.model.forward(
                        batch_latents, conditioning
                    )
            return clean_logits[:, :, :-1].argmax(dim=-1)

        cached_state_timestep: Optional[int] = None
        cached_estimate: Optional[_ValueEstimate] = None

        # Algorithm 5 base case: x_T must be corrected independently of G.
        initial_need_grad = num_inference_steps - 1 in active_steps
        initial_estimate = estimate_value(
            latents,
            continuous_times[0],
            lambdas[0] / kl_weight,
            need_grad=initial_need_grad,
        )
        initial_estimate, initial_resampled = correct_current_state(
            initial_estimate,
            torch.zeros_like(log_w),
            resample_now=True,
        )
        append_trace(
            state_timestep=num_inference_steps,
            state_scale=lambdas[0] / kl_weight,
            estimate=initial_estimate,
            resampled=initial_resampled,
        )
        cached_state_timestep = num_inference_steps
        cached_estimate = initial_estimate

        guided_correction_count = 0
        cleaned = False
        iterator = enumerate(reversed(range(num_inference_steps)))
        if not disable_progress_bar:
            iterator = tqdm(iterator, leave=False)

        for i, timestep in iterator:
            source_time = continuous_times[i]
            target_time = continuous_times[i + 1]
            source_scale = lambdas[i] / kl_weight
            target_scale = lambdas[i + 1] / kl_weight
            source_timestep = timestep + 1
            guided = timestep in active_steps

            if verbose:
                print(
                    f"transition {source_timestep}->{timestep}, "
                    f"guided={guided}, source_scale={source_scale}, "
                    f"target_scale={target_scale}"
                )

            source_time_batch = time_tensor(source_time, total_particles)
            target_time_batch = time_tensor(target_time, total_particles)

            if guided:
                if (
                    cached_state_timestep == source_timestep
                    and cached_estimate is not None
                    and cached_estimate.logits is not None
                    and cached_estimate.reward_grad is not None
                    and cached_estimate.logits_are_from_base_model
                ):
                    source_estimate = cached_estimate
                else:
                    source_estimate = estimate_value(
                        latents,
                        source_time,
                        source_scale,
                        need_grad=True,
                    )
                assert source_estimate.logits is not None
                assert source_estimate.reward_grad is not None
                sched_out = self.scheduler.step_with_approx_guidance(
                    latents=latents,
                    logits=source_estimate.logits,
                    approx_guidance=source_estimate.reward_grad * target_scale,
                    t=source_time_batch,
                    next_t=target_time_batch,
                )
                latents = sched_out.new_latents
                transition_log_ratio = (
                    sched_out.log_prob_diffusion
                    - sched_out.log_prob_proposal
                )
            else:
                if (
                    cached_state_timestep == source_timestep
                    and cached_estimate is not None
                    and cached_estimate.logits is not None
                    and cached_estimate.logits_are_from_base_model
                ):
                    base_logits = cached_estimate.logits
                else:
                    base_logits = get_logits(self.model, latents, source_time)
                sched_out = self.scheduler.step(
                    latents=latents,
                    logits=base_logits,
                    t=source_time_batch,
                    next_t=target_time_batch,
                )
                latents = sched_out.new_latents
                transition_log_ratio = torch.zeros_like(log_w)

            cached_state_timestep = None
            cached_estimate = None

            if timestep == 0:
                latents = remove_remaining_noise(latents)
                cleaned = True

            if guided:
                need_target_grad = timestep - 1 in active_steps
                target_estimate = estimate_value(
                    latents,
                    target_time,
                    target_scale,
                    need_grad=need_target_grad,
                    exact=timestep == 0,
                )
                guided_correction_count += 1
                resample_now = (
                    guided_correction_count % resample_frequency == 0
                )
                target_estimate, was_resampled = correct_current_state(
                    target_estimate,
                    transition_log_ratio,
                    resample_now=resample_now,
                )
                append_trace(
                    state_timestep=timestep,
                    state_scale=target_scale,
                    estimate=target_estimate,
                    resampled=was_resampled,
                )
                cached_state_timestep = timestep
                cached_estimate = target_estimate

        if not cleaned:
            latents = remove_remaining_noise(latents)

        # Exact terminal rewards are needed for selection/diagnostics, but they
        # affect SMC weights only when timestep 0 belongs to G.
        if cached_state_timestep == 0 and cached_estimate is not None:
            rewards = cached_estimate.rewards
        else:
            with torch.no_grad():
                rewards = reward_fn(latents).detach()

        try:
            tag = os.environ.get("SMC_TRACE_TAG")
            suffix = f"_{tag}" if tag else ""
            trace_path = f"reward_trace{suffix}.jsonl"
            with open(trace_path, "w", encoding="utf-8") as handle:
                for row in reward_trace:
                    handle.write(json.dumps(row) + "\n")
            with open("reward_trace.jsonl", "w", encoding="utf-8") as handle:
                for row in reward_trace:
                    handle.write(json.dumps(row) + "\n")
        except Exception as exc:
            if verbose:
                print(
                    "[bold yellow]Warning:[/bold yellow] "
                    f"failed to write reward trace: {exc}"
                )

        matrix_log_w = log_w.reshape(batches, num_particles)
        matrix_latents = latents.reshape(
            batches, num_particles, length
        )
        matrix_rewards = rewards.reshape(batches, num_particles)

        if verbose:
            print("Final log weights: ", matrix_log_w)
            print(
                "Final normalized weights: ",
                normalize_weights(matrix_log_w, dim=-1),
            )

        if final_strategy == "all_particles":
            # Produce N equally weighted output slots per independent SMC batch
            # by fully resampling the terminal weighted empirical distribution.
            matrix_latents, _ = systematic_resample_population(
                matrix_latents, matrix_log_w
            )
            return matrix_latents.reshape(
                batches * num_particles, length
            )
        if final_strategy == "multinomial":
            final_indices = torch.multinomial(
                normalize_weights(matrix_log_w, dim=-1),
                num_samples=1,
            ).squeeze(-1)
        elif final_strategy == "argmax_rewards":
            final_indices = matrix_rewards.argmax(dim=-1)
        elif final_strategy == "argmax_weights":
            final_indices = matrix_log_w.argmax(dim=-1)
        else:
            raise NotImplementedError(
                f"Final strategy {final_strategy} is not implemented."
            )

        return matrix_latents[
            torch.arange(batches, device=matrix_latents.device),
            final_indices,
        ]
