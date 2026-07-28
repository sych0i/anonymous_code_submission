from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SchedulerStepOutput:
    new_latents: torch.Tensor


@dataclass
class SchedulerApproxGuidanceOutput:
    new_latents: torch.Tensor
    log_prob_proposal: torch.Tensor
    log_prob_diffusion: torch.Tensor


class BaseScheduler(ABC):
    @abstractmethod
    def step(
        self,
        latents: torch.Tensor,
        logits: torch.Tensor,
        t,
        next_t,
    ) -> SchedulerStepOutput:
        pass

    @abstractmethod
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
        t,
        next_t,
    ) -> SchedulerApproxGuidanceOutput:
        pass


class MDLMScheduler(BaseScheduler):
    
    def __init__(
        self,
        model,
        *,
        reproduce_allstep_numerics: bool = False,
    ) -> None:
        self.model = model
        self.reproduce_allstep_numerics = reproduce_allstep_numerics
    
    def get_q_xs_dist(
        self,
        latents: torch.Tensor,
        logits: torch.Tensor,
        t,
        next_t,
    ):
        sigma_t, _ = self.model.noise(t)
        sigma_s, _ = self.model.noise(next_t)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t) # t
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        log_p_x0 = self.model._subs_parameterization(logits=logits, xt=latents)
        assert move_chance_t.ndim == log_p_x0.ndim
        p_x0 = log_p_x0.exp()
        if self.reproduce_allstep_numerics:
            # The retained all-step reference was generated before the
            # probability validation cleanup. Preserve its two-reduction
            # floating-point path so its traces and derived scores reproduce.
            assert torch.allclose(
                p_x0.sum(dim=-1),
                torch.tensor(1.0, device=p_x0.device),
                atol=6e-5,
            )
            p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)
        else:
            if not torch.isfinite(p_x0).all():
                raise ValueError("p_x0 contains non-finite probabilities")

            p_x0_sum = p_x0.sum(dim=-1, keepdim=True)
            if not torch.isfinite(p_x0_sum).all() or not torch.all(p_x0_sum > 0):
                raise ValueError("p_x0 must have a finite, strictly positive sum")

            # This is the numerical path used for the retained sparse sweep.
            p_x0 = p_x0 / p_x0_sum
            normalized_sum = p_x0.sum(dim=-1)
            if not torch.isfinite(p_x0).all() or not torch.allclose(
                normalized_sum,
                torch.ones_like(normalized_sum),
                atol=1e-6,
                rtol=1e-5,
            ):
                max_error = (normalized_sum - 1.0).abs().max().item()
                raise ValueError(
                    f"p_x0 normalization failed (maximum sum error: {max_error})"
                )
        q_xs = p_x0 * (move_chance_t - move_chance_s) + F.one_hot(latents, num_classes=self.model.vocab_size) * move_chance_s
        q_xs /= move_chance_t
        assert torch.allclose(q_xs.sum(dim=-1), torch.tensor(1.0, device=q_xs.device), atol=1e-6), f"Off by {(q_xs.sum(dim=-1) - 1.0).abs().max()}"
        return torch.distributions.Categorical(probs=q_xs)
    
    def step(
        self,
        latents: torch.Tensor,
        logits: torch.Tensor,
        t,
        next_t,
    ) -> SchedulerStepOutput:
        q_xs_dist = self.get_q_xs_dist(latents, logits, t, next_t)
        new_latents = q_xs_dist.sample()
        return SchedulerStepOutput(new_latents=new_latents)
    
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
        t,
        next_t,
    ) -> SchedulerApproxGuidanceOutput:
        q_xs_diffusion_dist = self.get_q_xs_dist(latents, logits, t, next_t)
        q_xs_proposal_dist = self.get_q_xs_dist(latents, logits + approx_guidance, t, next_t)
        
        new_latents = q_xs_proposal_dist.sample()
        log_prob_proposal = q_xs_proposal_dist.log_prob(new_latents).sum(dim=-1)
        log_prob_diffusion = q_xs_diffusion_dist.log_prob(new_latents).sum(dim=-1)
        
        return SchedulerApproxGuidanceOutput(
            new_latents=new_latents,
            log_prob_proposal=log_prob_proposal,
            log_prob_diffusion=log_prob_diffusion,
        )
