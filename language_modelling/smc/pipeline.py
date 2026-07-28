from typing import Optional, Tuple, Callable, Any, Set
import math
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import DictConfig
from rich import print
from peft import PeftModel

_LM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MDLM_ROOT = os.path.join(_LM_ROOT, "mdlm")
if _MDLM_ROOT not in sys.path:
    sys.path.insert(0, _MDLM_ROOT)

from mdlm import dataloader
from mdlm_diffusion import MDLMDiffusion
from smc.scheduler import BaseScheduler
from smc.resampling import compute_ess_from_log_w, normalize_weights


def normalize_proposal_type(proposal_type: str) -> str:
    aliases = {
        "locally_optimal": "grad",
        "locally-optimal": "grad",
        "SMC grad": "grad",
        "smc grad": "grad",
        "reduced_SMC_grad": "grad",
        "smc_grad": "grad",
        "SMC_grad": "grad",
        "reduced_grad": "grad",
        "reduced_SMC_reverse": "reverse",
        "SMC reverse": "reverse",
        "smc reverse": "reverse",
        "smc_reverse": "reverse",
        "SMC_reverse": "reverse",
        "reduced_reverse": "reverse",
    }
    return aliases.get(str(proposal_type), str(proposal_type))


def _load_from_checkpoint(config, tokenizer, device: torch.device):
    """Load model from checkpoint onto ``device``."""
    if 'hf' in config.backbone:
        return MDLMDiffusion(config, tokenizer=tokenizer).to(device)

    model = MDLMDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )
    return model.to(device)


def logmeanexp(x, dim=None, keepdim=False):
    """Numerically stable log-mean-exp using torch.logsumexp."""
    if dim is None:
        x = x.view(-1)
        dim = 0
    # log-sum-exp with or without keeping the reduced dim
    lse = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    # subtract log(N) to convert sum into mean (broadcasts correctly)
    return lse - math.log(x.size(dim))


class Pipeline:

    def __init__(self, config: DictConfig, scheduler: BaseScheduler, device, dtype=torch.float) -> None:
        self.config = config
        self.tokenizer = dataloader.get_tokenizer(config)
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self._torch_device = device

        # p: pretrained base; q: optional fine-tuned (same as previous single-model behavior when ft is set).
        self.model_p = _load_from_checkpoint(self.config, self.tokenizer, device)
        self.model_q = self.model_p
        if self.config.ft_model.ckpt_path:
            ckpt_path = self.config.ft_model.ckpt_path
            # LoRA adapters are saved as a directory; full finetune is a single .pth file.
            # If the user omits ft_model.lora=True but passes .../lora/, load as PEFT anyway.
            use_peft = bool(self.config.ft_model.lora) or os.path.isdir(ckpt_path)
            if use_peft:
                self.model_q = _load_from_checkpoint(self.config, self.tokenizer, device)
                self.model_q.backbone = PeftModel.from_pretrained(
                    self.model_q.backbone, ckpt_path
                )
            else:
                self.model_q = _load_from_checkpoint(self.config, self.tokenizer, device)
                self.model_q.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded fine-tuned model (q) from {ckpt_path}")
        self.model = self.model_q
        self.model_p.eval()
        self.model_q.eval()

        self._execution_device = self.model.device
        self.scheduler = scheduler
        self.model_dtype = dtype

    @torch.no_grad()
    def __call__(
        self,
        reward_fn: Callable,
        resample_fn: Callable,
        prompt_text: Optional[str] = None,
        resample_frequency: int = 1,
        kl_weight: float = 1.0,
        lambdas: Optional[torch.Tensor] = None,
        num_inference_steps: int = 48,
        num_particles: int = 1,
        batch_p: int = 1,
        phi: int = 1, # number of samples for reward approximation
        tau: float = 1.0, # temperature for taking x0 samples
        proposal_type: str = "grad",
        guidance_steps: Optional[Set[int]] = None,
        use_continuous_formulation: bool = False, # Whether to use a continuous formulation of carry over unmasking
        disable_progress_bar: bool = False,
        verbose=True,
    ):
        original_proposal_type = str(proposal_type)
        legacy_reduced_proposal = original_proposal_type in {
            "reduced_SMC_grad",
            "reduced_SMC_reverse",
            "reduced_grad",
            "reduced_reverse",
        }
        proposal_type = normalize_proposal_type(original_proposal_type)
        ft_model = getattr(self.config, "ft_model", None)
        if (
            ft_model is not None
            and getattr(ft_model, "ckpt_path", None)
            and proposal_type in {"grad", "reverse"}
        ):
            raise ValueError(
                "SMC-grad/SMC-base require the pretrained reference model. "
                "Fine-tuned proposals need an explicit p/q importance "
                "correction and are not supported by these proposal types."
            )

        # Per-step diagnostics (saved to CWD; Hydra sets CWD to run dir)
        reward_trace: list[dict[str, Any]] = []

        # Set default lambdas
        if lambdas is None:
            lambdas = torch.ones(num_inference_steps + 1)
        assert len(lambdas) == num_inference_steps + 1, f"lambdas must of length {num_inference_steps + 1}"
        lambdas = lambdas.clamp_min(0.001).to(self._execution_device)

        # 1. Tokenize prompt
        if prompt_text is not None:
            assert isinstance(prompt_text, str)
            prompt = self.tokenizer([prompt_text], return_tensors='pt', padding=False)
            prompt_ids = prompt['input_ids'][:, :-1].to(self._execution_device) # type: ignore
            # Set prompt length in scheduler if supported
            if hasattr(self.scheduler, 'set_prompt_length'):
                self.scheduler.set_prompt_length(prompt_ids.shape[1]) # type: ignore
            else:
                print("[bold red]Warning:[/bold red] Scheduler does not support setting prompt length. The prompt may be get modified in the text samples.")
        else:
            prompt_ids = None

        # 3. Intialize latents
        latents = self.model._sample_prior(num_particles, self.config.model.length, prompt_ids=prompt_ids).to( # type: ignore
            self._execution_device
        )

        # Set some constant vectors
        vocab_size = self.model.vocab_size
        assert self.scheduler.mask_token_id == self.model.mask_index # type: ignore
        ONE = torch.ones(vocab_size, device=self._execution_device).float()
        MASK = F.one_hot(torch.tensor(self.scheduler.mask_token_id), num_classes=vocab_size).float().to(self._execution_device) # type: ignore

        # 4. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # 5. Set SMC variables. `last_log_twist` belongs to the most recent
        # corrected state on each particle lineage.
        log_w = torch.zeros((num_particles,), device=self._execution_device)
        last_log_twist = torch.zeros((num_particles,), device=self._execution_device)

        if guidance_steps is None:
            active_guidance_steps = set(range(num_inference_steps))
        elif legacy_reduced_proposal:
            active_guidance_steps = set(
                range(num_inference_steps // 2, num_inference_steps)
            )
        else:
            active_guidance_steps = {int(step) for step in guidance_steps}
        invalid_guidance_steps = active_guidance_steps - set(
            range(num_inference_steps)
        )
        if invalid_guidance_steps:
            raise ValueError(
                "guidance_steps must be in [0, num_inference_steps): "
                f"{sorted(invalid_guidance_steps)}"
            )

        def append_value_trace(
            step: int,
            timestep_value: int,
            scale_value,
            log_values: torch.Tensor,
            rewards_value: torch.Tensor,
            resampled: bool,
        ) -> None:
            reward_trace.append(
                {
                    "step": int(step),
                    "timestep": int(timestep_value),
                    "scale_cur": float(scale_value),
                    "scale_next": float(scale_value),
                    "phi": int(phi),
                    "resampled": bool(resampled),
                    "value_estimates": (
                        log_values.detach().float().exp().cpu().reshape(-1).tolist()
                    ),
                    # Kept for readers of legacy trace files. These are rollout
                    # log-reward aggregates, not scores of partially masked text.
                    "reward_aggregated": (
                        rewards_value.detach().float().cpu().reshape(-1).tolist()
                    ),
                }
            )

        def estimate_value(
            state: torch.Tensor,
            time_value: float,
            scale_value: torch.Tensor,
            *,
            need_grad: bool,
        ):
            """Estimate log V_t with J rollouts; optionally differentiate v_t."""
            state_logits = torch.zeros(
                (*state.shape, vocab_size), device=self._execution_device
            )
            state_rewards = torch.zeros(
                state.shape[0], device=self._execution_device
            )
            state_log_values = torch.zeros_like(state_rewards)
            state_grads = (
                torch.zeros(
                    (*state.shape, vocab_size), device=self._execution_device
                )
                if need_grad
                else None
            )

            for j in range(0, state.shape[0], batch_p):
                state_batch = state[j : j + batch_p]
                t_batch = torch.full(
                    (state_batch.shape[0], 1),
                    float(time_value),
                    device=self._execution_device,
                )

                if time_value == 0.0:
                    sample = F.one_hot(
                        state_batch, num_classes=vocab_size
                    ).to(dtype=self.model_dtype)
                    batch_scores = reward_fn(sample).reshape(-1, 1)
                    batch_log_values = (batch_scores[:, 0] * scale_value).detach()
                    batch_rewards = batch_scores[:, 0].detach()
                    state_log_values[j : j + batch_p] = batch_log_values
                    state_rewards[j : j + batch_p] = batch_rewards
                    continue

                grad_context = torch.enable_grad() if need_grad else torch.no_grad()
                with grad_context:
                    state_one_hot = F.one_hot(
                        state_batch, num_classes=vocab_size
                    ).to(dtype=self.model_dtype)
                    if need_grad:
                        state_one_hot.requires_grad_(True)
                    batch_logits = self.model.get_logits(state_one_hot, t_batch)
                    batch_scores = torch.zeros(
                        state_batch.shape[0],
                        phi,
                        device=self._execution_device,
                    )
                    gamma = 1 - (
                        (ONE - MASK) * state_one_hot
                    ).sum(dim=-1, keepdim=True)
                    for phi_i in range(phi):
                        sample = F.gumbel_softmax(
                            batch_logits, tau=tau, hard=True
                        )
                        if use_continuous_formulation:
                            sample = (
                                gamma * sample
                                + (ONE - MASK) * state_one_hot
                            )
                        batch_scores[:, phi_i] = reward_fn(sample)

                    batch_log_values = logmeanexp(
                        batch_scores * scale_value, dim=-1
                    )
                    batch_rewards = batch_log_values / scale_value
                    if need_grad:
                        batch_grads = torch.autograd.grad(
                            outputs=batch_rewards,
                            inputs=state_one_hot,
                            grad_outputs=torch.ones_like(batch_rewards),
                        )[0]

                state_logits[j : j + batch_p] = batch_logits.detach()
                state_log_values[j : j + batch_p] = batch_log_values.detach()
                state_rewards[j : j + batch_p] = batch_rewards.detach()
                if state_grads is not None:
                    state_grads[j : j + batch_p] = batch_grads.detach()

            return state_logits, state_rewards, state_log_values, state_grads

        def reference_step(
            state: torch.Tensor, step_index: int, time_value: float
        ) -> torch.Tensor:
            logits = torch.zeros(
                (*state.shape, vocab_size), device=self._execution_device
            )
            for j in range(0, state.shape[0], batch_p):
                state_batch = state[j : j + batch_p]
                t_batch = torch.full(
                    (state_batch.shape[0], 1),
                    float(time_value),
                    device=self._execution_device,
                )
                logits[j : j + batch_p] = self.model.get_logits(
                    state_batch, t_batch
                ).detach()
            return self.scheduler.step(
                latents=state, step=step_index, logits=logits
            ).new_latents

        if proposal_type not in {
            "grad",
            "reverse",
            "without_SMC",
            "straight_through_gradients",
        }:
            raise NotImplementedError(
                f"Proposal type {proposal_type} is not implemented."
            )

        cached_logits = None
        cached_grads = None
        if proposal_type != "without_SMC":
            boundary_scale = lambdas[0] / kl_weight
            need_boundary_grad = (
                proposal_type in {"grad", "straight_through_gradients"}
                and num_inference_steps - 1 in active_guidance_steps
            )
            (
                cached_logits,
                boundary_rewards,
                boundary_log_values,
                cached_grads,
            ) = estimate_value(
                latents,
                1.0,
                boundary_scale,
                need_grad=need_boundary_grad,
            )
            log_w += boundary_log_values
            boundary_indices, boundary_resampled, log_w = resample_fn(log_w)
            if boundary_resampled:
                latents = latents[boundary_indices]
                boundary_rewards = boundary_rewards[boundary_indices]
                boundary_log_values = boundary_log_values[boundary_indices]
                if cached_logits is not None:
                    cached_logits = cached_logits[boundary_indices]
                if cached_grads is not None:
                    cached_grads = cached_grads[boundary_indices]
            last_log_twist = boundary_log_values
            append_value_trace(
                -1,
                num_inference_steps,
                boundary_scale,
                boundary_log_values,
                boundary_rewards,
                boundary_resampled,
            )

        bar = enumerate(reversed(range(num_inference_steps)))
        if not disable_progress_bar:
            bar = tqdm(bar, leave=False)
        for i, timestep in bar:
            source_time = (timestep + 1) / num_inference_steps
            target_time = timestep / num_inference_steps
            scale_cur = lambdas[i] / kl_weight
            scale_next = lambdas[i + 1] / kl_weight
            guided = timestep in active_guidance_steps
            resample_condition = (i + 1) % resample_frequency == 0

            if verbose:
                print(
                    f"timestep={timestep}, guided={guided}, "
                    f"scale_cur={scale_cur}, scale_next={scale_next}"
                )

            if proposal_type == "without_SMC" or not guided:
                latents = reference_step(latents, i, source_time)
                cached_logits = None
                cached_grads = None
                continue

            if proposal_type in {"grad", "straight_through_gradients"}:
                if cached_logits is None or cached_grads is None:
                    (
                        cached_logits,
                        _,
                        _,
                        cached_grads,
                    ) = estimate_value(
                        latents,
                        source_time,
                        scale_cur,
                        need_grad=True,
                    )
                sched_out = self.scheduler.step_with_approx_guidance(
                    latents=latents,
                    step=i,
                    logits=cached_logits,
                    approx_guidance=cached_grads * scale_next,
                )
                proposed_latents = sched_out.new_latents
                log_prob_proposal = sched_out.log_prob_proposal
                log_prob_diffusion = sched_out.log_prob_diffusion
            else:
                logits = torch.zeros(
                    (*latents.shape, vocab_size), device=self._execution_device
                )
                for j in range(0, num_particles, batch_p):
                    latents_batch = latents[j : j + batch_p]
                    t_batch = torch.full(
                        (latents_batch.shape[0], 1),
                        float(source_time),
                        device=self._execution_device,
                    )
                    logits[j : j + batch_p] = self.model.get_logits(
                        latents_batch, t_batch
                    ).detach()
                proposed_latents = self.scheduler.step(
                    latents=latents, step=i, logits=logits
                ).new_latents
                log_prob_proposal = torch.zeros_like(log_w)
                log_prob_diffusion = torch.zeros_like(log_w)

            need_target_grad = (
                proposal_type in {"grad", "straight_through_gradients"}
                and timestep > 0
                and timestep - 1 in active_guidance_steps
            )
            (
                target_logits,
                target_rewards,
                target_log_values,
                target_grads,
            ) = estimate_value(
                proposed_latents,
                target_time,
                scale_next,
                need_grad=need_target_grad,
            )

            incremental_log_w = (
                log_prob_diffusion
                - log_prob_proposal
                + target_log_values
                - last_log_twist
            )
            log_w += incremental_log_w
            if verbose:
                print("Incremental log weights: ", incremental_log_w)
                print("ESS: ", compute_ess_from_log_w(log_w))

            resampled_this_step = False
            if resample_condition:
                resample_indices, resampled_this_step, log_w = resample_fn(
                    log_w
                )
                if resampled_this_step:
                    proposed_latents = proposed_latents[resample_indices]
                    target_rewards = target_rewards[resample_indices]
                    target_log_values = target_log_values[resample_indices]
                    target_logits = target_logits[resample_indices]
                    if target_grads is not None:
                        target_grads = target_grads[resample_indices]

            latents = proposed_latents
            last_log_twist = target_log_values
            cached_logits = target_logits if need_target_grad else None
            cached_grads = target_grads if need_target_grad else None
            append_value_trace(
                i,
                timestep,
                scale_next,
                target_log_values,
                target_rewards,
                resampled_this_step,
            )

        # Write diagnostics
        try:
            tag = os.environ.get("SMC_TRACE_TAG")
            suffix = f"_{tag}" if tag else ""
            trace_path = f"reward_trace{suffix}.jsonl"
            with open(trace_path, "w") as f:
                for row in reward_trace:
                    f.write(json.dumps(row) + "\n")
            # Convenience pointer to latest
            with open("reward_trace.jsonl", "w") as f:
                for row in reward_trace:
                    f.write(json.dumps(row) + "\n")
        except Exception as e:
            if verbose:
                print(f"[bold yellow]Warning:[/bold yellow] failed to write reward_trace.jsonl: {e}")

        # Decode latents
        text = self.decode_latents(latents)
        return latents, text

    def decode_latents(self, latents):
        return self.model.tokenizer.batch_decode(latents)
