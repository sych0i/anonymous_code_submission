"""VISTA-SMC "Reduced SMC Guidance" sampler, SMC-base proposal family.

Implements Algorithm 2 / Algorithm 5 of "Value-Informed Schedule Trimming for
Accelerated Sequential Monte Carlo Guidance in Discrete Diffusion", including
the Section 4.3 warmup + DP schedule search (`run_warmup`,
`solve_vista_schedule`, `search_schedule`) alongside manually-specified
guided-step sets G (`first`/`last`/`middle`/`uniform`/`full`). `main()` can
evaluate several schedules back to back (`--schedules`) and report reward /
success-rate for each, mirroring the paper's Table 3-5 baseline comparisons.

Key simplifications specific to this SMC-base variant:
  - Proposal q_{t|t+1} = pretrained transition p_{t|t+1} (`method='base'`), so
    the p/q importance ratio in the Algorithm 5 weight update is always 1 and
    guidance acts purely through value-based reweighting/resampling.
  - Value estimation V_t(xt) = E_{x0~p_{0|t}(.|xt)}[R(x0)] is computed with a
    *single* forward pass rather than a multi-step ancestral rollout. For the
    'subs' (absorbing-state) parameterization, `model.forward` already
    returns log p_theta(x0|xt) with each token predicted independently and
    already-revealed tokens pinned to their current value ("copy-over"), i.e.
    exactly the model's p_{0|t}(x0|xt). Drawing J iid samples from that
    distribution and averaging the reward *is* the exact rollout the paper's
    Algorithm 4/5 describes for this model family -- not an approximation of
    it -- and it reproduces Table 2's O(N*T) transition-model cost (one
    forward pass per particle per guided step) plus O(N*T*J) reward calls,
    rather than the O(N*T^2) that a literal re-run of the reverse chain down
    to t=0 would cost.
  - Resampling matches the implementation underlying the SMC proposal
    family used by the paper: adaptive ESS-triggered systematic partial
    resampling. The selected subset consists of the highest-weight particle
    and the `partial_resample - 1` lowest-weight particles. Its normalized
    mass is redistributed uniformly while unselected particles retain their
    normalized weights.
  - `sample()` returns every one of the N final particles {x_0^(i)} per
    requested generation, not a single selected particle: output row count
    per schedule is `--total-samples x --num-particles`. The
    `--final-selection` CLI arg and plumbing (and `run_warmup`'s
    `final_selection`) are kept for backward compatibility but no longer
    select anything.

SMC-grad proposal (`proposal_method='grad'`, Section 3.3):
  - Implements q proportional to p * exp((1/alpha) grad(v) dot x).
  - V is differentiated through J hard straight-through Gumbel-softmax
    samples from p_theta(x_0|x_t), the reward classifier, and the diffusion
    backbone, matching the reference SMC-grad estimator.
  - CIFAR's UNet maps categorical pixel c to the scalar c before affine
    rescaling. Thus the exact one-hot chain rule is
    d log(V)/d one_hot[c] = d log(V)/d scalar * c, avoiding a materialized
    (batch, 3072, 256) one-hot input.
"""

import argparse
import json
import os
import time
import typing

import einops
import lightning as L
import lpips
import omegaconf
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

import dataloader
import diffusion
import reward_classifier
from smc import (
  _effective_sample_size, _gather_transition_log_prob, _guidance_step_indices)


def _register_resolvers():
  resolvers = {
    'cwd': os.getcwd,
    'device_count': torch.cuda.device_count,
    'eval': eval,
    'div_up': lambda x, y: (x + y - 1) // y,
    'if_then_else': lambda condition, x, y: x if condition else y,
  }
  for name, resolver in resolvers.items():
    if not omegaconf.OmegaConf.has_resolver(name):
      omegaconf.OmegaConf.register_new_resolver(name, resolver)


def _load_config(path):
  _register_resolvers()
  config = omegaconf.OmegaConf.load(path)
  omegaconf.OmegaConf.set_struct(config, False)
  return config


def load_reward_classifier(checkpoint_path: str, device: torch.device):
  ckpt = torch.load(checkpoint_path, map_location=device)
  architecture = ckpt.get('architecture', 'small-cnn')
  if architecture == 'small-cnn':
    model = reward_classifier.SmallCNNClassifier(num_classes=10)
  elif architecture == 'cifar-resnet18':
    model = reward_classifier.CIFARResNet18()
  else:
    raise ValueError(
      f'Unsupported reward-classifier architecture {architecture!r} in '
      f'{checkpoint_path}.')
  model.load_state_dict(ckpt['model_state_dict'])
  model.to(device).eval()
  for param in model.parameters():
    param.requires_grad_(False)
  return model, ckpt['classes']


def load_lpips_model(device: torch.device) -> torch.nn.Module:
  model = lpips.LPIPS(net='alex').to(device).eval()
  for param in model.parameters():
    param.requires_grad_(False)
  return model


@torch.no_grad()
def _reward(
  classifier: torch.nn.Module,
  tokenizer,
  x0_tokens: torch.Tensor,
  target_class: int,
  alpha: float,
) -> torch.Tensor:
  """R(x0) = exp(r(x0)/alpha), r(x0) = log p(target_class | x0)."""
  images = (
    tokenizer.batch_decode(x0_tokens).float() / 255.0).clamp_(0.0, 1.0)
  log_probs = classifier.log_prob(images)
  r = log_probs[:, target_class]
  return (r / alpha).exp()


@torch.no_grad()
def _score_samples(
  classifier: torch.nn.Module,
  tokenizer,
  x0_tokens: torch.Tensor,
  target_class: int,
  alpha: float,
  device: torch.device,
) -> typing.Tuple[torch.Tensor, torch.Tensor, float]:
  """Decodes final particles to images and scores them against the reward
  classifier. Returns (images, reward, success_rate)."""
  images = (
    tokenizer.batch_decode(x0_tokens).float() / 255.0).clamp_(0.0, 1.0)
  log_probs = classifier.log_prob(images.to(device))
  reward = (log_probs[:, target_class] / alpha).exp()
  predicted = log_probs.argmax(dim=-1)
  success_rate = (predicted == target_class).float().mean().item()
  return images, reward, success_rate


def _unconditional_class_condition(
  model: diffusion.Diffusion,
  batch_size: int,
  condition_mode: str = 'auto',
) -> typing.Optional[torch.Tensor]:
  """Return the unconditional class token used during CFG training.

  A class/CFG-trained checkpoint represents its unconditional branch with the
  extra masked-class embedding at index ``num_classes``. ``cond=None`` bypasses
  class conditioning altogether and is an out-of-distribution path for that
  model. Non-CFG checkpoints retain their native ``cond=None`` behavior.
  """
  if condition_mode == 'none':
    return None
  if condition_mode != 'auto':
    raise ValueError(
      f'Unknown value condition mode {condition_mode!r}.')
  if model.config.training.guidance is None:
    return None
  return torch.full(
    (batch_size,),
    model.config.data.num_classes,
    dtype=torch.long,
    device=model.device)


@torch.no_grad()
def _unique_success_count(
  images: torch.Tensor,
  reward: torch.Tensor,
  reward_threshold: float,
  lpips_model: torch.nn.Module,
  lpips_threshold: float,
) -> typing.Dict[str, typing.Any]:
  """Diversity-aware success metric: among samples whose `reward` exceeds
  `reward_threshold`, greedily collapses near-duplicates -- two images are
  treated as the same sample if their LPIPS (Learned Perceptual Image Patch
  Similarity; Zhang et al. 2018) distance is <= `lpips_threshold` -- and
  returns the number of remaining unique samples. LPIPS compares AlexNet
  activations rather than raw pixels, so two samples that merely share
  overall brightness/color (which previously fooled a raw-pixel cosine
  similarity into reading them as near-1.0 similar, e.g. a mostly-sky mdlm
  sample) are not conflated unless they are also perceptually alike.

  Clustering is a single left-to-right greedy pass (each sample joins the
  first earlier "representative" it's similar enough to, else becomes a
  new representative), not a full transitive-closure clustering; this is
  an explicit, cheap choice rather than an exact algorithm."""
  reward_for_index = reward.detach().to(images.device)
  passed_idx = (reward_for_index > reward_threshold).nonzero(as_tuple=True)[0]
  num_passed = int(passed_idx.numel())
  if num_passed == 0:
    return {
      'reward_threshold': reward_threshold,
      'lpips_threshold': lpips_threshold,
      'num_passed_reward_threshold': 0,
      'num_unique_successes': 0,
    }

  lpips_device = next(lpips_model.parameters()).device
  passed_images = images[passed_idx].to(lpips_device)
  # Compare each candidate only with previously accepted representatives.
  # This is exactly the same greedy clustering rule as constructing the full
  # N x N distance matrix, but bounds LPIPS activation memory for large runs.
  representative_indices = []
  max_lpips_batch_size = 256
  for i in range(num_passed):
    is_duplicate = False
    for start in range(0, len(representative_indices),
                       max_lpips_batch_size):
      chunk_indices = representative_indices[
        start:start + max_lpips_batch_size]
      reference_images = passed_images[chunk_indices]
      candidate_images = passed_images[i:i + 1].expand_as(reference_images)
      distances = lpips_model(
        reference_images, candidate_images, normalize=True).reshape(-1)
      if bool((distances <= lpips_threshold).any()):
        is_duplicate = True
        break
    if not is_duplicate:
      representative_indices.append(i)

  return {
    'reward_threshold': reward_threshold,
    'lpips_threshold': lpips_threshold,
    'num_passed_reward_threshold': num_passed,
    'num_unique_successes': len(representative_indices),
  }


@torch.no_grad()
def _estimate_value(
  model: diffusion.Diffusion,
  xt: torch.Tensor,
  time_conditioning: torch.Tensor,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_rollout_samples: int,
  condition_mode: str = 'auto',
) -> torch.Tensor:
  """V_t(xt) = E_{x0 ~ p_theta(x0|xt)}[R(x0)], estimated via J iid samples
  from the model's one-step clean-data prediction (see module docstring)."""
  value_cond = _unconditional_class_condition(
    model, xt.shape[0], condition_mode=condition_mode)
  log_x_theta = model.forward(
    xt, time_conditioning, cond=value_cond)
  if model.config.sampling.use_float64:
    log_x_theta = log_x_theta.to(torch.float64)
  probs = log_x_theta.exp()
  rewards = []
  for _ in range(num_rollout_samples):
    x0_sample = diffusion._sample_categorical(probs)
    rewards.append(_reward(
      classifier, model.tokenizer, x0_sample, target_class, alpha))
  return torch.stack(rewards, dim=0).mean(dim=0)


@torch.no_grad()
def _estimate_value_multistep(
  model: diffusion.Diffusion,
  xt: torch.Tensor,
  step_idx: int,
  total_steps: int,
  eps: float,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_rollout_samples: int,
) -> torch.Tensor:
  """V_t(xt) = E_{x0 ~ p_{0|t}(.|xt)}[R(x0)] via genuine multi-step
  ancestral rollout (ordinary unconditional generation continuing from xt),
  replicated `num_rollout_samples` times, rather than a single forward
  pass. `xt` is the state produced by reverse step `step_idx` (0-indexed,
  code convention: step_idx=0 is chronologically first), so
  `total_steps - step_idx - 1` reverse steps remain until x0. Pass
  step_idx=-1 to roll out from the untouched prior (all `total_steps`
  steps remain).

  The single-forward-pass estimate (`_estimate_value`) is exactly the
  model's *defined* p_{0|t}(x0|xt) under its per-token-independence
  assumption, but that assumption is a poor approximation at high-noise,
  partially-revealed states for spatially-correlated data (natural images):
  the resulting one-shot "x0" samples are patchy/incoherent, which biases
  the reward classifier's output in ways that don't track actual
  downstream generation quality. This genuine rollout costs O(remaining
  steps) forward passes instead of one, but matches what the classifier
  was actually trained to score (real, fully-generated images).
  """
  batch_size = xt.shape[0]
  remaining_steps = total_steps - step_idx - 1
  if remaining_steps <= 0:
    return _reward(classifier, model.tokenizer, xt, target_class, alpha)

  rolled = xt.repeat_interleave(num_rollout_samples, dim=0)
  timesteps = torch.linspace(1, eps, total_steps + 1, device=model.device)
  dt = (1 - eps) / total_steps
  for k in range(step_idx + 1, total_steps):
    t = timesteps[k]
    if model.T > 0:
      t = (t * model.T).to(torch.int)
      t = t / model.T
      t += (1 / model.T)
    t = t * torch.ones(rolled.shape[0], 1, device=model.device)
    sigma_t, _ = model.noise(t)
    sigma_s, _ = model.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
    move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]
    q_xs = model._diffusion_transition_probs(
      xt=rolled, time_conditioning=sigma_t,
      move_chance_t=move_chance_t, move_chance_s=move_chance_s,
      method='base')
    rolled = diffusion._sample_categorical(q_xs)

  reward = _reward(classifier, model.tokenizer, rolled, target_class, alpha)
  return reward.view(batch_size, num_rollout_samples).mean(dim=1)


def _estimate_value_grad(
  model: diffusion.Diffusion,
  xt: torch.Tensor,
  time_conditioning: torch.Tensor,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_rollout_samples: int,
  condition_mode: str = 'auto',
  gumbel_tau: float = 1.0,
  grad_batch_size: typing.Optional[int] = 20,
) -> torch.Tensor:
  """Estimate ``grad log V_t(xt)`` with the paper's SMC-grad estimator.

  Each of J hard straight-through Gumbel-softmax clean samples is
  differentiated through the reward classifier and diffusion backbone.
  Since ``log V = logmeanexp(r / alpha)``, the result is the
  softmax-weighted average of ``grad r / alpha``. Sequential backward passes
  and optional batch chunking keep peak activation memory bounded.
  """
  if alpha <= 0:
    raise ValueError(f'alpha must be positive, got {alpha}.')
  if num_rollout_samples <= 0:
    raise ValueError(
      'num_rollout_samples must be positive, got '
      f'{num_rollout_samples}.')
  if gumbel_tau <= 0:
    raise ValueError(f'gumbel_tau must be positive, got {gumbel_tau}.')

  batch_size = xt.shape[0]
  if grad_batch_size is None or grad_batch_size <= 0:
    grad_batch_size = batch_size
  gradients = []
  num_pixel_values = model.mask_index

  for start in range(0, batch_size, grad_batch_size):
    stop = min(start + grad_batch_size, batch_size)
    xt_chunk = xt[start:stop]
    time_chunk = time_conditioning[start:stop]
    xt_scalar = xt_chunk.to(torch.float32).detach().requires_grad_(True)
    sigma = model._process_sigma(time_chunk)
    value_cond = _unconditional_class_condition(
      model, xt_chunk.shape[0], condition_mode=condition_mode)

    reward_samples, reward_gradients = [], []
    with torch.enable_grad():
      logits = model.backbone(xt_scalar, sigma, value_cond)
      log_x_theta = model._subs_parameterization(
        logits=logits, xt=xt_chunk)
      pixel_logits = log_x_theta[..., :num_pixel_values]
      pixel_values = torch.arange(
        num_pixel_values, device=xt.device, dtype=pixel_logits.dtype)

      for rollout_idx in range(num_rollout_samples):
        x0_one_hot = F.gumbel_softmax(
          pixel_logits, tau=gumbel_tau, hard=True, dim=-1)
        x0_scalar = (x0_one_hot * pixel_values).sum(dim=-1)
        soft_image = einops.rearrange(
          x0_scalar, 'b (c h w) -> b c h w', c=3, h=32, w=32)
        soft_image = soft_image.to(torch.float32) / 255.0
        raw_reward = classifier.log_prob(soft_image)[:, target_class]
        grad_reward, = torch.autograd.grad(
          raw_reward.sum(),
          xt_scalar,
          retain_graph=rollout_idx + 1 < num_rollout_samples)
        reward_samples.append(raw_reward.detach())
        reward_gradients.append(grad_reward.detach())

    stacked_rewards = torch.stack(reward_samples, dim=0)
    mixture_weights = (stacked_rewards / alpha).softmax(dim=0)
    stacked_gradients = torch.stack(reward_gradients, dim=0)
    grad_log_value = (
      mixture_weights[..., None] * stacked_gradients).sum(dim=0) / alpha
    gradients.append(grad_log_value)

  result = torch.cat(gradients, dim=0)
  if not torch.isfinite(result).all():
    raise FloatingPointError('SMC-grad produced a non-finite value gradient.')
  return result


def _grad_twisted_probs(
  model: diffusion.Diffusion,
  base_probs: torch.Tensor,
  grad_xt: torch.Tensor,
  xt: torch.Tensor,
  gamma_grad: float,
) -> torch.Tensor:
  """Apply the paper's linear categorical Taylor twist.

  The UNet consumes token ``c`` as a scalar, hence the one-hot score is
  ``grad_xt * c`` by the chain rule. Centering c at the mask index only
  subtracts a position-wise constant and is exactly softmax-invariant.
  ``xt`` remains in the signature to make the proposal/state relationship
  explicit and to validate its shape.
  """
  if model.diffusion != 'absorbing_state':
    raise NotImplementedError(
      "SMC-grad proposal is only implemented for absorbing_state "
      f"diffusion, got diffusion={model.diffusion!r}.")
  if xt.shape != base_probs.shape[:-1]:
    raise ValueError(
      f'xt shape {tuple(xt.shape)} does not match transition shape '
      f'{tuple(base_probs.shape[:-1])}.')
  if grad_xt.shape != base_probs.shape[:-1]:
    raise ValueError(
      f'grad_xt shape {tuple(grad_xt.shape)} does not match transition '
      f'shape {tuple(base_probs.shape[:-1])}.')
  if gamma_grad == 0:
    return base_probs.clone()
  if not torch.isfinite(grad_xt).all():
    raise FloatingPointError('Cannot twist with a non-finite gradient.')

  token_values = torch.arange(
    base_probs.shape[-1],
    device=base_probs.device,
    dtype=base_probs.dtype) - model.mask_index
  taylor_score = (
    gamma_grad * grad_xt.to(base_probs.dtype))[..., None] * token_values
  negative_infinity = torch.full_like(base_probs, float('-inf'))
  base_log_probs = torch.where(
    base_probs > 0, base_probs.log(), negative_infinity)
  twisted = (base_log_probs + taylor_score).softmax(dim=-1)
  # Structural zeros encode impossible transitions and must remain exact.
  twisted = torch.where(base_probs > 0, twisted, torch.zeros_like(twisted))
  return twisted / twisted.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def _systematic_resample_k(weights: torch.Tensor, k: int) -> torch.Tensor:
  """Draws k ancestor indices per batch row via systematic resampling from
  the supplied normalized distribution."""
  batch_size, _ = weights.shape
  positions = (
    torch.rand(batch_size, 1, device=weights.device)
    + torch.arange(k, device=weights.device)
  ) / k
  cumulative = weights.cumsum(dim=1)
  cumulative[:, -1] = 1.0
  return torch.stack([
    torch.searchsorted(cumulative[i], positions[i], right=True)
    for i in range(batch_size)
  ])


def _maybe_partial_resample(
  xt: torch.Tensor,
  log_weights: torch.Tensor,
  last_guided_value: torch.Tensor,
  ess_threshold: float,
  partial_resample: int,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Adaptive systematic partial resampling used by the SMC reference.

  When ESS is low, select the highest-weight particle and the ``k - 1``
  lowest-weight particles. Systematically resample ancestors only within
  this subset, assign every selected output slot the subset's total
  normalized mass divided by k, and retain all other normalized weights.
  This preserves total mass and is invariant to a common log-weight shift.
  """
  batch_size, num_particles = log_weights.shape
  if partial_resample <= 0:
    raise ValueError(
      f'partial_resample must be positive, got {partial_resample}.')
  ess = _effective_sample_size(log_weights)
  should_resample = ess < (ess_threshold * num_particles)
  if not should_resample.any():
    return xt, log_weights, last_guided_value, should_resample

  k = min(partial_resample, num_particles)
  normalized_log_weights = log_weights.log_softmax(dim=1)
  weights = normalized_log_weights.exp()
  if k == 1:
    subset_idx = weights.argmax(dim=1, keepdim=True)
  else:
    high_idx = weights.argmax(dim=1, keepdim=True)
    low_idx = torch.topk(
      weights, k - 1, dim=1, largest=False).indices
    subset_idx = torch.cat([high_idx, low_idx], dim=1)

  subset_log_weights = normalized_log_weights.gather(1, subset_idx)
  subset_weights = subset_log_weights.softmax(dim=1)
  local_ancestor_idx = _systematic_resample_k(subset_weights, k)
  ancestor_idx = subset_idx.gather(1, local_ancestor_idx)

  identity_idx = torch.arange(
    num_particles, device=xt.device).expand(batch_size, -1).clone()
  resample_map = identity_idx.scatter(
    dim=1, index=subset_idx, src=ancestor_idx)
  resample_map = torch.where(
    should_resample[:, None], resample_map, identity_idx)
  grouped_xt = xt.view(batch_size, num_particles, xt.shape[-1])
  new_xt = grouped_xt.gather(
    dim=1,
    index=resample_map[..., None].expand(-1, -1, xt.shape[-1]),
  ).reshape_as(xt)
  new_value = last_guided_value.gather(dim=1, index=resample_map)

  subset_mass = weights.gather(1, subset_idx).sum(
    dim=1, keepdim=True)
  uniform_subset_log_weight = (subset_mass / k).log().expand(-1, k)
  resampled_log_weights = normalized_log_weights.scatter(
    dim=1, index=subset_idx, src=uniform_subset_log_weight)
  new_log_weights = torch.where(
    should_resample[:, None],
    resampled_log_weights,
    log_weights)
  return new_xt, new_log_weights, new_value, should_resample


def _flatten_all_particles(
  xt: torch.Tensor, batch_size: int, num_particles: int,
) -> torch.Tensor:
  """Returns every one of the N particles per batch row (Algorithm 5's
  final particle set {x_0^(i)}), flattened to (batch_size * num_particles,
  seq_len) instead of collapsing to a single selected particle per row."""
  return xt.view(batch_size * num_particles, xt.shape[-1])


@torch.no_grad()
def sample(
  model: diffusion.Diffusion,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_particles: int,
  num_rollout_samples: int,
  ess_threshold: float,
  partial_resample: int,
  guidance_steps: int,
  guidance_schedule: str,
  final_selection: str,
  total_steps: int,
  batch_size: int,
  num_batches: int,
  eps: float,
  explicit_guidance_steps: typing.Optional[typing.Iterable[int]] = None,
  value_estimation: str = 'one-shot',
  proposal_method: str = 'base',
  grad_scale: float = 1.0,
  record_final_pool_diagnostics: bool = False,
  value_condition_mode: str = 'auto',
  gumbel_tau: float = 1.0,
  grad_batch_size: typing.Optional[int] = 20,
):
  """Returns every one of the N final particles per requested sample
  (Algorithm 5's full {x_0^(i)} particle set), flattened to
  (num_batches * batch_size * num_particles, seq_len) -- not a single
  particle selected per row. `final_selection` is kept for call-site/CLI
  compatibility but no longer selects anything."""
  del final_selection
  if proposal_method not in ('base', 'grad'):
    raise ValueError(
      f"Unknown proposal_method={proposal_method!r}. Use 'base' or 'grad'.")
  if explicit_guidance_steps is not None:
    guidance_step_set = set(explicit_guidance_steps)
  else:
    guidance_step_set = set(_guidance_step_indices(
      total_steps=total_steps,
      guidance_steps=guidance_steps,
      schedule=guidance_schedule))

  output_batches, all_stats = [], []
  model.load_ema_params()
  try:
    for batch_idx in tqdm(range(num_batches), desc='VISTA-SMC batches'):
      particle_batch_size = batch_size * num_particles
      xt = model._sample_prior(
        particle_batch_size, model.config.model.length).to(model.device)

      timesteps = torch.linspace(1, eps, total_steps + 1, device=model.device)
      dt = (1 - eps) / total_steps

      # Algorithm 1 lines 1-8 / Algorithm 4 lines 1-8: the t=T boundary
      # importance-weight-and-resample step is unconditional (independent of
      # the guided-step set G) -- estimate a per-particle V_hat_T at the
      # untouched prior, weight w_T^(i) proportional to V_hat_T^(i) (q_T=p_T
      # here, so the p/q ratio is exactly 1), and resample, mirroring what
      # every guided step below does with V_hat_t. `model._sample_prior` is
      # deterministic for absorbing_state diffusion (always the all-mask
      # state), so this only matters through the Monte Carlo noise in each
      # particle's own J-rollout value estimate, but the paper's Algorithm
      # 1/4 performs it regardless. `t0` is exactly `timesteps[0]`, i.e. the
      # same "own state" noise level step_idx=0 below computes as its
      # `sigma_t`, since the prior *is* step 0's input state.
      t0 = timesteps[0]
      if model.T > 0:
        t0 = (t0 * model.T).to(torch.int)
        t0 = t0 / model.T
        t0 += (1 / model.T)
      t0 = t0 * torch.ones(xt.shape[0], 1, device=model.device)
      sigma_prior, _ = model.noise(t0)
      if sigma_prior.ndim > 1:
        sigma_prior = sigma_prior.squeeze(-1)
      if value_estimation == 'one-shot':
        prior_value = _estimate_value(
          model=model, xt=xt, time_conditioning=sigma_prior,
          classifier=classifier, target_class=target_class,
          alpha=alpha, num_rollout_samples=num_rollout_samples,
          condition_mode=value_condition_mode,
        ).view(batch_size, num_particles)
      else:
        prior_value = _estimate_value_multistep(
          model=model, xt=xt, step_idx=-1, total_steps=total_steps,
          eps=eps, classifier=classifier, target_class=target_class,
          alpha=alpha, num_rollout_samples=num_rollout_samples,
        ).view(batch_size, num_particles)
      log_weights = prior_value.clamp_min(1e-30).log()
      last_guided_value = prior_value
      xt, log_weights, last_guided_value, _ = _maybe_partial_resample(
        xt, log_weights, last_guided_value, ess_threshold, partial_resample)

      batch_stats = []

      for step_idx in tqdm(
          range(total_steps), desc='VISTA-SMC sampling', leave=False):
        guidance_active = step_idx in guidance_step_set
        t = timesteps[step_idx]
        if model.T > 0:
          t = (t * model.T).to(torch.int)
          t = t / model.T
          t += (1 / model.T)
        t = t * torch.ones(xt.shape[0], 1, device=model.device)

        sigma_t, _ = model.noise(t)
        sigma_s, _ = model.noise(t - dt)
        if sigma_t.ndim > 1:
          sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
          sigma_s = sigma_s.squeeze(-1)
        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

        base_probs = model._diffusion_transition_probs(
          xt=xt, time_conditioning=sigma_t,
          move_chance_t=move_chance_t, move_chance_s=move_chance_s,
          method='base')

        # Per Algorithm 2, only guided steps use the (possibly twisted)
        # proposal q_{t|t+1}; unguided steps always propagate with the
        # unconditional reference transition p_{t|t+1} (`base_probs`).
        use_grad_proposal = guidance_active and proposal_method == 'grad'
        if use_grad_proposal:
          grad_xt = _estimate_value_grad(
            model=model, xt=xt, time_conditioning=sigma_t,
            classifier=classifier, target_class=target_class, alpha=alpha,
            num_rollout_samples=num_rollout_samples,
            condition_mode=value_condition_mode, gumbel_tau=gumbel_tau,
            grad_batch_size=grad_batch_size)
          q_xs = _grad_twisted_probs(
            model=model, base_probs=base_probs, grad_xt=grad_xt,
            xt=xt, gamma_grad=grad_scale)
        else:
          q_xs = base_probs
        xs = diffusion._sample_categorical(q_xs)

        if guidance_active:
          if use_grad_proposal:
            # Algorithm 5's importance-weight update includes a p/q ratio
            # that is folded in here (log p(xs) - log q(xs)); it is exactly
            # 0 when proposal_method='base' since q_xs is base_probs there.
            log_weights = log_weights + (
              _gather_transition_log_prob(base_probs, xs)
              - _gather_transition_log_prob(q_xs, xs)
            ).view(batch_size, num_particles)

          if value_estimation == 'one-shot':
            value_t = _estimate_value(
            model=model, xt=xs, time_conditioning=sigma_s,
            classifier=classifier, target_class=target_class,
            alpha=alpha, num_rollout_samples=num_rollout_samples,
            condition_mode=value_condition_mode,
          ).view(batch_size, num_particles)
          else:
            value_t = _estimate_value_multistep(
              model=model, xt=xs, step_idx=step_idx, total_steps=total_steps,
              eps=eps, classifier=classifier, target_class=target_class,
              alpha=alpha, num_rollout_samples=num_rollout_samples,
            ).view(batch_size, num_particles)

          log_weights = (
            log_weights + value_t.clamp_min(1e-30).log()
            - last_guided_value.clamp_min(1e-30).log())
          last_guided_value = value_t

          ess_before = _effective_sample_size(log_weights)
          xs, log_weights, last_guided_value, resampled = (
            _maybe_partial_resample(
              xs, log_weights, last_guided_value,
              ess_threshold, partial_resample))
        else:
          ess_before = _effective_sample_size(log_weights)
          resampled = torch.zeros(
            batch_size, dtype=torch.bool, device=model.device)

        xt = xs
        batch_stats.append({
          'step': step_idx,
          'guidance_active': guidance_active,
          'mean_ess': ess_before.mean().item(),
          'min_ess': ess_before.min().item(),
          'resampled_batches': int(resampled.sum().item()),
          # Algorithm 4/5 line 14 computes V_hat_t^(i) strictly *before* the
          # resample at line 17; that pre-resample per-particle estimate
          # (value_t) is exactly what Eq. (11)'s V_hat_t^(i,m) refers to, so
          # log it here rather than last_guided_value, which has already
          # been overwritten by this step's post-resample duplication.
          'mean_value': value_t.mean().item() if guidance_active else None,
        })

      final_pool_diagnostics = None
      if record_final_pool_diagnostics:
        final_images = (
          model.tokenizer.batch_decode(xt).float() / 255.0).clamp_(0.0, 1.0)
        final_log_probs = classifier.log_prob(final_images)
        final_target_probabilities = (
          final_log_probs[:, target_class].exp().view(
            batch_size, num_particles))
        final_predictions = final_log_probs.argmax(dim=-1).view(
          batch_size, num_particles)

        final_weights = log_weights.softmax(dim=1)
        final_successes = (
          final_predictions == target_class).to(final_weights.dtype)
        centered_weights = (
          final_weights - final_weights.mean(dim=1, keepdim=True))
        centered_probabilities = (
          final_target_probabilities
          - final_target_probabilities.mean(dim=1, keepdim=True))
        correlation_denominator = (
          centered_weights.square().sum(dim=1)
          * centered_probabilities.square().sum(dim=1)
        ).sqrt()
        correlations = (
          centered_weights * centered_probabilities).sum(dim=1)
        correlations = correlations / correlation_denominator.clamp_min(
          1e-12)

        weight_best_idx = log_weights.argmax(dim=1)
        reward_best_idx = final_target_probabilities.argmax(dim=1)
        row_idx = torch.arange(batch_size, device=xt.device)
        final_pool_diagnostics = {
          'mean_target_probability': (
            final_target_probabilities.mean().item()),
          'weighted_mean_target_probability': (
            (final_weights * final_target_probabilities)
            .sum(dim=1).mean().item()),
          'argmax_success_fraction': final_successes.mean().item(),
          'weighted_argmax_success_probability': (
            (final_weights * final_successes).sum(dim=1).mean().item()),
          'oracle_argmax_success_rate': (
            final_successes.bool().any(dim=1).float().mean().item()),
          'weight_best_argmax_success_rate': (
            final_successes[row_idx, weight_best_idx].mean().item()),
          'reward_best_argmax_success_rate': (
            final_successes[row_idx, reward_best_idx].mean().item()),
          'weight_target_probability_pearson': (
            correlations.mean().item()),
        }

      output_batches.append(
        _flatten_all_particles(xt, batch_size, num_particles).detach().cpu())
      batch_result = {
        'batch': batch_idx,
        'final_mean_ess': _effective_sample_size(log_weights).mean().item(),
        'steps': batch_stats,
      }
      if final_pool_diagnostics is not None:
        batch_result['final_pool_diagnostics'] = final_pool_diagnostics
      all_stats.append(batch_result)
  finally:
    model._restore_non_ema_params()

  return torch.cat(output_batches, dim=0), all_stats


@torch.no_grad()
def estimate_prior_value(
  model: diffusion.Diffusion,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_particles_total: int,
  num_rollout_samples: int,
  eps: float,
) -> float:
  """V_hat_T: value estimate at the untouched prior x_T (the t=T boundary
  used in Algorithm 6's base case, distinct from V_hat[t] for t=0..T-1,
  which are evaluated *after* a reverse step has been taken)."""
  xt = model._sample_prior(
    num_particles_total, model.config.model.length).to(model.device)
  t = torch.ones(xt.shape[0], 1, device=model.device) * (1 - eps)
  if model.T > 0:
    t = (t * model.T).to(torch.int) / model.T + (1 / model.T)
  sigma_t, _ = model.noise(t)
  if sigma_t.ndim > 1:
    sigma_t = sigma_t.squeeze(-1)
  value = _estimate_value(
    model=model, xt=xt, time_conditioning=sigma_t,
    classifier=classifier, target_class=target_class, alpha=alpha,
    num_rollout_samples=num_rollout_samples)
  return value.mean().item()


@torch.no_grad()
def estimate_prior_value_multistep(
  model: diffusion.Diffusion,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_samples: int,
  total_steps: int,
  eps: float,
  num_rollout_samples: int = 1,
) -> float:
  """V_hat_T via genuine multi-step ancestral rollout from the prior all the
  way to x0 (ordinary unconditional generation), instead of a single
  one-shot forward pass. Thin wrapper around `_estimate_value_multistep`
  with step_idx=-1 (all `total_steps` steps remain).

  num_rollout_samples: J, matched to the J used for v_hat_by_t (Table 2)
    rather than hard-coded to 1, so this Z=E_{x0~p0}[R(x0)] estimate has
    comparable variance to the guided v_hat_by_t entries it is compared
    against in the DP (a large J mismatch made this boundary estimate
    disproportionately noisy relative to v_hat_by_t)."""
  xt = model._sample_prior(
    num_samples, model.config.model.length).to(model.device)
  value = _estimate_value_multistep(
    model=model, xt=xt, step_idx=-1, total_steps=total_steps, eps=eps,
    classifier=classifier, target_class=target_class, alpha=alpha,
    num_rollout_samples=num_rollout_samples)
  return value.mean().item()


def run_warmup(
  model: diffusion.Diffusion,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_particles: int,
  num_rollout_samples: int,
  ess_threshold: float,
  partial_resample: int,
  total_steps: int,
  num_warmup_runs: int,
  eps: float,
  multistep_prior: bool = True,
  value_estimation: str = 'one-shot',
  proposal_method: str = 'base',
  grad_scale: float = 1.0,
  final_selection: str = 'sample',
  value_condition_mode: str = 'auto',
  gumbel_tau: float = 1.0,
  grad_batch_size: typing.Optional[int] = 20,
) -> typing.Tuple[
    typing.List[float], float, torch.Tensor, typing.List[typing.Any]]:
  """Runs M=num_warmup_runs full-guidance (every step guided) generations
  and returns (v_hat_by_t, v_hat_prior, warmup_samples, warmup_stats) per
  Eq. (11):
    v_hat_by_t[t] for t=0..total_steps-1 (paper's t-index, t=0 is the final
    /clean step), and v_hat_prior = V_hat_T at the t=T boundary.

  `warmup_samples` (every one of the N final particles from each of the M
  warmup runs, flattened to (num_warmup_runs * num_particles, seq_len) --
  see `sample()`) and `warmup_stats` (its per-run stats) are returned so
  callers *may* retain them, but they are dense (every-step-guided)
  generations, not samples from any particular sparse schedule's actual
  guided-step set -- `run_schedule_suite` uses them only to estimate
  v_hat_by_t/v_hat_prior for the DP schedule search and does not mix them
  into any schedule's eval pool.

  final_selection: unused -- `sample()` now always returns the full
    particle pool. Kept for call-site/CLI compatibility only.

  multistep_prior: if True (default), V_hat_T is estimated via genuine
    multi-step ancestral rollout (ordinary unconditional generation) rather
    than the single-forward-pass estimate used for v_hat_by_t. See
    `estimate_prior_value_multistep` docstring for why: the one-shot
    estimate at the fully-masked prior was found to be systematically
    biased high relative to one-shot estimates at partially-revealed
    states, which skewed the DP schedule search.
  """
  # `sample()` evaluates all timestep values under EMA parameters. The prior
  # boundary value must use the same model; mixing raw checkpoint parameters
  # for V_T with EMA parameters for V_0..V_{T-1} distorts the DP objective.
  model.load_ema_params()
  try:
    if multistep_prior:
      v_hat_prior = estimate_prior_value_multistep(
        model=model, classifier=classifier, target_class=target_class,
        alpha=alpha, num_samples=num_warmup_runs * num_particles,
        total_steps=total_steps, eps=eps,
        num_rollout_samples=num_rollout_samples)
    else:
      v_hat_prior = estimate_prior_value(
        model=model, classifier=classifier, target_class=target_class,
        alpha=alpha, num_particles_total=num_warmup_runs * num_particles,
        num_rollout_samples=num_rollout_samples, eps=eps)
  finally:
    model._restore_non_ema_params()

  warmup_samples, warmup_stats = sample(
    model=model, classifier=classifier, target_class=target_class,
    alpha=alpha, num_particles=num_particles,
    num_rollout_samples=num_rollout_samples, ess_threshold=ess_threshold,
    partial_resample=partial_resample, guidance_steps=total_steps,
    guidance_schedule='all', final_selection=final_selection,
    total_steps=total_steps, batch_size=num_warmup_runs, num_batches=1,
    eps=eps, value_estimation=value_estimation,
    proposal_method=proposal_method, grad_scale=grad_scale,
    value_condition_mode=value_condition_mode, gumbel_tau=gumbel_tau,
    grad_batch_size=grad_batch_size)

  mean_value_by_step = [None] * total_steps
  for step_stat in warmup_stats[0]['steps']:
    mean_value_by_step[step_stat['step']] = step_stat['mean_value']

  # step_idx = total_steps - 1 - t  <=>  t = total_steps - 1 - step_idx
  v_hat_by_t = [None] * total_steps
  for step_idx, value in enumerate(mean_value_by_step):
    v_hat_by_t[total_steps - 1 - step_idx] = value

  return v_hat_by_t, v_hat_prior, warmup_samples, warmup_stats


def solve_vista_schedule(
  v_hat_by_t: typing.List[float],
  v_hat_prior: float,
  budget: int,
  max_gap: typing.Optional[int] = None,
) -> typing.List[int]:
  """Algorithm 6 (DP schedule search, Appendix A.5), matching the paper's
  pseudocode exactly, including line 16:
    t_{T'} <- argmax_s ( dp[s][T'] + s * V_hat[s] )
  (the DP is free to leave the very last step t=0 unguided when that is
  optimal for the surrogate objective; the paper's Algorithm 6 does not
  hard-code `0 in G`).

  v_hat_by_t: V_hat[t] for t=0..T-1 (paper's t-index; t=0 is the final/
    clean step, t=T-1 is the step right after the prior).
  v_hat_prior: V_hat_T, the value at the untouched prior (t=T boundary).
  budget: T' = number of guided steps to select.
  max_gap: optional, off (None) by default, and not part of Algorithm 6 --
    a repo-specific robustness heuristic, not a paper mechanism. If set,
    forbids any unguided run (including the two runs at the t=0 and t=T
    boundaries) longer than `max_gap` steps. `v_hat_by_t` is
    estimated from a warmup trajectory that guides *every* step, so its
    value only reflects the effect of many consecutive guided steps; the
    unconstrained DP was empirically found to exploit this by concentrating
    almost the whole budget into one region and leaving one 700+-step
    unguided run elsewhere in the chain, which experiments
    (`eval_runs/vista_resnet_schedule/experiment_report.md`, "Root-cause
    investigation") showed performs worse than `uniform` at every budget
    tested. Capping the gap keeps the value-curve-informed step placement
    while removing the single-large-gap degenerate solutions.

  Returns a sorted list of paper-t indices, i.e. the guided-step set G.
  """
  total_steps = len(v_hat_by_t)
  if budget <= 0:
    return []
  if budget >= total_steps:
    return list(range(total_steps))
  if max_gap is not None and max_gap < 0:
    raise ValueError(f'max_gap must be non-negative, got {max_gap}.')

  neg_inf = float('-inf')
  # dp[i][s], parent[i][s] for i in 1..budget, s in 0..total_steps-1.
  dp = [[neg_inf] * total_steps for _ in range(budget + 1)]
  parent = [[None] * total_steps for _ in range(budget + 1)]

  for s in range(total_steps):
    gap = total_steps - 1 - s
    if max_gap is None or gap <= max_gap:
      dp[1][s] = gap * v_hat_prior

  for i in range(2, budget + 1):
    for s in range(0, total_steps - i + 1):
      best_c, best_sp = neg_inf, None
      sp_limit = total_steps - i + 2
      if max_gap is not None:
        sp_limit = min(sp_limit, s + 1 + max_gap + 1)
      for sp in range(s + 1, sp_limit):
        if dp[i - 1][sp] == neg_inf:
          continue
        c = dp[i - 1][sp] + (sp - s - 1) * v_hat_by_t[sp]
        if c > best_c:
          best_c, best_sp = c, sp
      dp[i][s] = best_c
      parent[i][s] = best_sp

  best_val, t_last = neg_inf, None
  for s in range(0, total_steps - budget + 1):
    if dp[budget][s] == neg_inf:
      continue
    gap = s
    if max_gap is not None and gap > max_gap:
      continue
    val = dp[budget][s] + s * v_hat_by_t[s]
    if val > best_val:
      best_val, t_last = val, s

  if t_last is None:
    raise ValueError(
      f'No feasible schedule for budget={budget}, max_gap={max_gap} over '
      f'total_steps={total_steps}. max_gap must be at least roughly '
      f'total_steps / budget.')

  schedule = [None] * (budget + 1)  # 1-indexed: schedule[1..budget]
  schedule[budget] = t_last
  for j in range(budget - 1, 0, -1):
    schedule[j] = parent[j + 1][schedule[j + 1]]

  return sorted(schedule[1:])


def empirical_schedule_objective(
  v_hat_by_t: typing.Sequence[float],
  v_hat_prior: float,
  guided_t: typing.Iterable[int],
) -> float:
  r"""Compute Eq. (12): S_hat(G) = sum_{t in T\G} V_hat_{ceil(t)}.

  ``guided_t`` uses the paper's reverse-time convention (t=0 is the clean
  endpoint).  If there is no guided step at or above ``t``, ``ceil(t)=T``
  and the prior estimate ``v_hat_prior`` is used.
  """
  total_steps = len(v_hat_by_t)
  guided = sorted(set(int(t) for t in guided_t))
  if any(t < 0 or t >= total_steps for t in guided):
    raise ValueError(
      f'guided_t must be inside [0, {total_steps}); got {guided}.')

  objective = 0.0
  previous_t = -1
  for active_t in guided:
    # The unguided steps after the previous active step and before this one
    # all have ceil(t)=active_t. The active step itself is excluded.
    objective += (active_t - previous_t - 1) * float(
      v_hat_by_t[active_t])
    previous_t = active_t
  # Steps above the final active step have ceil(t)=T.
  objective += (total_steps - previous_t - 1) * float(v_hat_prior)
  return objective


def search_schedule(
  model: diffusion.Diffusion,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_particles: int,
  num_rollout_samples: int,
  ess_threshold: float,
  partial_resample: int,
  total_steps: int,
  budget: int,
  num_warmup_runs: int,
  eps: float,
  multistep_prior: bool = True,
  value_estimation: str = 'one-shot',
  proposal_method: str = 'base',
  grad_scale: float = 1.0,
):
  """Runs the warmup + DP schedule search and returns the selected schedule
  as a set of `step_idx` values (code convention, step_idx=0 chronologically
  first), ready to pass as `explicit_guidance_steps` to `sample()`."""
  v_hat_by_t, v_hat_prior, warmup_samples, warmup_stats = run_warmup(
    model=model, classifier=classifier, target_class=target_class,
    alpha=alpha, num_particles=num_particles,
    num_rollout_samples=num_rollout_samples, ess_threshold=ess_threshold,
    partial_resample=partial_resample, total_steps=total_steps,
    num_warmup_runs=num_warmup_runs, eps=eps, multistep_prior=multistep_prior,
    value_estimation=value_estimation,
    proposal_method=proposal_method, grad_scale=grad_scale)

  guided_t = solve_vista_schedule(v_hat_by_t, v_hat_prior, budget)
  guided_step_idx = {total_steps - 1 - t for t in guided_t}
  return guided_step_idx, {
    'v_hat_by_t': v_hat_by_t,
    'v_hat_prior': v_hat_prior,
    'guided_t': guided_t,
    'warmup_samples': warmup_samples,
    'warmup_stats': warmup_stats,
  }


def solve_local_gain_schedule(
  v_hat_by_t: typing.List[float],
  v_hat_prior: float,
  budget: int,
) -> typing.List[int]:
  """Local reward-gain schedule: greedily picks the `budget` timesteps whose
  per-step value increment V_t(x_t) - V_{t+1}(x_{t+1}) is largest, i.e. the
  steps where guided denoising did the most work during warmup.

  v_hat_by_t: V_hat[t] for t=0..T-1 (paper's t-index; t=0 is the final/
    clean step, t=T-1 is the step right after the prior).
  v_hat_prior: V_hat_T, the value at the untouched prior (t=T boundary),
    used as V_{t+1} for the t=T-1 step (the step taken right off the prior).
  budget: T' = number of guided steps to select.

  Returns a sorted list of paper-t indices, i.e. the guided-step set G.
  """
  total_steps = len(v_hat_by_t)
  if budget <= 0:
    return []
  if budget >= total_steps:
    return list(range(total_steps))

  gain = [0.0] * total_steps
  for t in range(total_steps - 1):
    gain[t] = v_hat_by_t[t] - v_hat_by_t[t + 1]
  gain[total_steps - 1] = v_hat_by_t[total_steps - 1] - v_hat_prior

  top_t = sorted(range(total_steps), key=lambda t: gain[t], reverse=True)
  return sorted(top_t[:budget])


def solve_topk_value_schedule(
  v_hat_by_t: typing.List[float],
  budget: int,
) -> typing.List[int]:
  """Top-k value schedule: picks the `budget` timesteps with the largest
  value V_t(x_t) itself (as opposed to the value increment), i.e. the states
  warmup found to already be most rewarding.

  v_hat_by_t: V_hat[t] for t=0..T-1 (paper's t-index; t=0 is the final/
    clean step, t=T-1 is the step right after the prior).
  budget: T' = number of guided steps to select.

  Returns a sorted list of paper-t indices, i.e. the guided-step set G.
  """
  total_steps = len(v_hat_by_t)
  if budget <= 0:
    return []
  if budget >= total_steps:
    return list(range(total_steps))

  top_t = sorted(
    range(total_steps), key=lambda t: v_hat_by_t[t], reverse=True)
  return sorted(top_t[:budget])


def search_local_gain_schedule(
  model: diffusion.Diffusion,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_particles: int,
  num_rollout_samples: int,
  ess_threshold: float,
  partial_resample: int,
  total_steps: int,
  budget: int,
  num_warmup_runs: int,
  eps: float,
  multistep_prior: bool = True,
  value_estimation: str = 'one-shot',
  proposal_method: str = 'base',
  grad_scale: float = 1.0,
):
  """Runs the warmup + local-gain schedule search and returns the selected
  schedule as a set of `step_idx` values (code convention, step_idx=0
  chronologically first), ready to pass as `explicit_guidance_steps` to
  `sample()`."""
  v_hat_by_t, v_hat_prior, warmup_samples, warmup_stats = run_warmup(
    model=model, classifier=classifier, target_class=target_class,
    alpha=alpha, num_particles=num_particles,
    num_rollout_samples=num_rollout_samples, ess_threshold=ess_threshold,
    partial_resample=partial_resample, total_steps=total_steps,
    num_warmup_runs=num_warmup_runs, eps=eps, multistep_prior=multistep_prior,
    value_estimation=value_estimation,
    proposal_method=proposal_method, grad_scale=grad_scale)

  guided_t = solve_local_gain_schedule(v_hat_by_t, v_hat_prior, budget)
  guided_step_idx = {total_steps - 1 - t for t in guided_t}
  return guided_step_idx, {
    'v_hat_by_t': v_hat_by_t,
    'v_hat_prior': v_hat_prior,
    'guided_t': guided_t,
    'warmup_samples': warmup_samples,
    'warmup_stats': warmup_stats,
  }


def search_topk_schedule(
  model: diffusion.Diffusion,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_particles: int,
  num_rollout_samples: int,
  ess_threshold: float,
  partial_resample: int,
  total_steps: int,
  budget: int,
  num_warmup_runs: int,
  eps: float,
  multistep_prior: bool = True,
  value_estimation: str = 'one-shot',
  proposal_method: str = 'base',
  grad_scale: float = 1.0,
):
  """Runs the warmup + top-k value schedule search and returns the selected
  schedule as a set of `step_idx` values (code convention, step_idx=0
  chronologically first), ready to pass as `explicit_guidance_steps` to
  `sample()`."""
  v_hat_by_t, v_hat_prior, warmup_samples, warmup_stats = run_warmup(
    model=model, classifier=classifier, target_class=target_class,
    alpha=alpha, num_particles=num_particles,
    num_rollout_samples=num_rollout_samples, ess_threshold=ess_threshold,
    partial_resample=partial_resample, total_steps=total_steps,
    num_warmup_runs=num_warmup_runs, eps=eps, multistep_prior=multistep_prior,
    value_estimation=value_estimation,
    proposal_method=proposal_method, grad_scale=grad_scale)

  guided_t = solve_topk_value_schedule(v_hat_by_t, budget)
  guided_step_idx = {total_steps - 1 - t for t in guided_t}
  return guided_step_idx, {
    'v_hat_by_t': v_hat_by_t,
    'v_hat_prior': v_hat_prior,
    'guided_t': guided_t,
    'warmup_samples': warmup_samples,
    'warmup_stats': warmup_stats,
  }


SCHEDULE_NAMES = (
  'full', 'first', 'last', 'middle', 'uniform', 'vista', 'vista_localgain',
  'vista_topk', 'top_v', 'top_dv', 'interval1', 'interval2', 'interval3',
  'interval4', 'interval5')


def run_schedule_suite(
  model: diffusion.Diffusion,
  classifier: torch.nn.Module,
  target_class: int,
  alpha: float,
  num_particles: int,
  num_rollout_samples: int,
  ess_threshold: float,
  partial_resample: int,
  budget: int,
  final_selection: str,
  total_steps: int,
  batch_size: int,
  num_batches: int,
  eps: float,
  num_warmup_runs: int,
  schedules: typing.Sequence[str],
  output_prefix: str,
  device: torch.device,
  reward_threshold: float,
  lpips_model: torch.nn.Module,
  lpips_threshold: float,
  value_estimation: str = 'one-shot',
  proposal_method: str = 'base',
  grad_scale: float = 1.0,
  record_final_pool_diagnostics: bool = False,
  value_condition_mode: str = 'auto',
  gumbel_tau: float = 1.0,
  grad_batch_size: typing.Optional[int] = 20,
  evaluation_seed: typing.Optional[int] = None,
  precomputed_warmup: typing.Optional[
    typing.Tuple[typing.List[float], float]] = None,
):
  """Runs `sample()` once per entry in `schedules` (Table 3-style baselines
  'full'/'first'/'last'/'middle'/'uniform' plus the VISTA-SMC-optimized
  'vista' schedule) at the same sample budget, decodes+scores every run
  against the reward classifier, and returns a dict of per-schedule metrics
  mirroring the paper's Table 3/4/5 columns (Time, Reward, success rate)."""
  results = {}
  warmup_schedule_meta = {}
  schedule_solvers = {
    'vista': lambda values, prior: solve_vista_schedule(
      values, prior, budget),
    'vista_localgain': lambda values, prior: solve_local_gain_schedule(
      values, prior, budget),
    'vista_topk': lambda values, _prior: solve_topk_value_schedule(
      values, budget),
    'top_v': lambda values, _prior: solve_topk_value_schedule(
      values, budget),
    'top_dv': lambda values, prior: solve_local_gain_schedule(
      values, prior, budget),
  }
  requested_vista_schedules = [
    schedule for schedule in schedules if schedule in schedule_solvers]
  shared_warmup = None
  shared_warmup_seconds = 0.0
  if requested_vista_schedules:
    if precomputed_warmup is not None:
      values, prior = precomputed_warmup
      if len(values) != total_steps:
        raise ValueError(
          f'Precomputed warmup has {len(values)} timesteps; expected '
          f'{total_steps}.')
      shared_warmup = (
        values, prior,
        torch.empty((0, 1), dtype=torch.long, device=device),
        [])
    else:
      t0 = time.time()
      shared_warmup = run_warmup(
        model=model, classifier=classifier, target_class=target_class,
        alpha=alpha, num_particles=num_particles,
        num_rollout_samples=num_rollout_samples,
        ess_threshold=ess_threshold,
        partial_resample=partial_resample, total_steps=total_steps,
        num_warmup_runs=num_warmup_runs, eps=eps,
        multistep_prior=(value_estimation != 'one-shot'),
        value_estimation=value_estimation,
        proposal_method=proposal_method, grad_scale=grad_scale,
        final_selection=final_selection,
        value_condition_mode=value_condition_mode, gumbel_tau=gumbel_tau,
        grad_batch_size=grad_batch_size)
      shared_warmup_seconds = time.time() - t0

  total_requested = batch_size * num_batches
  for schedule in schedules:
    warmup_seconds = 0.0
    reused_warmup_samples = None
    if schedule in schedule_solvers:
      v_hat_by_t, v_hat_prior, warmup_samples, warmup_stats = shared_warmup
      guided_t = schedule_solvers[schedule](v_hat_by_t, v_hat_prior)
      vista_step_idx = {total_steps - 1 - t for t in guided_t}
      vista_meta = {
        'v_hat_by_t': v_hat_by_t,
        'v_hat_prior': v_hat_prior,
        'guided_t': guided_t,
      }
      # Report the cost this VISTA variant would pay when run alone. The
      # suite executes it once and shares the result among variants.
      warmup_seconds = shared_warmup_seconds
      warmup_schedule_meta[schedule] = vista_meta
      explicit_guidance_steps = vista_step_idx
      guided_step_indices = sorted(vista_step_idx)
      guidance_steps, guidance_schedule = budget, 'uniform'
      # The M full-guidance warmup generations are used only to estimate
      # v_hat_by_t/v_hat_prior for the DP schedule search -- they are dense
      # (every-step-guided) runs, not samples from this schedule's actual
      # sparse guided-step set, so they are never mixed into the eval pool.
      # Every one of `total_requested` samples is generated fresh under the
      # schedule's real (sparse) guidance below. (A dense-guidance reference
      # row, if wanted, should be reported separately via schedule='full'.)
      num_reused = 0
      reused_warmup_samples = warmup_samples[:0]
      num_remaining = total_requested
    elif schedule == 'full':
      explicit_guidance_steps = None
      guided_step_indices = list(range(total_steps))
      guidance_steps, guidance_schedule = total_steps, 'all'
      num_remaining = total_requested
    else:
      explicit_guidance_steps = None
      guided_step_indices = _guidance_step_indices(
        total_steps, budget, schedule)
      guidance_steps, guidance_schedule = budget, schedule
      num_remaining = total_requested

    guided_t_for_objective = sorted(
      total_steps - 1 - step_idx for step_idx in guided_step_indices)
    if len(guided_t_for_objective) == total_steps:
      # T\G is empty for full guidance, so Eq. (12) is exactly zero and
      # does not require a warm-up value curve.
      s_hat_g = 0.0
    elif shared_warmup is None:
      s_hat_g = None
    else:
      s_hat_g = empirical_schedule_objective(
        shared_warmup[0], shared_warmup[1], guided_t_for_objective)

    if evaluation_seed is not None:
      L.seed_everything(evaluation_seed)
    t0 = time.time()
    if num_remaining > 0:
      # Chunk at the caller's original `batch_size` (preserves memory
      # behavior), then trim the possibly-overshot last chunk down to
      # exactly `num_remaining` samples.
      remaining_num_batches = -(-num_remaining // batch_size)
      new_samples, new_stats = sample(
        model=model, classifier=classifier, target_class=target_class,
        alpha=alpha, num_particles=num_particles,
        num_rollout_samples=num_rollout_samples, ess_threshold=ess_threshold,
        partial_resample=partial_resample, guidance_steps=guidance_steps,
        guidance_schedule=guidance_schedule, final_selection=final_selection,
        total_steps=total_steps, batch_size=batch_size,
        num_batches=remaining_num_batches, eps=eps,
        explicit_guidance_steps=explicit_guidance_steps,
        value_estimation=value_estimation,
        proposal_method=proposal_method, grad_scale=grad_scale,
        record_final_pool_diagnostics=record_final_pool_diagnostics,
        value_condition_mode=value_condition_mode, gumbel_tau=gumbel_tau,
        grad_batch_size=grad_batch_size)
      new_samples = new_samples[:num_remaining * num_particles]
    else:
      new_samples, new_stats = reused_warmup_samples[:0], []
    generation_seconds = time.time() - t0

    if reused_warmup_samples is not None and reused_warmup_samples.shape[0]:
      samples = torch.cat([reused_warmup_samples, new_samples], dim=0)
      stats = warmup_stats + new_stats
    else:
      samples, stats = new_samples, new_stats

    images, reward, success_rate = _score_samples(
      classifier=classifier, tokenizer=model.tokenizer, x0_tokens=samples,
      target_class=target_class, alpha=alpha, device=device)
    unique_metric = _unique_success_count(
      images=images, reward=reward, reward_threshold=reward_threshold,
      lpips_model=lpips_model, lpips_threshold=lpips_threshold)

    torch.save({
      'samples': samples,
      'stats': stats,
      'schedule': schedule,
    }, f'{output_prefix}_{schedule}.pt')
    preview_path = f'{output_prefix}_{schedule}.png'
    torchvision.utils.save_image(
      images.clamp(0, 1), preview_path,
      nrow=max(1, min(8, images.shape[0])))

    results[schedule] = {
      'warmup_seconds': warmup_seconds,
      # Wall time for all `generation_runs` SMC generations, and the
      # per-generation time the tables report.
      'generation_seconds': generation_seconds,
      'generation_runs': total_requested,
      'generation_seconds_per_run': generation_seconds / total_requested,
      'num_samples': int(samples.shape[0]),
      'mean_reward': reward.mean().item(),
      'reward_std': reward.std().item(),
      # reward = p(target|x) ** (1 / alpha).  Keep the untempered
      # probability too so runs with different alpha remain comparable.
      'mean_target_probability': reward.pow(alpha).mean().item(),
      'target_probability_std': reward.pow(alpha).std().item(),
      'success_rate': success_rate,
      'guided_step_indices': guided_step_indices,
      's_hat_g': s_hat_g,
      'unique_success_count': unique_metric,
    }
    print(f'[{schedule}] warmup={warmup_seconds:.2f}s '
          f'gen={generation_seconds / total_requested:.2f}s/run '
          f'({generation_seconds:.2f}s for {total_requested} runs) '
          f'reward={results[schedule]["mean_reward"]:.4f} '
          f'success_rate={results[schedule]["success_rate"]:.4f} '
          f'unique_successes={unique_metric["num_unique_successes"]}'
          f'/{unique_metric["num_passed_reward_threshold"]}')

  for schedule, vista_meta in warmup_schedule_meta.items():
    key = 'vista_schedule_meta' if schedule == 'vista' else f'{schedule}_meta'
    results[key] = {
      'v_hat_by_t': vista_meta['v_hat_by_t'],
      'v_hat_prior': vista_meta['v_hat_prior'],
      'guided_t': vista_meta['guided_t'],
    }
  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--config', default='outputs/cifar10/mdlm/.hydra/config.yaml')
  parser.add_argument(
    '--checkpoint', default='outputs/cifar10/mdlm/checkpoints/last.ckpt')
  parser.add_argument(
    '--classifier-checkpoint',
    default='outputs/cifar10/reward_classifier/best.ckpt')
  parser.add_argument('--tag', default='mdlm',
                       help='Short label (e.g. mdlm/udlm) used as a prefix '
                            'for output files, so runs for different '
                            'models do not overwrite each other.')
  parser.add_argument('--target-class', type=int, default=0,
                       help='0=airplane,1=automobile,...,9=truck')
  parser.add_argument('--alpha', type=float, default=1.0)
  parser.add_argument('--num-particles', type=int, default=20)
  parser.add_argument('--num-rollout-samples', type=int, default=1)
  parser.add_argument('--ess-threshold', type=float, default=0.95,
                       help='Resample when ESS falls below this fraction '
                            'of num_particles.')
  parser.add_argument('--partial-resample', type=int, default=None,
                       help='Particles replaced per resample event '
                            '(default: floor(num_particles/2))')
  parser.add_argument('--guidance-steps', type=int, default=25,
                       help="Guidance budget T' shared by every baseline "
                            'schedule and by the VISTA-SMC DP search.')
  parser.add_argument(
    '--schedules', default=','.join(SCHEDULE_NAMES),
    help='Comma-separated list of schedules to evaluate and report reward '
         f'for. Choices: {SCHEDULE_NAMES}.')
  parser.add_argument('--num-warmup-runs', type=int, default=3,
                       help='M, number of full-guidance warmup runs used '
                            "by the 'vista' schedule's DP search.")
  parser.add_argument(
    '--reuse-warmup-comparison', default=None,
    help='Completed comparison JSON whose VISTA v_hat_by_t and v_hat_prior '
         'are reused instead of running a new full-guidance warmup. The '
         'source seed, M, J, N, ESS, resampling size, and total steps must '
         'match this invocation; the guidance budget may differ.')
  parser.add_argument('--final-selection', default='sample',
                       choices=[
                         'sample', 'best', 'reward-best', 'margin-best'],
                       help='Unused: every one of the N final particles is '
                            'now kept (see --total-samples). Kept for '
                            'config/log compatibility only.')
  parser.add_argument(
    '--record-final-pool-diagnostics', action='store_true',
    help='Record classifier quality of all final particles, weighted quality, '
         'oracle/reward-best success, and the Pearson correlation between '
         'SMC weights and target probability. Off by default.')
  parser.add_argument('--steps', type=int, default=128)
  parser.add_argument('--total-samples', type=int, default=100,
                       help='Number of independent SMC generations per '
                            'schedule. Must be divisible by --batch-size. '
                            'Every one of the N final particles from each '
                            'generation is kept, so the actual output row '
                            'count per schedule is --total-samples x '
                            '--num-particles.')
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--eps', type=float, default=1e-5)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument(
    '--output-dir', default='eval_runs')
  parser.add_argument(
    '--reward-threshold', type=float, default=0.5,
    help='Reward value (= p(target_class|x0) when alpha=1.0) a sample must '
         "exceed to count toward the unique-success-count metric. Default "
         "0.5 mirrors the argmax success criterion at alpha=1.0.")
  parser.add_argument(
    '--lpips-threshold', type=float, default=0.03,
    help='LPIPS (AlexNet-backbone) perceptual distance between two '
         'samples at or below which they are treated as the same sample '
         'for the unique-success-count metric. Empirically, pairwise '
         'LPIPS distances among generated CIFAR-10 samples here span '
         '~0.004-0.2 (median ~0.04); 0.03 sits in the lower quartile so '
         'only distinctly closer-than-typical pairs count as duplicates. '
         'Tune per dataset/resolution.')
  parser.add_argument(
    '--value-estimation', default='one-shot',
    choices=['multistep', 'one-shot'],
    help="'one-shot' (default): V_t(xt) via a single model.forward pass "
         "under the per-token-independence assumption (see "
         "_estimate_value), matching Eq. (5)'s V_t(xt) := "
         "E_{x0~p_{0|t}(.|xt)}[R(x0)] and Table 2's stated cost model; also "
         "switches the VISTA warmup's prior-value estimate (V_hat_T) to its "
         "one-shot form. 'multistep': V_t(xt) via a genuine ancestral "
         "rollout to x0 (see _estimate_value_multistep) -- a repo-specific "
         "research alternative that estimates a different quantity than "
         "Eq. (5) and costs more than Table 2 assumes; opt-in only.")
  parser.add_argument(
    '--value-condition-mode', default='auto', choices=['auto', 'none'],
    help="'auto' uses the CFG masked-class embedding for class/CFG-trained "
         "checkpoints and None otherwise. 'none' reproduces the legacy "
         "one-shot OOD ablation. This only affects one-shot value estimates.")
  parser.add_argument(
    '--proposal-method', default='base', choices=['base', 'grad'],
    help="SMC proposal family (Section 3.3). 'base': q=p, the paper's "
         "SMC-base. 'grad': the paper's SMC-grad, a value-gradient-twisted "
         'proposal with straight-through Gumbel-softmax value gradients.')
  parser.add_argument(
    '--grad-scale', type=float, default=1.0,
    help="Extra tunable multiplier on the SMC-grad twist (on top of the "
         "1/alpha the paper's formula already prescribes), analogous to "
         "the `gamma` knob on CBG/CFG guidance. No effect unless "
         "--proposal-method=grad.")
  parser.add_argument(
    '--gumbel-tau', type=float, default=1.0,
    help='Straight-through Gumbel-softmax temperature for SMC-grad '
         '(reference default: 1.0).')
  parser.add_argument(
    '--grad-batch-size', type=int, default=20,
    help='Maximum particle batch per SMC-grad backward pass. Use 0 for the '
         'whole particle batch at once.')
  args = parser.parse_args()

  if args.partial_resample is None:
    args.partial_resample = max(1, args.num_particles // 2)
  if args.total_samples % args.batch_size != 0:
    raise ValueError(
      f'--total-samples ({args.total_samples}) must be divisible by '
      f'--batch-size ({args.batch_size}).')
  num_batches = args.total_samples // args.batch_size
  schedules = [s.strip() for s in args.schedules.split(',') if s.strip()]
  for schedule in schedules:
    if schedule not in SCHEDULE_NAMES:
      raise ValueError(
        f'Unknown schedule {schedule!r}. Choices: {SCHEDULE_NAMES}.')

  precomputed_warmup = None
  if args.reuse_warmup_comparison is not None:
    with open(args.reuse_warmup_comparison, encoding='utf-8') as handle:
      warmup_source = json.load(handle)
    source_metadata = warmup_source['metadata']
    required_matches = {
      'seed': args.seed,
      'num_warmup_runs': args.num_warmup_runs,
      'num_rollout_samples': args.num_rollout_samples,
      'num_particles': args.num_particles,
      'ess_threshold': args.ess_threshold,
      'partial_resample': args.partial_resample,
      'steps': args.steps,
      'value_estimation': args.value_estimation,
      'value_condition_mode': args.value_condition_mode,
      'proposal_method': args.proposal_method,
    }
    mismatches = {
      key: (source_metadata.get(key), expected)
      for key, expected in required_matches.items()
      if source_metadata.get(key) != expected
    }
    if mismatches:
      raise ValueError(
        f'Warmup source metadata mismatch: {mismatches}.')
    source_schedule = warmup_source['results']['vista_schedule_meta']
    precomputed_warmup = (
      source_schedule['v_hat_by_t'], source_schedule['v_hat_prior'])

  L.seed_everything(args.seed)
  config = _load_config(args.config)
  config.eval.checkpoint_path = args.checkpoint
  config.guidance = None

  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion.load_from_checkpoint(
    args.checkpoint, tokenizer=tokenizer, config=config,
    logger=False).to(args.device)
  model.eval()

  classifier, classes = load_reward_classifier(
    args.classifier_checkpoint, args.device)
  lpips_model = load_lpips_model(args.device)

  os.makedirs(args.output_dir, exist_ok=True)
  output_prefix = os.path.join(args.output_dir, f'vista_smc_{args.tag}')

  results = run_schedule_suite(
    model=model,
    classifier=classifier,
    target_class=args.target_class,
    alpha=args.alpha,
    num_particles=args.num_particles,
    num_rollout_samples=args.num_rollout_samples,
    ess_threshold=args.ess_threshold,
    partial_resample=args.partial_resample,
    budget=args.guidance_steps,
    final_selection=args.final_selection,
    total_steps=args.steps,
    batch_size=args.batch_size,
    num_batches=num_batches,
    eps=args.eps,
    num_warmup_runs=args.num_warmup_runs,
    schedules=schedules,
    output_prefix=output_prefix,
    device=args.device,
    reward_threshold=args.reward_threshold,
    lpips_model=lpips_model,
    lpips_threshold=args.lpips_threshold,
    value_estimation=args.value_estimation,
    proposal_method=args.proposal_method,
    grad_scale=args.grad_scale,
    record_final_pool_diagnostics=args.record_final_pool_diagnostics,
    value_condition_mode=args.value_condition_mode,
    gumbel_tau=args.gumbel_tau,
    grad_batch_size=args.grad_batch_size,
    evaluation_seed=args.seed,
    precomputed_warmup=precomputed_warmup)

  summary_path = f'{output_prefix}_comparison.json'
  with open(summary_path, 'w') as f:
    json.dump({
      'metadata': {
        'tag': args.tag,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'classifier_checkpoint': args.classifier_checkpoint,
        'classes': classes,
        'target_class': args.target_class,
        'target_class_name': classes[args.target_class],
        'alpha': args.alpha,
        'num_particles': args.num_particles,
        'num_rollout_samples': args.num_rollout_samples,
        'ess_threshold': args.ess_threshold,
        'partial_resample': args.partial_resample,
        'guidance_steps': args.guidance_steps,
        'num_warmup_runs': args.num_warmup_runs,
        'final_selection': args.final_selection,
        'record_final_pool_diagnostics': (
          args.record_final_pool_diagnostics),
        'steps': args.steps,
        'total_samples': args.total_samples,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'device': args.device,
        'schedules': schedules,
        'reward_threshold': args.reward_threshold,
        'lpips_threshold': args.lpips_threshold,
        'value_estimation': args.value_estimation,
        'value_condition_mode': args.value_condition_mode,
        'proposal_method': args.proposal_method,
        'grad_scale': args.grad_scale,
        'gumbel_tau': args.gumbel_tau,
        'grad_batch_size': args.grad_batch_size,
        'warmup_reused': precomputed_warmup is not None,
        'warmup_source_comparison': args.reuse_warmup_comparison,
      },
      'results': results,
    }, f, indent=2)

  print(f'\n=== {args.tag} summary (target={classes[args.target_class]}) ===')
  print('{:>4s}  {:<10s}  {:>18s}  {:>14s}  {:>14s}'.format(
    "T'", 'Policy', 'executed time [s]', 'unique success', 'S_hat(G)'))
  for schedule in schedules:
    r = results[schedule]
    u = r['unique_success_count']
    score = '–' if r['s_hat_g'] is None else f'{r["s_hat_g"]:.6g}'
    print(
      f'{args.guidance_steps:4d}  {schedule:<10s}  '
      f'{r["generation_seconds_per_run"]:18.2f}  '
      f'{u["num_unique_successes"]:14d}  {score:>14s}')
  print(f'Saved per-schedule samples/previews to {output_prefix}_<schedule>.*')
  print(f'Saved comparison summary to {summary_path}')


if __name__ == '__main__':
  main()
