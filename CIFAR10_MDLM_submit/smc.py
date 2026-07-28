"""Sequential Monte Carlo sampling for trained diffusion models."""

import json
import os
import typing

import torch
import torchvision
from tqdm import tqdm

import diffusion


def _cfg_value(config, name, default=None):
  if not hasattr(config, 'smc') or config.smc is None:
    return default
  value = config.smc.get(name, default)
  return default if value is None else value


def _systematic_resample(weights: torch.Tensor) -> torch.Tensor:
  batch_size, num_particles = weights.shape
  positions = (
    torch.rand(batch_size, 1, device=weights.device)
    + torch.arange(num_particles, device=weights.device)
  ) / num_particles
  cumulative = weights.cumsum(dim=1)
  cumulative[:, -1] = 1.0
  return torch.stack([
    torch.searchsorted(cumulative[i], positions[i])
    for i in range(batch_size)
  ])


def _effective_sample_size(log_weights: torch.Tensor) -> torch.Tensor:
  weights = log_weights.softmax(dim=1)
  return 1.0 / weights.square().sum(dim=1).clamp_min(1e-30)


def _guidance_step_indices(
  total_steps: int,
  guidance_steps: int,
  schedule: str,
) -> typing.List[int]:
  if total_steps <= 0:
    raise ValueError(f'total_steps must be positive, got {total_steps}.')
  if guidance_steps < 0:
    raise ValueError(
      f'guidance_steps must be non-negative, got {guidance_steps}.')
  if guidance_steps == 0:
    return []
  if guidance_steps >= total_steps or schedule == 'all':
    return list(range(total_steps))

  if schedule == 'first':
    return list(range(guidance_steps))
  if schedule == 'last':
    return list(range(total_steps - guidance_steps, total_steps))
  if schedule == 'middle':
    start = (total_steps - guidance_steps) // 2
    return list(range(start, start + guidance_steps))
  if schedule == 'uniform':
    if guidance_steps == 1:
      return [0]
    return sorted({
      round((total_steps - 1) * i / (guidance_steps - 1))
      for i in range(guidance_steps)
    })
  if schedule.startswith('interval') and schedule[len('interval'):].isdigit():
    # Generalizes first/middle/last to a 5-way split of the guided block's
    # start position: interval1 == first, interval3 == middle,
    # interval5 == last; interval2/4 are the quarter/three-quarter points.
    position = int(schedule[len('interval'):])
    if not 1 <= position <= 5:
      raise ValueError(
        f'Unknown smc.guidance_schedule={schedule}. interval position '
        'must be 1-5.')
    start = round((total_steps - guidance_steps) * (position - 1) / 4)
    return list(range(start, start + guidance_steps))
  raise ValueError(
    f"Unknown smc.guidance_schedule={schedule}. "
    "Use one of: all, first, last, middle, uniform, interval1-5.")


def _gather_transition_log_prob(
  probs: torch.Tensor,
  samples: torch.Tensor,
) -> torch.Tensor:
  token_probs = probs.gather(
    dim=-1,
    index=samples[..., None]).squeeze(-1)
  return token_probs.clamp_min(1e-30).log().sum(dim=-1)


def _resolve_output_path(config) -> str:
  output_path = config.eval.generated_samples_path
  if output_path:
    return output_path
  checkpoint_path = config.eval.checkpoint_path
  if checkpoint_path:
    checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    return os.path.join(checkpoint_dir, 'smc_samples')
  return os.path.join(config.checkpointing.save_dir, 'smc_samples')


def _build_cond(config, device: torch.device, batch_size: int) -> torch.Tensor:
  condition = _cfg_value(config, 'condition', None)
  if condition is None and getattr(config, 'guidance', None) is not None:
    condition = config.guidance.condition
  if condition is None:
    raise ValueError(
      'SMC target/proposal uses cfg, but no condition was provided. '
      'Set smc.condition or guidance.condition.')
  return (
    torch.ones(batch_size, device=device, dtype=torch.long)
    * int(condition)
  )


def _transition_probs(
  model: diffusion.Diffusion,
  config,
  xt: torch.Tensor,
  time_conditioning: torch.Tensor,
  move_chance_t: torch.Tensor,
  move_chance_s: torch.Tensor,
  role: str,
  cond: typing.Optional[torch.Tensor],
) -> torch.Tensor:
  method = _cfg_value(config, f'{role}_method', None)
  if method is None:
    method = _cfg_value(config, role, 'cfg')

  if method == 'base':
    return model._diffusion_transition_probs(
      xt=xt,
      time_conditioning=time_conditioning,
      move_chance_t=move_chance_t,
      move_chance_s=move_chance_s,
      method='base')

  if method != 'cfg':
    raise NotImplementedError(
      f"SMC {role} method {method} is not implemented.")

  if role == 'proposal':
    gamma = _cfg_value(config, 'proposal_gamma', 0.0)
  else:
    gamma = _cfg_value(config, 'target_gamma', None)
    if gamma is None and getattr(config, 'guidance', None) is not None:
      gamma = config.guidance.gamma
    gamma = 1.0 if gamma is None else gamma

  return model._diffusion_transition_probs(
    xt=xt,
    time_conditioning=time_conditioning,
    move_chance_t=move_chance_t,
    move_chance_s=move_chance_s,
    method='cfg',
    cond=cond,
    gamma=float(gamma))


def _maybe_resample(
  xt: torch.Tensor,
  log_weights: torch.Tensor,
  threshold: float,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  batch_size, num_particles = log_weights.shape
  ess = _effective_sample_size(log_weights)
  should_resample = ess < (threshold * num_particles)
  if not should_resample.any():
    return xt, log_weights, should_resample

  weights = log_weights.softmax(dim=1)
  ancestors = _systematic_resample(weights)
  grouped_xt = xt.view(batch_size, num_particles, xt.shape[-1])
  resampled_xt = grouped_xt.gather(
    dim=1,
    index=ancestors[..., None].expand(-1, -1, xt.shape[-1]))
  grouped_xt = torch.where(
    should_resample[:, None, None],
    resampled_xt,
    grouped_xt)
  log_weights = torch.where(
    should_resample[:, None],
    torch.zeros_like(log_weights),
    log_weights)
  return grouped_xt.reshape_as(xt), log_weights, should_resample


def _select_final_particles(
  xt: torch.Tensor,
  log_weights: torch.Tensor,
  final_selection: str,
) -> torch.Tensor:
  batch_size, num_particles = log_weights.shape
  grouped_xt = xt.view(batch_size, num_particles, xt.shape[-1])
  if final_selection == 'best':
    selected = log_weights.argmax(dim=1)
  elif final_selection == 'sample':
    selected = torch.multinomial(log_weights.softmax(dim=1), 1).squeeze(1)
  else:
    raise ValueError(
      f"Unknown smc.final_selection={final_selection}. "
      "Use 'sample' or 'best'.")
  return grouped_xt[
    torch.arange(batch_size, device=xt.device),
    selected]


def _save_samples(
  config,
  tokenizer,
  samples: torch.Tensor,
  stats: typing.Dict[str, typing.Any],
) -> None:
  output_path = _resolve_output_path(config)
  os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

  if config.is_vision:
    tensor_path = output_path if output_path.endswith('.pt') else f'{output_path}.pt'
    images = tokenizer.batch_decode(samples.cpu()).float() / 255.0
    torch.save({
      'samples': samples.cpu(),
      'images': images,
      'stats': stats,
    }, tensor_path)
    preview_path = os.path.splitext(tensor_path)[0] + '.png'
    torchvision.utils.save_image(
      images.clamp(0, 1),
      preview_path,
      nrow=max(1, min(8, images.shape[0])))
    print(f"Saved SMC tensor samples to {tensor_path}")
    print(f"Saved SMC preview grid to {preview_path}")
    return

  decoded_samples = tokenizer.batch_decode(samples)
  json_path = output_path if output_path.endswith('.json') else f'{output_path}.json'
  with open(json_path, 'w') as f:
    json.dump({
      'generated_seqs': decoded_samples,
      'stats': stats,
    }, f, indent=2)
  print(f"Saved SMC samples to {json_path}")


@torch.no_grad()
def sample(model: diffusion.Diffusion, config):
  num_particles = int(_cfg_value(config, 'num_particles', 16))
  ess_threshold = float(_cfg_value(config, 'ess_threshold', 0.5))
  final_selection = _cfg_value(config, 'final_selection', 'sample')
  eps = float(_cfg_value(config, 'eps', 1e-5))
  total_steps = int(config.sampling.steps)
  guidance_steps = int(_cfg_value(config, 'guidance_steps', total_steps))
  guidance_schedule = _cfg_value(config, 'guidance_schedule', 'all')
  guidance_step_indices = _guidance_step_indices(
    total_steps=total_steps,
    guidance_steps=guidance_steps,
    schedule=guidance_schedule)
  guidance_step_set = set(guidance_step_indices)

  output_batches = []
  all_stats = []

  if not config.eval.disable_ema:
    model.load_ema_params()
  try:
    for batch_idx in tqdm(
        range(config.sampling.num_sample_batches),
        desc='SMC batches',
        leave=False):
      batch_size = int(config.sampling.batch_size)
      particle_batch_size = batch_size * num_particles
      xt = model._sample_prior(
        particle_batch_size,
        config.model.length).to(model.device)
      log_weights = torch.zeros(
        batch_size, num_particles, device=model.device)

      cond = None
      if (_cfg_value(config, 'proposal', 'cfg') == 'cfg'
          or _cfg_value(config, 'proposal_method', None) == 'cfg'
          or _cfg_value(config, 'target', 'cfg') == 'cfg'
          or _cfg_value(config, 'target_method', None) == 'cfg'):
        cond = _build_cond(config, model.device, particle_batch_size)

      timesteps = torch.linspace(
        1, eps, total_steps + 1, device=model.device)
      dt = (1 - eps) / total_steps
      batch_stats = []

      for step_idx in tqdm(
          range(total_steps),
          desc='SMC sampling',
          leave=False):
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

        proposal_probs = _transition_probs(
          model=model,
          config=config,
          xt=xt,
          time_conditioning=sigma_t,
          move_chance_t=move_chance_t,
          move_chance_s=move_chance_s,
          role='proposal',
          cond=cond)
        xs = diffusion._sample_categorical(proposal_probs)
        if guidance_active:
          target_probs = _transition_probs(
            model=model,
            config=config,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            role='target',
            cond=cond)
        else:
          target_probs = proposal_probs

        log_increment = (
          _gather_transition_log_prob(target_probs, xs)
          - _gather_transition_log_prob(proposal_probs, xs)
        ).view(batch_size, num_particles)
        log_weights = log_weights + log_increment
        log_weights = log_weights - torch.logsumexp(
          log_weights, dim=1, keepdim=True)

        ess_before = _effective_sample_size(log_weights)
        xs, log_weights, resampled = _maybe_resample(
          xs, log_weights, ess_threshold)
        xt = xs

        batch_stats.append({
          'step': step_idx,
          'guidance_active': guidance_active,
          'mean_ess': ess_before.mean().item(),
          'min_ess': ess_before.min().item(),
          'resampled_batches': int(resampled.sum().item()),
        })

      output_batches.append(_select_final_particles(
        xt, log_weights, final_selection).detach().cpu())
      all_stats.append({
        'batch': batch_idx,
        'num_particles': num_particles,
        'ess_threshold': ess_threshold,
        'total_steps': total_steps,
        'guidance_schedule': guidance_schedule,
        'requested_guidance_steps': guidance_steps,
        'actual_guidance_steps': len(guidance_step_indices),
        'guidance_step_indices': guidance_step_indices,
        'final_mean_ess': _effective_sample_size(log_weights).mean().item(),
        'final_min_ess': _effective_sample_size(log_weights).min().item(),
        'steps': batch_stats,
      })
  finally:
    if not config.eval.disable_ema:
      model._restore_non_ema_params()

  return torch.cat(output_batches, dim=0), all_stats


def run(model: diffusion.Diffusion, config) -> torch.Tensor:
  model.eval()
  samples, stats = sample(model, config)
  _save_samples(config, model.tokenizer, samples, {
    'smc': {
      'num_particles': int(_cfg_value(config, 'num_particles', 16)),
      'proposal': _cfg_value(config, 'proposal', 'cfg'),
      'proposal_gamma': _cfg_value(config, 'proposal_gamma', 0.0),
      'target': _cfg_value(config, 'target', 'cfg'),
      'target_gamma': _cfg_value(config, 'target_gamma', None),
      'guidance_schedule': _cfg_value(config, 'guidance_schedule', 'all'),
      'guidance_steps': int(_cfg_value(
        config, 'guidance_steps', config.sampling.steps)),
      'condition': _cfg_value(config, 'condition', None),
      'ess_threshold': float(_cfg_value(config, 'ess_threshold', 0.5)),
      'final_selection': _cfg_value(config, 'final_selection', 'sample'),
    },
    'batches': stats,
  })
  return samples
