"""Aggregate independently generated VISTA-SMC sample files.

The script rescales neither rewards nor diversity counts by run. It concatenates
all token samples, decodes and classifies the union, and computes LPIPS greedy
unique successes across the complete set so cross-run duplicates are counted.
"""

import argparse
import json
import os

import torch
import torchvision

import dataloader
import vista_smc


def _aggregate_final_pool_diagnostics(payloads):
  """Average saved per-batch pool metrics over their output groups."""
  weighted_sums = {}
  total_groups = 0
  diagnosed_files = 0
  for payload in payloads:
    stats = payload.get('stats')
    if not stats:
      continue
    diagnostics = [
      batch.get('final_pool_diagnostics') for batch in stats
      if batch.get('final_pool_diagnostics') is not None
    ]
    if not diagnostics:
      continue
    if len(diagnostics) != len(stats):
      raise ValueError(
        'A sample file has pool diagnostics for only some output batches.')
    num_samples = int(payload['samples'].shape[0])
    if num_samples % len(stats):
      raise ValueError(
        'Sample count is not divisible by the number of saved batches.')
    groups_per_batch = num_samples // len(stats)
    diagnosed_files += 1
    for diagnostic in diagnostics:
      for key, value in diagnostic.items():
        weighted_sums[key] = (
          weighted_sums.get(key, 0.0) + float(value) * groups_per_batch)
      total_groups += groups_per_batch
  if not total_groups:
    return None
  return {
    'num_diagnosed_files': diagnosed_files,
    'num_output_groups': total_groups,
    **{
      key: value / total_groups
      for key, value in weighted_sums.items()
    },
  }


def _counterfactual_margin_samples(payloads):
  """Recover the pure-SMC draws retained by diagnostic margin runs."""
  groups = []
  found_any = False
  missing_any = False
  for payload in payloads:
    for batch in payload.get('stats') or []:
      tokens = batch.get('counterfactual_sample_tokens')
      if tokens is None:
        missing_any = True
      else:
        found_any = True
        groups.append(tokens)
  if found_any and missing_any:
    raise ValueError(
      'Counterfactual sampled-SMC tokens are missing from some batches.')
  if not found_any:
    return None
  return torch.cat(groups, dim=0)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--sample-files', nargs='+', required=True)
  parser.add_argument(
    '--config', default='outputs/cifar10/mdlm/.hydra/config.yaml')
  parser.add_argument(
    '--classifier-checkpoint',
    default='outputs/cifar10/reward_classifier/best.ckpt')
  parser.add_argument('--target-class', type=int, default=0)
  parser.add_argument('--alpha', type=float, default=1.0)
  parser.add_argument('--reward-threshold', type=float, default=0.5)
  parser.add_argument('--lpips-threshold', type=float, default=0.03)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--output-prefix', required=True)
  args = parser.parse_args()

  sample_groups = []
  payloads = []
  for path in args.sample_files:
    payload = torch.load(path, map_location='cpu')
    if 'samples' not in payload:
      raise KeyError(f'{path} has no "samples" tensor.')
    payloads.append(payload)
    sample_groups.append(payload['samples'])
  samples = torch.cat(sample_groups, dim=0)
  pool_diagnostics = _aggregate_final_pool_diagnostics(payloads)
  counterfactual_samples = _counterfactual_margin_samples(payloads)

  config = vista_smc._load_config(args.config)
  tokenizer = dataloader.get_tokenizer(config)
  classifier, classes = vista_smc.load_reward_classifier(
    args.classifier_checkpoint, args.device)
  lpips_model = vista_smc.load_lpips_model(torch.device(args.device))

  images, reward, success_rate = vista_smc._score_samples(
    classifier=classifier,
    tokenizer=tokenizer,
    x0_tokens=samples,
    target_class=args.target_class,
    alpha=args.alpha,
    device=torch.device(args.device))
  unique_metric = vista_smc._unique_success_count(
    images=images,
    reward=reward,
    reward_threshold=args.reward_threshold,
    lpips_model=lpips_model,
    lpips_threshold=args.lpips_threshold)
  counterfactual_result = None
  counterfactual_reward = None
  if counterfactual_samples is not None:
    (
      _counterfactual_images,
      counterfactual_reward,
      counterfactual_success_rate,
    ) = vista_smc._score_samples(
      classifier=classifier,
      tokenizer=tokenizer,
      x0_tokens=counterfactual_samples,
      target_class=args.target_class,
      alpha=args.alpha,
      device=torch.device(args.device))
    counterfactual_result = {
      'num_samples': int(counterfactual_samples.shape[0]),
      'mean_reward': counterfactual_reward.mean().item(),
      'reward_std': counterfactual_reward.std().item(),
      'mean_target_probability': (
        counterfactual_reward.pow(args.alpha).mean().item()),
      'target_probability_std': (
        counterfactual_reward.pow(args.alpha).std().item()),
      'success_rate': counterfactual_success_rate,
    }

  output_dir = os.path.dirname(args.output_prefix)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  torch.save({
    'samples': samples,
    'reward': reward.detach().cpu(),
    'counterfactual_sampled_smc_samples': counterfactual_samples,
    'counterfactual_sampled_smc_reward': (
      counterfactual_reward.detach().cpu()
      if counterfactual_reward is not None else None),
    'source_sample_files': args.sample_files,
  }, f'{args.output_prefix}.pt')
  torchvision.utils.save_image(
    images.clamp(0, 1),
    f'{args.output_prefix}.png',
    nrow=max(1, min(16, images.shape[0])))

  result = {
    'metadata': {
      'source_sample_files': args.sample_files,
      'config': args.config,
      'classifier_checkpoint': args.classifier_checkpoint,
      'classes': classes,
      'target_class': args.target_class,
      'target_class_name': classes[args.target_class],
      'alpha': args.alpha,
      'reward_threshold': args.reward_threshold,
      'lpips_threshold': args.lpips_threshold,
      'device': args.device,
    },
    'results': {
      'num_samples': int(samples.shape[0]),
      'mean_reward': reward.mean().item(),
      'reward_std': reward.std().item(),
      'mean_target_probability': reward.pow(args.alpha).mean().item(),
      'target_probability_std': reward.pow(args.alpha).std().item(),
      'success_rate': success_rate,
      'unique_success_count': unique_metric,
    },
  }
  if pool_diagnostics is not None:
    result['results']['final_pool_diagnostics'] = pool_diagnostics
  if counterfactual_result is not None:
    result['results']['counterfactual_sampled_smc'] = (
      counterfactual_result)
  with open(f'{args.output_prefix}.json', 'w') as f:
    json.dump(result, f, indent=2)

  print(json.dumps(result['results'], indent=2))
  print(f'Saved aggregate artifacts to {args.output_prefix}.*')


if __name__ == '__main__':
  main()
