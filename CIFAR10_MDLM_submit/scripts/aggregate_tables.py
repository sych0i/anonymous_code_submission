#!/usr/bin/env python3
"""Aggregate the CIFAR-10 MDLM table sweep across random seeds."""

import argparse
import csv
import json
import os
import statistics


DEFAULT_BUDGETS = (5, 15, 25)
DEFAULT_SCHEDULES = (
  'interval1', 'interval2', 'interval3', 'interval4', 'interval5',
  'uniform', 'vista', 'top_v', 'top_dv')
POLICY_DISPLAY_NAMES = {
  'full': 'Full',
  'interval1': 'Interval 1',
  'interval2': 'Interval 2',
  'interval3': 'Interval 3',
  'interval4': 'Interval 4',
  'interval5': 'Interval 5',
  'uniform': 'Uniform',
  'vista': 'VISTA-DP',
  'top_v': 'Top V',
  'top_dv': 'Top dV',
}
METRICS = {
  'reward': lambda row: row['mean_reward'],
  'success_rate': lambda row: row['success_rate'],
  'executed_time_seconds': lambda row: row['generation_seconds'],
  'unique_success_lpips003': (
    lambda row: row['unique_success_count']['num_unique_successes']),
}


def _parse_int_list(text):
  return tuple(int(value.strip()) for value in text.split(',') if value.strip())


def _parse_str_list(text):
  return tuple(value.strip() for value in text.split(',') if value.strip())


def _load_run(input_root, budget, seed, schedules):
  path = os.path.join(
    input_root, f't{budget}',
    f'vista_smc_seed{seed}_t{budget}_comparison.json')
  with open(path, encoding='utf-8') as handle:
    run = json.load(handle)

  metadata = run['metadata']
  expected = {
    'guidance_steps': budget,
    'seed': seed,
    'steps': 1000,
    'num_particles': 20,
    'num_rollout_samples': 5,
    'num_warmup_runs': 5,
    'ess_threshold': 0.95,
    'partial_resample': 10,
    'reward_threshold': 0.5,
    'lpips_threshold': 0.03,
    'proposal_method': 'base',
  }
  mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise ValueError(f'{path}: metadata mismatch: {mismatches}')

  missing = [name for name in schedules if name not in run['results']]
  if missing:
    raise ValueError(f'{path}: missing schedules: {missing}')
  for schedule in schedules:
    if run['results'][schedule]['num_samples'] != 100:
      raise ValueError(
        f'{path}: {schedule} has '
        f'{run["results"][schedule]["num_samples"]} samples; expected 100')
  return run


def _load_full_run(input_root, seed):
  path = os.path.join(
    input_root, 'full', f'vista_smc_seed{seed}_full_comparison.json')
  with open(path, encoding='utf-8') as handle:
    run = json.load(handle)

  metadata = run['metadata']
  expected = {
    'guidance_steps': 1000,
    'seed': seed,
    'steps': 1000,
    'num_particles': 20,
    'num_rollout_samples': 5,
    'num_warmup_runs': 5,
    'ess_threshold': 0.95,
    'partial_resample': 10,
    'reward_threshold': 0.5,
    'lpips_threshold': 0.03,
    'proposal_method': 'base',
  }
  mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise ValueError(f'{path}: metadata mismatch: {mismatches}')
  if 'full' not in run['results']:
    raise ValueError(f'{path}: missing schedule: full')
  if run['results']['full']['num_samples'] != 100:
    raise ValueError(
      f'{path}: full has {run["results"]["full"]["num_samples"]} samples; '
      'expected 100')
  return run


def _empirical_schedule_objective(run, schedule):
  """Return Eq. (12), including support for JSONs made before s_hat_g."""
  row = run['results'][schedule]
  if row.get('s_hat_g') is not None:
    return float(row['s_hat_g'])

  try:
    warmup = run['results']['vista_schedule_meta']
    values = warmup['v_hat_by_t']
    prior = warmup['v_hat_prior']
    total_steps = run['metadata']['steps']
    guided_t = sorted(
      total_steps - 1 - int(step_idx)
      for step_idx in row['guided_step_indices'])
  except KeyError as error:
    raise ValueError(
      f'Cannot compute s_hat_g for {schedule!r}: missing {error.args[0]!r}. '
      'The run must contain VISTA warmup values and guided-step indices.'
    ) from error

  if len(values) != total_steps:
    raise ValueError(
      f'VISTA warmup has {len(values)} values; expected {total_steps}.')
  guided = sorted(set(guided_t))
  if any(t < 0 or t >= total_steps for t in guided):
    raise ValueError(f'Invalid guided paper-t indices: {guided}')

  objective = 0.0
  previous_t = -1
  for active_t in guided:
    objective += (
      (active_t - previous_t - 1) * float(values[active_t]))
    previous_t = active_t
  objective += (total_steps - previous_t - 1) * float(prior)
  return objective


def aggregate(input_root, budgets, schedules, seeds, include_full=False):
  runs = {
    (budget, seed): _load_run(input_root, budget, seed, schedules)
    for budget in budgets
    for seed in seeds
  }
  full_runs = (
    {seed: _load_full_run(input_root, seed) for seed in seeds}
    if include_full else None)
  output = {
    'metadata': {
      'budgets': list(budgets),
      'schedules': list(schedules),
      'seeds': list(seeds),
      'aggregation': 'arithmetic mean and sample standard deviation across seeds',
      'samples_per_schedule_per_seed': 100,
      'lpips_threshold': 0.03,
      'reward_threshold': 0.5,
    },
    'tables': {},
  }
  for metric_name, getter in METRICS.items():
    table = {}
    for schedule in schedules:
      table[schedule] = {}
      for budget in budgets:
        values = [
          float(getter(runs[(budget, seed)]['results'][schedule]))
          for seed in seeds
        ]
        table[schedule][str(budget)] = {
          'mean': statistics.mean(values),
          'std': statistics.stdev(values),
          'per_seed': values,
        }
    output['tables'][metric_name] = table

  score_table = {}
  for schedule in schedules:
    score_table[schedule] = {}
    for budget in budgets:
      values = [
        _empirical_schedule_objective(runs[(budget, seed)], schedule)
        for seed in seeds
      ]
      score_table[schedule][str(budget)] = {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values),
        'per_seed': values,
      }
  output['tables']['s_hat_g'] = score_table

  if full_runs is not None:
    full_metrics = dict(METRICS)
    full_metrics['s_hat_g'] = lambda _row: 0.0
    output['full_reference'] = {}
    for metric_name, getter in full_metrics.items():
      values = [
        float(getter(full_runs[seed]['results']['full']))
        for seed in seeds
      ]
      output['full_reference'][metric_name] = {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values),
        'per_seed': values,
      }
  return output


def _write_csv(path, table, budgets, schedules):
  with open(path, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['schedule'] + [f"T'={budget}" for budget in budgets])
    for schedule in schedules:
      cells = []
      for budget in budgets:
        cell = table[schedule][str(budget)]
        cells.append(f'{cell["mean"]:.4f} ± {cell["std"]:.4f}')
      writer.writerow([schedule] + cells)


def _write_markdown(path, tables, budgets, schedules):
  labels = {
    'reward': 'Reward',
    'success_rate': 'Success rate',
    'executed_time_seconds': 'Executed time [s] (warm-up excluded)',
    'unique_success_lpips003': (
      'Unique success @ LPIPS ≤ 0.03, reward > 0.5'),
    's_hat_g': r'Empirical schedule objective $\hat{S}(G)$',
  }
  with open(path, 'w', encoding='utf-8') as handle:
    for metric_name, table in tables.items():
      handle.write(f'## {labels[metric_name]}\n\n')
      handle.write('| schedule | ' + ' | '.join(
        f"T'={budget}" for budget in budgets) + ' |\n')
      handle.write('|---|' + '|'.join('---:' for _ in budgets) + '|\n')
      decimals = {
        'unique_success_lpips003': 1,
        'executed_time_seconds': 2,
        's_hat_g': 4,
      }.get(metric_name, 4)
      for schedule in schedules:
        cells = []
        for budget in budgets:
          cell = table[schedule][str(budget)]
          cells.append(
            f'{cell["mean"]:.{decimals}f} ± '
            f'{cell["std"]:.{decimals}f}')
        handle.write(f'| {schedule} | ' + ' | '.join(cells) + ' |\n')
      handle.write('\n')


def _policy_table_lines(
    tables, budgets, schedules, full_reference=None):
  yield (
    r"| T' | Policy | executed time [s] | unique success | "
    r"$\hat{S}(G)$ |")
  yield '|---:|:---|---:|---:|---:|'
  if full_reference is not None:
    time_cell = full_reference['executed_time_seconds']
    unique_cell = full_reference['unique_success_lpips003']
    score_cell = full_reference['s_hat_g']
    yield (
      f'| 1000 | {POLICY_DISPLAY_NAMES["full"]} | '
      f'{time_cell["mean"]:.2f} ± {time_cell["std"]:.2f} | '
      f'{unique_cell["mean"]:.1f} ± {unique_cell["std"]:.1f} | '
      f'{score_cell["mean"]:.4g} ± {score_cell["std"]:.4g} |')
  for budget in budgets:
    for schedule in schedules:
      time_cell = tables['executed_time_seconds'][schedule][str(budget)]
      unique_cell = tables['unique_success_lpips003'][schedule][str(budget)]
      score_cell = tables['s_hat_g'][schedule][str(budget)]
      yield (
        f'| {budget} | {POLICY_DISPLAY_NAMES.get(schedule, schedule)} | '
        f'{time_cell["mean"]:.2f} ± {time_cell["std"]:.2f} | '
        f'{unique_cell["mean"]:.1f} ± {unique_cell["std"]:.1f} | '
        f'{score_cell["mean"]:.4g} ± {score_cell["std"]:.4g} |')


def _write_policy_table_csv(
    path, tables, budgets, schedules, full_reference=None):
  with open(path, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(
      ["T'", 'Policy', 'executed time [s]', 'unique success', 'S_hat(G)'])
    if full_reference is not None:
      time_cell = full_reference['executed_time_seconds']
      unique_cell = full_reference['unique_success_lpips003']
      score_cell = full_reference['s_hat_g']
      writer.writerow([
        1000,
        POLICY_DISPLAY_NAMES['full'],
        f'{time_cell["mean"]:.2f} ± {time_cell["std"]:.2f}',
        f'{unique_cell["mean"]:.1f} ± {unique_cell["std"]:.1f}',
        f'{score_cell["mean"]:.4g} ± {score_cell["std"]:.4g}',
      ])
    for budget in budgets:
      for schedule in schedules:
        time_cell = tables['executed_time_seconds'][schedule][str(budget)]
        unique_cell = tables['unique_success_lpips003'][schedule][str(budget)]
        score_cell = tables['s_hat_g'][schedule][str(budget)]
        writer.writerow([
          budget,
          POLICY_DISPLAY_NAMES.get(schedule, schedule),
          f'{time_cell["mean"]:.2f} ± {time_cell["std"]:.2f}',
          f'{unique_cell["mean"]:.1f} ± {unique_cell["std"]:.1f}',
          f'{score_cell["mean"]:.4g} ± {score_cell["std"]:.4g}',
        ])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input-root', default='eval_runs/tables')
  parser.add_argument('--output-dir', default='eval_runs/tables/summary')
  parser.add_argument(
    '--budgets', default=','.join(str(value) for value in DEFAULT_BUDGETS))
  parser.add_argument(
    '--schedules', default=','.join(DEFAULT_SCHEDULES))
  parser.add_argument('--seeds', default='1,2,3,4,5,6,7,8,9,10')
  parser.add_argument(
    '--exclude-full', action='store_true',
    help='Do not load or print the T=1000 full-guidance reference runs.')
  args = parser.parse_args()

  budgets = _parse_int_list(args.budgets)
  schedules = _parse_str_list(args.schedules)
  seeds = _parse_int_list(args.seeds)
  summary = aggregate(
    args.input_root, budgets, schedules, seeds,
    include_full=not args.exclude_full)

  os.makedirs(args.output_dir, exist_ok=True)
  with open(
      os.path.join(args.output_dir, 'tables.json'),
      'w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2)
  for metric_name, table in summary['tables'].items():
    _write_csv(
      os.path.join(args.output_dir, f'{metric_name}.csv'),
      table, budgets, schedules)
  _write_markdown(
    os.path.join(args.output_dir, 'tables.md'),
    summary['tables'], budgets, schedules)
  policy_lines = list(
    _policy_table_lines(
      summary['tables'], budgets, schedules,
      summary.get('full_reference')))
  with open(
      os.path.join(args.output_dir, 'policy_table.md'),
      'w', encoding='utf-8') as handle:
    handle.write('\n'.join(policy_lines) + '\n')
  _write_policy_table_csv(
    os.path.join(args.output_dir, 'policy_table.csv'),
    summary['tables'], budgets, schedules,
    summary.get('full_reference'))
  print('\n'.join(policy_lines))
  print(f'Wrote tables to {args.output_dir}')


if __name__ == '__main__':
  main()
