import json
import os
import tempfile
import unittest

from scripts import aggregate_tables


class AggregateTablesTest(unittest.TestCase):

  def test_mean_and_sample_standard_deviation(self):
    schedules = ('uniform', 'vista')
    budgets = (12, 25)
    seeds = (1, 2)
    with tempfile.TemporaryDirectory() as root:
      for budget in budgets:
        os.makedirs(os.path.join(root, f't{budget}'))
        for seed in seeds:
          value = budget / 100 + seed / 10
          results = {}
          for offset, schedule in enumerate(schedules):
            results[schedule] = {
              'num_samples': 100,
              'generation_seconds': value * 10 + offset,
              'mean_reward': value + offset,
              'success_rate': value / 2 + offset,
              'guided_step_indices': [0],
              'unique_success_count': {
                'num_unique_successes': seed + offset,
              },
            }
          results['vista_schedule_meta'] = {
            'v_hat_by_t': [float(seed)] * 1000,
            'v_hat_prior': 0.0,
            'guided_t': [999],
          }
          run = {
            'metadata': {
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
            },
            'results': results,
          }
          path = os.path.join(
            root, f't{budget}',
            f'vista_smc_seed{seed}_t{budget}_comparison.json')
          with open(path, 'w', encoding='utf-8') as handle:
            json.dump(run, handle)

      os.makedirs(os.path.join(root, 'full'))
      for seed in seeds:
        full_path = os.path.join(
          root, 'full', f'vista_smc_seed{seed}_full_comparison.json')
        with open(full_path, 'w', encoding='utf-8') as handle:
          json.dump({
            'metadata': {
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
            },
            'results': {
              'full': {
                'num_samples': 100,
                'generation_seconds': 20.0 + seed,
                'mean_reward': 0.9,
                'success_rate': 0.8,
                'guided_step_indices': list(range(1000)),
                's_hat_g': 0.0,
                'unique_success_count': {
                  'num_unique_successes': 30 + seed,
                },
              },
            },
          }, handle)

      summary = aggregate_tables.aggregate(
        root, budgets, schedules, seeds, include_full=True)
      reward = summary['tables']['reward']['uniform']['12']
      self.assertAlmostEqual(reward['mean'], 0.27)
      self.assertAlmostEqual(reward['std'], 0.1 / (2 ** 0.5))
      unique = (
        summary['tables']['unique_success_lpips003']['vista']['25'])
      self.assertEqual(unique['mean'], 2.5)
      self.assertAlmostEqual(unique['std'], 1 / (2 ** 0.5))
      executed = (
        summary['tables']['executed_time_seconds']['uniform']['12'])
      self.assertAlmostEqual(executed['mean'], 2.7)
      score = summary['tables']['s_hat_g']['uniform']['12']
      self.assertAlmostEqual(score['mean'], 1498.5)
      self.assertAlmostEqual(score['std'], 999 / (2 ** 0.5))

      lines = list(aggregate_tables._policy_table_lines(
        summary['tables'], budgets, schedules,
        summary['full_reference']))
      self.assertIn(
        "| T' | Policy | executed time [s] | unique success | "
        r"$\hat{S}(G)$ |", lines[0])
      self.assertTrue(any('| 12 | uniform |' in line for line in lines))
      self.assertTrue(any('| 1000 | full |' in line for line in lines))
      self.assertEqual(
        summary['full_reference']['s_hat_g']['mean'], 0.0)

  def test_rejects_wrong_lpips_threshold(self):
    with tempfile.TemporaryDirectory() as root:
      path_dir = os.path.join(root, 't12')
      os.makedirs(path_dir)
      path = os.path.join(
        path_dir, 'vista_smc_seed1_t12_comparison.json')
      with open(path, 'w', encoding='utf-8') as handle:
        json.dump({
          'metadata': {
            'guidance_steps': 12,
            'seed': 1,
            'steps': 1000,
            'num_particles': 20,
            'num_rollout_samples': 5,
            'num_warmup_runs': 5,
            'ess_threshold': 0.95,
            'partial_resample': 10,
            'reward_threshold': 0.5,
            'lpips_threshold': 0.006,
            'proposal_method': 'base',
          },
          'results': {
            'uniform': {
              'num_samples': 100,
              'generation_seconds': 1.0,
              'mean_reward': 0.5,
              'success_rate': 0.5,
              'guided_step_indices': [0],
              'unique_success_count': {'num_unique_successes': 10},
            },
          },
        }, handle)
      with self.assertRaisesRegex(ValueError, 'lpips_threshold'):
        aggregate_tables.aggregate(root, (12,), ('uniform',), (1,))


if __name__ == '__main__':
  unittest.main()
