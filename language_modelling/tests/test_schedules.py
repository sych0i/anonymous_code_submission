import unittest
from types import SimpleNamespace

import numpy as np

from eval import _schedule_warmup_samples, _selected_strategies
from warmup_trace_utils import (
    baseline_schedules,
    interval_steps,
    top_delta_value_steps,
    top_value_steps,
)


class TopValueScheduleTest(unittest.TestCase):
    def test_top_v_uses_only_timesteps_zero_through_t_minus_one(self):
        series = np.asarray([1.0, 9.0, 5.0, 8.0, 7.0, 1000.0])
        self.assertEqual(top_value_steps(series, T=5, T_=2), [1, 3])

    def test_top_dv_uses_v_t_minus_v_t_plus_one(self):
        # Deltas are [3, -2, 6, 1, 2].
        series = np.asarray([10.0, 7.0, 9.0, 3.0, 2.0, 0.0])
        self.assertEqual(
            top_delta_value_steps(series, T=5, T_=3),
            [0, 2, 4],
        )

    def test_ties_prefer_smaller_timesteps_and_keep_exact_budget(self):
        series = np.ones(7)
        self.assertEqual(top_value_steps(series, T=6, T_=3), [0, 1, 2])
        self.assertEqual(
            top_delta_value_steps(series, T=6, T_=3),
            [0, 1, 2],
        )

    def test_baseline_collection_contains_both_top_policies(self):
        series = np.asarray([10.0, 7.0, 9.0, 3.0, 2.0, 0.0])
        schedules = baseline_schedules(
            T=5,
            T_=2,
            reference_steps=[1, 4],
            mean_series=series,
        )
        self.assertEqual(schedules["top_v"], [0, 2])
        self.assertEqual(schedules["top_dv"], [0, 2])

    def test_cli_aliases_map_to_canonical_names(self):
        config = SimpleNamespace(
            run_all=SimpleNamespace(
                strategies=["top_v", "top dv"],
            )
        )
        self.assertEqual(_selected_strategies(config), {"topv", "topdv"})

        config.run_all.strategies = ["all", "fullstep"]
        self.assertEqual(_selected_strategies(config), {"allstep"})

        config.run_all.strategies = ["first", "middlekstep", "last"]
        self.assertEqual(
            _selected_strategies(config),
            {"interval1", "interval3", "interval5"},
        )

    def test_interval_endpoints_are_first_and_last_for_every_budget(self):
        for budget in (5, 10, 15, 20, 25, 30):
            self.assertEqual(
                interval_steps(T=100, T_=budget, interval=1),
                list(range(100 - budget, 100)),
            )
            self.assertEqual(
                interval_steps(T=100, T_=budget, interval=5),
                list(range(budget)),
            )

    def test_five_intervals_are_evenly_positioned(self):
        starts = [
            interval_steps(T=100, T_=20, interval=interval)[0]
            for interval in range(1, 6)
        ]
        self.assertEqual(starts, [80, 60, 40, 20, 0])

        for budget in (5, 10, 15, 20, 25, 30):
            schedules = [
                interval_steps(T=100, T_=budget, interval=interval)
                for interval in range(1, 6)
            ]
            self.assertTrue(all(len(schedule) == budget for schedule in schedules))
            self.assertTrue(
                all(0 <= schedule[0] <= schedule[-1] < 100 for schedule in schedules)
            )

    def test_schedule_warmups_are_selected_by_exact_tag(self):
        samples = [
            SimpleNamespace(tag="p2_r0_allstep"),
            SimpleNamespace(tag="p0_r0_allstep"),
            SimpleNamespace(tag="p1_r0_allstep"),
            SimpleNamespace(tag="p0_r1_allstep"),
        ]
        config = SimpleNamespace(
            run_all=SimpleNamespace(
                eval_schedule_warmup_tags=[
                    "p0_r0_allstep",
                    "p1_r0_allstep",
                    "p2_r0_allstep",
                ]
            )
        )
        selected = _schedule_warmup_samples(config, samples)
        self.assertEqual(
            [sample.tag for sample in selected],
            ["p0_r0_allstep", "p1_r0_allstep", "p2_r0_allstep"],
        )


if __name__ == "__main__":
    unittest.main()
