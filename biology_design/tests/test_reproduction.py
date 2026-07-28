from __future__ import annotations

import hashlib
import math
from pathlib import Path
import unittest

import numpy as np

from smc.trace import exp_reward_from_trace_row
from smc.vista import (
    build_schedule_map,
    calculate_skipped_guidance_sum,
    top_delta_value_schedule,
)


ROOT = Path(__file__).resolve().parents[1]


class ReproductionTest(unittest.TestCase):
    def test_required_asset_hashes(self):
        expected = {
            ROOT / "model_weights/data_and_model/mdlm/outputs_gosai/pretrained.ckpt":
                "00e6998fe729ca38390d02a3811221bc33d0de6751070e1dade819b4ae1f5129",
            ROOT / "model_weights/data_and_model/mdlm/gosai_data/binary_atac_cell_lines.ckpt":
                "6d1b21f1a2ec894c012aa3fef8b024463693b4e330ce9d2755b3b6f4456b932b",
        }
        for path, digest in expected.items():
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                digest,
            )

    def test_trace_estimator_is_particle_weighted(self):
        row = {
            "reward_aggregated": [0.0, math.log(2.0)],
            "normalized_weights": [0.25, 0.75],
            "num_batches": 1,
            "num_particles": 2,
            "scale_cur": 1.0,
        }
        self.assertEqual(exp_reward_from_trace_row(row), 1.75)

    def test_retained_schedules_match_reference(self):
        values = np.load(
            ROOT / "tests/data/weighted_first3_mean_series.npy"
        )
        expected_vista = {
            8: [5, 13, 24, 43, 58, 65, 119, 123],
            16: [
                5, 13, 24, 43, 58, 65, 69, 119,
                120, 121, 122, 123, 124, 125, 126, 127,
            ],
            24: [
                5, 13, 24, 43, 58, 65, 69, 70, 71, 72, 73, 74,
                75, 76, 77, 119, 120, 121, 122, 123, 124, 125, 126, 127,
            ],
        }
        expected_top_dv = {
            8: [10, 22, 24, 34, 41, 43, 55, 58],
            16: [
                5, 10, 22, 24, 30, 32, 34, 36,
                41, 43, 46, 49, 55, 58, 65, 119,
            ],
            24: [
                2, 5, 10, 13, 14, 22, 24, 27, 30, 32, 34, 36,
                41, 43, 46, 49, 55, 58, 60, 62, 65, 69, 73, 119,
            ],
            32: [
                2, 5, 9, 10, 11, 13, 14, 16, 20, 22, 24, 27, 30, 32,
                34, 36, 39, 41, 43, 46, 49, 55, 58, 60, 62, 65, 69, 73,
                80, 83, 84, 119,
            ],
            40: [
                2, 3, 5, 7, 9, 10, 11, 13, 14, 16, 19, 20, 22, 24,
                25, 27, 30, 32, 34, 36, 39, 41, 43, 44, 46, 49, 55, 58,
                60, 62, 65, 67, 69, 71, 73, 75, 80, 83, 84, 119,
            ],
        }
        expected_vista_scores = {
            8: 12.4017111073,
            16: 12.3763406141,
            24: 12.0790444159,
            32: 11.7817482176,
            40: 11.4844520193,
        }
        expected_top_dv_scores = {
            8: 8.945354469618358,
            16: 10.639124956510432,
            24: 9.576635149925554,
            32: 8.323036575024938,
            40: 7.137143724523229,
        }
        expected_methods = {
            "vista",
            "uniformsteps",
            "top_v",
            "top_dv",
            "interval1",
            "interval2",
            "interval3",
            "interval4",
            "interval5",
        }
        for budget, expected_score in expected_vista_scores.items():
            schedules = build_schedule_map(128, budget, values)
            self.assertEqual(set(schedules), expected_methods)
            if budget in expected_vista:
                self.assertEqual(
                    sorted(schedules["vista"]),
                    expected_vista[budget],
                )
            score = calculate_skipped_guidance_sum(
                128,
                    schedules["vista"],
                    values,
                )
            self.assertAlmostEqual(score, expected_score, delta=1.0e-10)
            self.assertEqual(
                sorted(schedules["top_dv"]),
                expected_top_dv[budget],
            )
            top_dv_score = calculate_skipped_guidance_sum(
                128,
                schedules["top_dv"],
                values,
            )
            self.assertAlmostEqual(
                top_dv_score,
                expected_top_dv_scores[budget],
                delta=1.0e-10,
            )

    def test_top_dv_uses_signed_adjacent_difference(self):
        values = [5.0, 0.0, 10.0, 9.0, 8.0, 0.0]
        self.assertEqual(
            top_delta_value_schedule(5, 2, values),
            {0, 4},
        )


if __name__ == "__main__":
    unittest.main()
