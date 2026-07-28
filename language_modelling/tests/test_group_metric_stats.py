import math
from pathlib import Path

import pytest

from evaluation.group_metric_stats import (
    STD_SECTION_START,
    group_distinctness,
    group_means,
    group_perplexities,
    group_unique_ratios,
    group_unique_ratio_sweep,
    mean_sample_std,
    particle_group_slices,
    replace_metric_std_section,
)


def test_particle_groups_do_not_cross_prompt_rows():
    assert particle_group_slices([4, 6], 3) == [
        (0, 3),
        (3, 4),
        (4, 7),
        (7, 10),
    ]


def test_group_accuracy_and_sample_std():
    slices = particle_group_slices([8, 8], 8)
    scores = group_means([1] * 8 + [0, 0, 0, 0, 1, 1, 1, 1], slices)
    assert scores == [1.0, 0.5]
    mean, std, count = mean_sample_std(scores)
    assert mean == 0.75
    assert std == pytest.approx(math.sqrt(0.125))
    assert count == 2


def test_group_perplexity_uses_generation_and_token_weighted_views():
    slices = [(0, 2)]
    mean_ppl, total_ppl = group_perplexities(
        [2.0, 8.0], [math.log(2.0), 3.0 * math.log(8.0)], [1, 3], slices
    )
    assert mean_ppl == [5.0]
    assert total_ppl == pytest.approx([2.0 ** 2.5])


def test_group_distinctness_and_unique_ratio():
    texts = ["a b", "a b", "c d", "e f"]
    slices = [(0, 2), (2, 4)]
    dist1, _, _ = group_distinctness(texts, slices)
    assert dist1 == [0.5, 1.0]
    assert group_unique_ratios(texts, slices, 0.0, 2) == [0.5, 1.0]
    assert group_unique_ratios(
        texts, slices, 0.0, 2, [1, 0, 1, 0]
    ) == [0.5, 0.5]
    assert group_unique_ratio_sweep(texts, slices, [0.0, 1.0], 2) == {
        0.0: [0.5, 1.0],
        1.0: [0.5, 0.5],
    }


def test_replace_std_section_is_idempotent_and_preserves_other_text(tmp_path: Path):
    result = tmp_path / "eval.txt"
    result.write_text("metric = 1\n\ntiming = 2\n", encoding="utf-8")
    replace_metric_std_section(result, {"metric": [1.0, 3.0]}, 8)
    replace_metric_std_section(result, {"metric": [2.0, 4.0]}, 8)
    content = result.read_text(encoding="utf-8")
    assert content.count(STD_SECTION_START) == 1
    assert "metric: mean=3, std=1.414213562, n=2" in content
    assert "timing = 2" in content
