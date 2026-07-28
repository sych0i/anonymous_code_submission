from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from text_samples_edit_distance import (
    pairwise_levenshtein_bounded,
    unique_count_clustering,
    unique_count_for_texts,
)


STD_SECTION_START = "--- Per-(prompt, run) metric mean/std over particle groups ---"
STD_SECTION_END = "--- End per-(prompt, run) metric mean/std ---"


def particle_group_slices(
    row_lengths: Sequence[int], num_particles: int
) -> list[tuple[int, int]]:
    """Return flat slices without allowing a particle group to cross a prompt row."""
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")

    slices: list[tuple[int, int]] = []
    offset = 0
    for row_length in row_lengths:
        if row_length < 0:
            raise ValueError("row lengths must be non-negative")
        for local_start in range(0, row_length, num_particles):
            start = offset + local_start
            slices.append((start, min(start + num_particles, offset + row_length)))
        offset += row_length
    return slices


def group_means(
    values: Sequence[float], group_slices: Sequence[tuple[int, int]]
) -> list[float]:
    means = []
    for start, end in group_slices:
        chunk = [float(value) for value in values[start:end] if math.isfinite(value)]
        means.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return means


def distinctness_for_texts(texts: Sequence[str]) -> tuple[float, float, float]:
    unigrams: set[str] = set()
    bigrams: set[str] = set()
    trigrams: set[str] = set()
    total_words = 0
    for text in texts:
        words = text.split(" ")
        total_words += len(words)
        unigrams.update(words)
        bigrams.update(
            words[index] + "_" + words[index + 1]
            for index in range(len(words) - 1)
        )
        trigrams.update(
            words[index] + "_" + words[index + 1] + "_" + words[index + 2]
            for index in range(len(words) - 2)
        )
    if total_words == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        len(unigrams) / total_words,
        len(bigrams) / total_words,
        len(trigrams) / total_words,
    )


def group_distinctness(
    texts: Sequence[str], group_slices: Sequence[tuple[int, int]]
) -> tuple[list[float], list[float], list[float]]:
    scores = [
        distinctness_for_texts(texts[start:end]) for start, end in group_slices
    ]
    if not scores:
        return [], [], []
    dist1, dist2, dist3 = zip(*scores)
    return list(dist1), list(dist2), list(dist3)


def group_unique_ratios(
    texts: Sequence[str],
    group_slices: Sequence[tuple[int, int]],
    threshold: float,
    num_particles: int,
    filter_labels: Sequence[int] | None = None,
) -> list[float]:
    if filter_labels is not None and len(filter_labels) != len(texts):
        raise ValueError("filter_labels length must match texts")

    ratios = []
    for start, end in group_slices:
        chunk = texts[start:end]
        if filter_labels is not None:
            chunk = [
                text
                for text, label in zip(chunk, filter_labels[start:end])
                if label == 1
            ]
        unique = unique_count_for_texts(chunk, threshold, "max_len")
        ratios.append(unique / float(num_particles))
    return ratios


def group_unique_ratio_sweep(
    texts: Sequence[str],
    group_slices: Sequence[tuple[int, int]],
    thresholds: Sequence[float],
    num_particles: int,
    filter_labels: Sequence[int] | None = None,
) -> dict[float, list[float]]:
    """Compute pairwise edit distances once per group and reuse them by threshold."""
    if filter_labels is not None and len(filter_labels) != len(texts):
        raise ValueError("filter_labels length must match texts")

    ratios = {threshold: [] for threshold in thresholds}
    for start, end in group_slices:
        chunk = list(texts[start:end])
        if filter_labels is not None:
            chunk = [
                text
                for text, label in zip(chunk, filter_labels[start:end])
                if label == 1
            ]
        raw_distances = pairwise_levenshtein_bounded(
            chunk, max(thresholds, default=0.0), "max_len"
        )
        for threshold in thresholds:
            unique, _ = unique_count_clustering(
                chunk, raw_distances, threshold, "max_len"
            )
            ratios[threshold].append(unique / float(num_particles))
    return ratios


def group_perplexities(
    perplexities: Sequence[float],
    nlls: Sequence[float],
    token_counts: Sequence[int],
    group_slices: Sequence[tuple[int, int]],
) -> tuple[list[float], list[float]]:
    if not (len(perplexities) == len(nlls) == len(token_counts)):
        raise ValueError("PPL detail arrays must have matching lengths")

    mean_ppls = []
    total_ppls = []
    for start, end in group_slices:
        valid_ppls = [
            float(ppl)
            for ppl in perplexities[start:end]
            if math.isfinite(ppl) and ppl < 1e4
        ]
        mean_ppls.append(
            sum(valid_ppls) / len(valid_ppls) if valid_ppls else float("nan")
        )

        group_nll = sum(float(value) for value in nlls[start:end])
        group_tokens = sum(int(value) for value in token_counts[start:end])
        total_ppls.append(
            math.exp(group_nll / group_tokens)
            if group_tokens > 0
            else float("nan")
        )
    return mean_ppls, total_ppls


def mean_sample_std(values: Iterable[float]) -> tuple[float, float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan"), 0
    mean = sum(finite) / len(finite)
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return mean, std, len(finite)


def format_metric_std_section(
    metric_groups: dict[str, Sequence[float]], num_particles: int
) -> str:
    lines = [
        STD_SECTION_START,
        f"unit: one (prompt, run) group of {num_particles} particles",
        "std: sample standard deviation across groups (ddof=1)",
    ]
    for name, values in metric_groups.items():
        mean, std, count = mean_sample_std(values)
        lines.append(f"{name}: mean={mean:.10g}, std={std:.10g}, n={count}")
    lines.append(STD_SECTION_END)
    return "\n".join(lines) + "\n"


def replace_metric_std_section(
    result_path: str | Path,
    metric_groups: dict[str, Sequence[float]],
    num_particles: int,
) -> None:
    path = Path(result_path)
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    start = content.find(STD_SECTION_START)
    if start >= 0:
        end = content.find(STD_SECTION_END, start)
        if end < 0:
            raise ValueError(f"Unterminated metric std section in {path}")
        end += len(STD_SECTION_END)
        content = content[:start].rstrip() + "\n" + content[end:].lstrip("\n")

    section = format_metric_std_section(metric_groups, num_particles)
    if content and not content.endswith("\n"):
        content += "\n"
    path.write_text(content + "\n" + section, encoding="utf-8")
