#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from warmup_trace_utils import (  # noqa: E402
    calculate_skipped_guidance_sum,
    dp_timestep_list_vista,
    interval_steps,
    load_warmup_samples,
    top_delta_value_steps,
    top_value_steps,
    uniform_steps,
)


RELEASE_ROOT = ROOT / "outputs" / "paper_toxic" / "20260724_120627"
POLICIES = (
    "vista",
    "topv",
    "topdv",
    "uniformstep",
    "interval1",
    "interval2",
    "interval3",
    "interval4",
    "interval5",
)
WARMUP_TAGS = ("p0_r0_allstep", "p1_r0_allstep", "p2_r0_allstep")
STD_SECTION = "--- Per-(prompt, run) metric mean/std over particle groups ---"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def nonempty_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def parse_dp(path: Path) -> tuple[dict[str, list[int]], dict[str, float]]:
    schedules: dict[str, list[int]] = {}
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "_steps:" in line:
            name, value = line.split("_steps:", 1)
            schedules[name] = list(ast.literal_eval(value.strip()))
        elif "_score:" in line:
            name, value = line.split("_score:", 1)
            scores[name] = float(value.strip())
    return schedules, scores


def validate_eval_result(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    require(text.count(STD_SECTION) == 1, f"{path}: missing/duplicate std section")
    section = text.split(STD_SECTION, 1)[1]
    metric_lines = [line for line in section.splitlines() if ": mean=" in line]
    require(len(metric_lines) == 14, f"{path}: expected 14 group metrics")
    unique_metric_lines = [
        line for line in metric_lines if line.startswith("unique_edit#")
    ]
    require(
        len(unique_metric_lines) == 3
        and all(line.startswith("unique_edit#0.05 ") for line in unique_metric_lines),
        f"{path}: expected only threshold 0.05 unique-edit metrics",
    )
    unique_count_lines = [
        line for line in metric_lines if line.startswith("unique_edit_count#")
    ]
    require(
        len(unique_count_lines) == 3
        and all(
            line.startswith("unique_edit_count#0.05 ")
            for line in unique_count_lines
        ),
        f"{path}: expected threshold 0.05 raw unique-count metrics",
    )
    require(
        all("n=30" in line for line in metric_lines),
        f"{path}: every group metric must use n=30",
    )
    require(
        not any(re.search(r"\b(?:nan|inf)\b", line, re.I) for line in metric_lines),
        f"{path}: non-finite group metric",
    )
    for key in ("runs", "mean_seconds_per_run", "std_seconds_per_run"):
        require(re.search(rf"^{key}:", text, re.M) is not None, f"{path}: no {key}")


def validate_strategy_artifacts(directory: Path, strategy: str) -> None:
    text_paths = sorted(directory.glob(f"text_samples_p*_r*_{strategy}.jsonl"))
    abc_paths = sorted(directory.glob(f"abc_ssdlm_gen_p*_r*_{strategy}.jsonl"))
    trace_paths = sorted(directory.glob(f"reward_trace_p*_r*_{strategy}.jsonl"))
    require(len(text_paths) == 30, f"{directory}/{strategy}: expected 30 text files")
    require(len(abc_paths) == 30, f"{directory}/{strategy}: expected 30 ABC files")
    require(len(trace_paths) == 30, f"{directory}/{strategy}: expected 30 traces")
    require(
        all(len(nonempty_lines(path)) == 8 for path in text_paths),
        f"{directory}/{strategy}: every text file must contain 8 particles",
    )

    merged_path = directory / f"abc_ssdlm_gen_merged_{strategy}.jsonl"
    merged = [json.loads(line) for line in nonempty_lines(merged_path)]
    require(len(merged) == 15, f"{merged_path}: expected 15 prompts")
    require(
        all(len(row["string"]) == 16 for row in merged),
        f"{merged_path}: expected 2 runs x 8 particles per prompt",
    )
    validate_eval_result(directory / f"eval_results_{strategy}.txt")


def validate_tprime(
    budget: int, mean_series: np.ndarray, warmup_samples
) -> None:
    directory = RELEASE_ROOT / f"Tprime{budget}"
    require(directory.is_dir(), f"missing {directory}")
    schedules, scores = parse_dp(directory / "dp_timestep.txt")
    expected = {
        "allstep": list(range(100)),
        "uniformstep": uniform_steps(100, budget),
        **{
            f"interval{index}": interval_steps(100, budget, index)
            for index in range(1, 6)
        },
        "vista": dp_timestep_list_vista(mean_series, 100, budget),
        "topv": top_value_steps(mean_series, 100, budget),
        "topdv": top_delta_value_steps(mean_series, 100, budget),
    }
    require(schedules == expected, f"T'={budget}: schedules do not reproduce")
    require(set(scores) == set(expected), f"T'={budget}: incomplete DP scores")
    for strategy, steps in expected.items():
        expected_score = calculate_skipped_guidance_sum(100, steps, mean_series)
        require(
            math.isclose(scores[strategy], expected_score, rel_tol=0, abs_tol=1e-10),
            f"T'={budget} {strategy}: DP score mismatch",
        )

    for strategy in POLICIES:
        validate_strategy_artifacts(directory, strategy)

    timing = [json.loads(line) for line in nonempty_lines(directory / "inference_timing.jsonl")]
    counts = Counter(row["tag"].rsplit("_", 1)[-1] for row in timing)
    require(
        counts == Counter({strategy: 30 for strategy in POLICIES}),
        f"T'={budget}: timing records are incomplete",
    )
    require(
        [sample.tag for sample in warmup_samples if sample.tag in WARMUP_TAGS]
        == list(WARMUP_TAGS),
        "warmup tags are not available in manifest order",
    )


def validate_checksums() -> None:
    manifest_path = ROOT / "MANIFEST.sha256"
    require(manifest_path.is_file(), "missing MANIFEST.sha256")
    for line in nonempty_lines(manifest_path):
        digest, relative = line.rstrip("\n").split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"checksum target missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == digest, f"checksum mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="also verify every file listed in MANIFEST.sha256",
    )
    args = parser.parse_args()

    config = OmegaConf.load(ROOT / "configs" / "eval.yaml")
    require(
        list(config.run_all.eval_schedule_warmup_tags) == list(WARMUP_TAGS),
        "config must select exactly the three released schedule traces",
    )
    require(
        int(config.run_all.eval_schedule_warmup_samples) == len(WARMUP_TAGS),
        "config fallback schedule warmup size must be M=3",
    )
    require(
        "warmup_sample_sizes" not in config.run_all,
        "obsolete warmup sensitivity sweep must not be in the released config",
    )

    allstep = RELEASE_ROOT / "allstep"
    manifest = allstep / "allstep_samples_manifest.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    require(manifest_data.get("T") == 100, "allstep manifest must use T=100")
    require(
        manifest_data.get("T_prime") == 100,
        "allstep manifest must record all 100 guided timesteps",
    )
    prompt_path = allstep / manifest_data["prompt_file"]
    require(prompt_path.resolve().is_file(), "manifest prompt_file is not portable")
    warmup_samples = load_warmup_samples(
        trace_manifest=str(manifest),
        trace_dir=None,
        trace_glob="reward_trace_*allstep*.jsonl",
    )
    require(len(warmup_samples) == 30, "allstep manifest must contain 30 samples")
    require(
        all(len(sample.series) == 101 for sample in warmup_samples),
        "allstep value series must include t=0,...,T",
    )
    require(
        all(
            Path(value).resolve().is_relative_to(allstep.resolve())
            for sample in warmup_samples
            for value in (sample.trace_path, sample.text_path, sample.abc_path)
            if value is not None
        ),
        "allstep manifest paths must resolve inside the released package",
    )
    validate_strategy_artifacts(allstep, "allstep")

    by_tag = {sample.tag: sample for sample in warmup_samples}
    mean_series = np.mean(
        np.asarray([by_tag[tag].series for tag in WARMUP_TAGS], dtype=float),
        axis=0,
    )
    for budget in (5, 10, 15):
        validate_tprime(budget, mean_series, warmup_samples)

    if args.checksums:
        validate_checksums()
    print("PASS: released allstep and T'=5,10,15 artifacts are internally consistent")


if __name__ == "__main__":
    main()
