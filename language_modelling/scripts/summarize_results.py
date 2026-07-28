#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "paper_toxic" / "20260724_120627"
POLICIES = (
    "interval1",
    "interval2",
    "interval3",
    "interval4",
    "interval5",
    "topv",
    "topdv",
    "uniformstep",
    "vista",
)


def result_values(path: Path) -> tuple[float, float, float, float]:
    text = path.read_text(encoding="utf-8")
    unique = re.search(
        r"^unique_edit_count#0\.05 toxic: mean=([^,]+), std=([^,]+)",
        text,
        re.MULTILINE,
    )
    mean_time = re.search(r"^mean_seconds_per_run: ([^\n]+)", text, re.MULTILINE)
    std_time = re.search(r"^std_seconds_per_run: ([^\n]+)", text, re.MULTILINE)
    if unique is None or mean_time is None or std_time is None:
        raise ValueError(f"Incomplete result file: {path}")
    return (
        float(mean_time.group(1)),
        float(std_time.group(1)),
        float(unique.group(1)),
        float(unique.group(2)),
    )


def main() -> None:
    print("| T' | Policy | Time (s) | Unique toxic | DP score |")
    print("|---:|---|---:|---:|---:|")
    time_mean, time_std, unique_mean, unique_std = result_values(
        RESULTS / "allstep" / "eval_results_allstep.txt"
    )
    print(
        f"| All | allstep | {time_mean:.3g} +/- {time_std:.2g} | "
        f"{unique_mean:.3g} +/- {unique_std:.2g} | 0 |"
    )
    for budget in (5, 10, 15):
        directory = RESULTS / f"Tprime{budget}"
        dp_text = (directory / "dp_timestep.txt").read_text(encoding="utf-8")
        for policy in POLICIES:
            time_mean, time_std, unique_mean, unique_std = result_values(
                directory / f"eval_results_{policy}.txt"
            )
            match = re.search(rf"^{policy}_score: ([^\n]+)", dp_text, re.MULTILINE)
            if match is None:
                raise ValueError(f"Missing {policy}_score in {directory}")
            score = float(match.group(1))
            label = {
                "uniformstep": "uniform",
                "topv": "top v",
                "topdv": "top dv",
            }.get(policy, policy.replace("interval", "interval "))
            print(
                f"| {budget} | {label} | {time_mean:.3g} +/- {time_std:.2g} | "
                f"{unique_mean:.3g} +/- {unique_std:.2g} | {score:.4g} |"
            )


if __name__ == "__main__":
    main()
