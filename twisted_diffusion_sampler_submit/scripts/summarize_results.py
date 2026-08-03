#!/usr/bin/env python3
"""Validate six policy shards and render vertical UV/UTD tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


POLICY_ORDER = [
    "interval-1",
    "interval-2",
    "interval-3",
    "interval-4",
    "interval-5",
    "uniform",
    "vista",
    "top-V",
    "top-dV",
    "weighted-vista",
    "weighted-vista-2",
    "weighted-vista-3",
]
GROUPS = ("i123", "i45u", "vtt")


def discover_budgets(root: Path, requested: set[int] | None = None) -> list[int]:
    """Find every T-prime for which all groups have a shard directory."""
    budgets_per_group: list[set[int]] = []
    for group in GROUPS:
        found: set[int] = set()
        for path in root.glob(f"tp*_{group}"):
            prefix = path.name[: -len(f"_{group}")]
            if path.name.endswith(f"_{group}") and prefix.startswith("tp"):
                try:
                    found.add(int(prefix[2:]))
                except ValueError:
                    continue
        budgets_per_group.append(found)
    common = set.intersection(*budgets_per_group) if budgets_per_group else set()
    if requested is not None:
        common &= requested
    return sorted(common)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        help="Only aggregate these T-prime budgets (prevents stale shards from leaking into a report).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if UV, schedule, or proxy UTD differs from expected_results.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def collect(
    root: Path, requested_budgets: set[int] | None = None
) -> dict[str, dict[str, dict]]:
    aggregate: dict[str, dict[str, dict]] = {}
    warmup_sources: set[str] = set()
    implementation_hashes: set[str] = set()
    for budget in discover_budgets(root, requested_budgets):
        policies: dict[str, dict] = {}
        for group in GROUPS:
            path = root / f"tp{budget}_{group}" / "summary.json"
            summary = load_json(path)
            config = summary["config"]
            if config["guidance_steps"] != budget:
                raise ValueError(f"wrong T-prime in {path}")
            if config["eval_start_index"] != 0:
                raise ValueError(f"expected eval_start_index=0 in {path}")
            expected_seeds = list(range(config["eval_runs"]))
            warmup_sources.add(config["warmup_from"])
            implementation_hashes.add(config["provenance"]["implementation_sha256"])
            for policy, result in summary["policies"].items():
                if policy in policies:
                    raise ValueError(f"duplicate policy {policy} for T-prime={budget}")
                seeds = [run["seed"] for run in result["runs"]]
                if seeds != expected_seeds:
                    raise ValueError(f"wrong sampling seeds for {policy}: {seeds}")
                uv = [run["metrics"]["unique_valid_count"] for run in result["runs"]]
                schedule = result["schedule"]
                if len(schedule) != budget or schedule != sorted(set(schedule)):
                    raise ValueError(f"invalid schedule for {policy}, T-prime={budget}")
                policies[policy] = {
                    "seedwise_unique_valid": uv,
                    "mean_unique_valid": sum(uv) / len(uv),
                    "proxy_utd": result["proxy_utd"],
                    "schedule": schedule,
                }
        if set(policies) != set(POLICY_ORDER):
            raise ValueError(f"incomplete policy set for T-prime={budget}")
        aggregate[str(budget)] = policies
    if requested_budgets is not None:
        found = {int(budget) for budget in aggregate}
        missing = sorted(requested_budgets - found)
        if missing:
            raise ValueError(
                "requested budgets are incomplete; missing complete shards for "
                + ", ".join(str(budget) for budget in missing)
            )
    if len(warmup_sources) != 1 or len(implementation_hashes) != 1:
        raise ValueError("shards do not share one warmup and implementation")
    return aggregate


def compare(actual: dict, expected_path: Path, strict: bool) -> list[str]:
    expected = load_json(expected_path)["budgets"]
    differences: list[str] = []
    for budget, policies in actual.items():
        for policy, result in policies.items():
            reference = expected[budget][policy]
            if result["seedwise_unique_valid"] != reference["seedwise_unique_valid"]:
                differences.append(f"T'={budget} {policy}: Unique Valid differs")
            if result["schedule"] != reference["schedule"]:
                differences.append(f"T'={budget} {policy}: schedule differs")
            if not math.isclose(result["proxy_utd"], reference["proxy_utd"], rel_tol=1e-7, abs_tol=1e-12):
                differences.append(f"T'={budget} {policy}: proxy UTD differs")
    if strict and differences:
        raise SystemExit("\n".join(differences))
    return differences


def render(aggregate: dict) -> None:
    for budget in sorted(aggregate, key=int):
        print(f"\n### T'={budget}\n")
        print("| Sparse schedule | Unique Valid | UTD ↓ |")
        print("|---|---:|---:|")
        for policy in POLICY_ORDER:
            row = aggregate[budget][policy]
            labels = {
                "vista": "VISTA",
                "weighted-vista": "Weighted VISTA (^1)",
                "weighted-vista-2": "Weighted VISTA (^2)",
                "weighted-vista-3": "Weighted VISTA (^3)",
            }
            label = labels.get(policy, policy)
            print(f"| {label} | {row['mean_unique_valid']:.2f} | {row['proxy_utd']:.4f} |")


def main() -> int:
    args = parse_args()
    requested = set(args.budgets) if args.budgets else None
    aggregate = collect(args.input_dir.resolve(), requested)
    payload = {"schema_version": 1, "budgets": aggregate}
    output = args.input_dir.resolve() / "aggregate_results.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render(aggregate)
    if args.expected:
        differences = compare(aggregate, args.expected.resolve(), args.strict)
        if differences:
            print("\nReference differences (CUDA metric-level determinism is not guaranteed):")
            for difference in differences:
                print(f"- {difference}")
        else:
            print("\nAll reference UV, schedules, and proxy UTD values match.")
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
