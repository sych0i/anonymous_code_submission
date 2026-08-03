#!/usr/bin/env python3
"""Validate Protein Fitness artifacts and render Tables E-1 and E-2.

Fresh metric artifacts are read from ``outputs/metrics/{backbone}_b{T'}.json``
and timing artifacts from ``outputs/timing/{backbone}_b{T'}.json``.  Both use
``format: 1`` and the exact configuration returned by
``experiment_spec.table_config(backbone, budget)``.  A metric artifact has
``kind: protein_fitness_metric`` and this relevant shape::

    {"policies": {policy: {
        "guided_set": [...], "shat": 0.0,
        "runs": [{"seed": 1, "elapsed_seconds": 1.0,
                  "combos": [...], "rewards": [...], "unique_valid": 3}]
    }}}

A timing artifact has ``kind: protein_fitness_timing`` and
``{"times": {policy: [seconds_for_each_run]}}``.  Reference tables use::

    {
      "format": 1,
      "kind": "protein_fitness_reference_tables",
      "metadata": {
        "experiment": "protein_fitness", "algo": "smc_base", "T": 128,
        "N": 32, "repeats": 20,
        "backbones": ["mdlm", "udlm"], "budgets": [128, 16, 32]
      },
      "tables": {
        "mdlm": {"rows": [{
          "budget": 128, "policy": "full", "time_mean": 44.53,
          "unique_valid_mean": 9.0, "unique_valid_sd": 2.77
        }]}
      }
    }

The omitted table cells in the example are mandatory.  Full guidance at 128
steps and all nine policies at T'=16 and T'=32 are required for both models.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

try:  # Supports both ``python scripts/...`` and importing ``scripts`` in tests.
    from .experiment_spec import (
        BACKBONES,
        DISPLAY_NAMES,
        N_PARTICLES,
        N_RUNS,
        POLICIES,
        SPARSE_T_PRIMES,
        T,
        table_config,
    )
except ImportError:  # pragma: no cover - exercised by the command-line entry point.
    from experiment_spec import (  # type: ignore
        BACKBONES,
        DISPLAY_NAMES,
        N_PARTICLES,
        N_RUNS,
        POLICIES,
        SPARSE_T_PRIMES,
        T,
        table_config,
    )


BACKBONE_ORDER = ("mdlm", "udlm")
TABLE_BUDGETS = (T, *SPARSE_T_PRIMES)
TABLE_NUMBERS = {"mdlm": "E-1", "udlm": "E-2"}
TIMING_CONFIG_EXTRAS = {
    "benchmark_warmups": 1,
    "priming_seed": 90000,
    "order_seed": 20260802,
    "timed_scope": "SMC_Base.inference(detokenize=True), CUDA synchronized",
}
MARKDOWN_POLICY_NAMES = {
    **DISPLAY_NAMES,
    "full": "Full",
    "top_v": r"Top-$V$",
    "top_dv": r"Top-$\Delta V$",
}
CSV_POLICY_NAMES = {**DISPLAY_NAMES, "full": "Full"}


def _expected_policies(budget: int) -> tuple[str, ...]:
    return ("full",) if budget == T else tuple(POLICIES)


def _table_metadata() -> dict[str, Any]:
    return {
        "experiment": "protein_fitness",
        "algo": "smc_base",
        "T": T,
        "N": N_PARTICLES,
        "repeats": N_RUNS,
        "backbones": list(BACKBONE_ORDER),
        "budgets": list(TABLE_BUDGETS),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"required artifact is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return payload


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _finite_number(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} keys mismatch; missing={missing}, extra={extra}")


def _validate_header(
    payload: Mapping[str, Any],
    path: Path,
    *,
    kind: str,
    backbone: str,
    budget: int,
) -> None:
    if payload.get("format") != 1:
        raise ValueError(f"{path}: format must be 1")
    if payload.get("kind") != kind:
        raise ValueError(f"{path}: kind must be {kind!r}")
    expected = table_config(backbone, budget)
    if kind == "protein_fitness_timing":
        expected = {**expected, **TIMING_CONFIG_EXTRAS}
    actual = payload.get("config")
    if actual != expected:
        raise ValueError(
            f"{path}: config metadata mismatch for {backbone} T'={budget}; "
            f"expected {expected!r}, got {actual!r}"
        )


def _validate_metric_artifact(
    payload: Mapping[str, Any], path: Path, backbone: str, budget: int
) -> dict[str, list[int]]:
    _validate_header(
        payload,
        path,
        kind="protein_fitness_metric",
        backbone=backbone,
        budget=budget,
    )
    policies = _mapping(payload.get("policies"), f"{path}: policies")
    expected_policies = _expected_policies(budget)
    _exact_keys(policies, set(expected_policies), f"{path}: policies")

    expected_config = table_config(backbone, budget)
    base_seed = expected_config["evaluation_seed"]
    expected_seeds = list(range(base_seed, base_seed + N_RUNS))
    values: dict[str, list[int]] = {}
    for policy in expected_policies:
        location = f"{path}: policies.{policy}"
        row = _mapping(policies[policy], location)

        guided_set = row.get("guided_set")
        if not isinstance(guided_set, list):
            raise ValueError(f"{location}.guided_set must be a list")
        if (
            len(guided_set) != budget
            or any(isinstance(step, bool) or not isinstance(step, int) for step in guided_set)
            or guided_set != sorted(set(guided_set))
            or any(step < 0 or step >= T for step in guided_set)
        ):
            raise ValueError(
                f"{location}.guided_set must contain {budget} sorted, unique steps "
                f"in [0, {T})"
            )
        _finite_number(row.get("shat"), f"{location}.shat")

        runs = row.get("runs")
        if not isinstance(runs, list) or len(runs) != N_RUNS:
            actual = len(runs) if isinstance(runs, list) else "not a list"
            raise ValueError(
                f"{location}.runs must contain exactly {N_RUNS} runs; got {actual}"
            )
        seeds: list[int] = []
        unique_values: list[int] = []
        for run_index, run_value in enumerate(runs):
            run_location = f"{location}.runs[{run_index}]"
            run = _mapping(run_value, run_location)
            seed = run.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"{run_location}.seed must be an integer")
            seeds.append(seed)
            _finite_number(
                run.get("elapsed_seconds"),
                f"{run_location}.elapsed_seconds",
                minimum=0.0,
            )

            combos = run.get("combos")
            if (
                not isinstance(combos, list)
                or len(combos) != N_PARTICLES
                or any(not isinstance(combo, str) for combo in combos)
            ):
                raise ValueError(
                    f"{run_location}.combos must contain {N_PARTICLES} strings"
                )
            rewards = run.get("rewards")
            if not isinstance(rewards, list) or len(rewards) != N_PARTICLES:
                raise ValueError(
                    f"{run_location}.rewards must contain {N_PARTICLES} values"
                )
            for reward_index, reward in enumerate(rewards):
                _finite_number(reward, f"{run_location}.rewards[{reward_index}]")

            unique_valid = run.get("unique_valid")
            if (
                isinstance(unique_valid, bool)
                or not isinstance(unique_valid, int)
                or not 0 <= unique_valid <= N_PARTICLES
            ):
                raise ValueError(
                    f"{run_location}.unique_valid must be an integer in "
                    f"[0, {N_PARTICLES}]"
                )
            unique_values.append(unique_valid)
        if seeds != expected_seeds:
            raise ValueError(
                f"{location}.runs seeds mismatch; expected {expected_seeds}, got {seeds}"
            )
        values[policy] = unique_values
    return values


def _validate_timing_artifact(
    payload: Mapping[str, Any], path: Path, backbone: str, budget: int
) -> dict[str, list[float]]:
    _validate_header(
        payload,
        path,
        kind="protein_fitness_timing",
        backbone=backbone,
        budget=budget,
    )
    raw_times = _mapping(payload.get("times"), f"{path}: times")
    expected_policies = _expected_policies(budget)
    _exact_keys(raw_times, set(expected_policies), f"{path}: times")
    times: dict[str, list[float]] = {}
    for policy in expected_policies:
        values = raw_times[policy]
        location = f"{path}: times.{policy}"
        if not isinstance(values, list) or len(values) != N_RUNS:
            actual = len(values) if isinstance(values, list) else "not a list"
            raise ValueError(
                f"{location} must contain exactly {N_RUNS} values; got {actual}"
            )
        times[policy] = [
            _finite_number(value, f"{location}[{index}]", minimum=0.0)
            for index, value in enumerate(values)
        ]
    return times


def aggregate_outputs(output_root: str | Path) -> dict[str, Any]:
    """Aggregate all required raw artifacts beneath *output_root*.

    ``unique_valid_sd`` is the sample standard deviation (``n - 1``), while
    ``time_seconds`` is the arithmetic mean of the dedicated timing artifact.
    """

    output_root = Path(output_root)
    tables: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for backbone in BACKBONE_ORDER:
        backbone_table: dict[str, dict[str, dict[str, float]]] = {}
        for budget in TABLE_BUDGETS:
            metric_path = output_root / "metrics" / f"{backbone}_b{budget}.json"
            timing_path = output_root / "timing" / f"{backbone}_b{budget}.json"
            metric_values = _validate_metric_artifact(
                _read_json(metric_path), metric_path, backbone, budget
            )
            timing_values = _validate_timing_artifact(
                _read_json(timing_path), timing_path, backbone, budget
            )
            policy_table: dict[str, dict[str, float]] = {}
            for policy in _expected_policies(budget):
                unique = metric_values[policy]
                policy_table[policy] = {
                    "time_seconds": statistics.mean(timing_values[policy]),
                    "unique_valid_mean": statistics.mean(unique),
                    "unique_valid_sd": statistics.stdev(unique),
                }
            backbone_table[str(budget)] = policy_table
        tables[backbone] = backbone_table
    return {
        "format": 1,
        "kind": "protein_fitness_aggregate_tables",
        "metadata": _table_metadata(),
        "tables": tables,
    }


def _validate_complete_tables(payload: Mapping[str, Any], source: str) -> None:
    if payload.get("format") != 1:
        raise ValueError(f"{source}: format must be 1")
    if payload.get("metadata") != _table_metadata():
        raise ValueError(
            f"{source}: metadata mismatch; expected {_table_metadata()!r}, "
            f"got {payload.get('metadata')!r}"
        )
    tables = _mapping(payload.get("tables"), f"{source}: tables")
    _exact_keys(tables, set(BACKBONE_ORDER), f"{source}: tables")
    for backbone in BACKBONE_ORDER:
        backbone_table = _mapping(tables[backbone], f"{source}: tables.{backbone}")
        expected_budget_keys = {str(budget) for budget in TABLE_BUDGETS}
        _exact_keys(
            backbone_table, expected_budget_keys, f"{source}: tables.{backbone}"
        )
        for budget in TABLE_BUDGETS:
            budget_table = _mapping(
                backbone_table[str(budget)],
                f"{source}: tables.{backbone}.{budget}",
            )
            expected_policies = _expected_policies(budget)
            _exact_keys(
                budget_table,
                set(expected_policies),
                f"{source}: tables.{backbone}.{budget}",
            )
            for policy in expected_policies:
                location = f"{source}: tables.{backbone}.{budget}.{policy}"
                cell = _mapping(budget_table[policy], location)
                _exact_keys(
                    cell,
                    {"time_seconds", "unique_valid_mean", "unique_valid_sd"},
                    location,
                )
                _finite_number(cell["time_seconds"], f"{location}.time_seconds", minimum=0.0)
                mean = _finite_number(
                    cell["unique_valid_mean"],
                    f"{location}.unique_valid_mean",
                    minimum=0.0,
                )
                if mean > N_PARTICLES:
                    raise ValueError(
                        f"{location}.unique_valid_mean must be <= {N_PARTICLES}"
                    )
                _finite_number(
                    cell["unique_valid_sd"],
                    f"{location}.unique_valid_sd",
                    minimum=0.0,
                )


def load_reference(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate ``results/reference_tables.json``."""

    path = Path(path)
    raw_payload = dict(_read_json(path))
    if raw_payload.get("format") != 1:
        raise ValueError(f"{path}: format must be 1")
    if raw_payload.get("kind") != "protein_fitness_reference_tables":
        raise ValueError(
            f"{path}: kind must be 'protein_fitness_reference_tables'"
        )
    if raw_payload.get("metadata") != _table_metadata():
        raise ValueError(
            f"{path}: metadata mismatch; expected {_table_metadata()!r}, "
            f"got {raw_payload.get('metadata')!r}"
        )
    raw_tables = _mapping(raw_payload.get("tables"), f"{path}: tables")
    _exact_keys(raw_tables, set(BACKBONE_ORDER), f"{path}: tables")

    tables: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    expected_cells = {
        (budget, policy)
        for budget in TABLE_BUDGETS
        for policy in _expected_policies(budget)
    }
    for backbone in BACKBONE_ORDER:
        raw_table = _mapping(raw_tables[backbone], f"{path}: tables.{backbone}")
        _exact_keys(
            raw_table,
            {"display_name", "rows"},
            f"{path}: tables.{backbone}",
        )
        expected_display_name = BACKBONES[backbone]["display_name"]
        if raw_table["display_name"] != expected_display_name:
            raise ValueError(
                f"{path}: tables.{backbone}.display_name must be "
                f"{expected_display_name!r}"
            )
        rows = raw_table["rows"]
        if not isinstance(rows, list):
            raise ValueError(f"{path}: tables.{backbone}.rows must be a list")
        backbone_table: dict[str, dict[str, dict[str, float]]] = {
            str(budget): {} for budget in TABLE_BUDGETS
        }
        seen: set[tuple[int, str]] = set()
        for index, row_value in enumerate(rows):
            location = f"{path}: tables.{backbone}.rows[{index}]"
            row = _mapping(row_value, location)
            _exact_keys(
                row,
                {
                    "budget",
                    "policy",
                    "time_mean",
                    "unique_valid_mean",
                    "unique_valid_sd",
                },
                location,
            )
            budget = row["budget"]
            policy = row["policy"]
            if (
                isinstance(budget, bool)
                or not isinstance(budget, int)
                or not isinstance(policy, str)
            ):
                raise ValueError(f"{location}: budget/policy types are invalid")
            identity = (budget, policy)
            if identity not in expected_cells:
                raise ValueError(f"{location}: unexpected table cell {identity!r}")
            if identity in seen:
                raise ValueError(f"{location}: duplicate table cell {identity!r}")
            seen.add(identity)
            backbone_table[str(budget)][policy] = {
                "time_seconds": _finite_number(
                    row["time_mean"], f"{location}.time_mean", minimum=0.0
                ),
                "unique_valid_mean": _finite_number(
                    row["unique_valid_mean"],
                    f"{location}.unique_valid_mean",
                    minimum=0.0,
                ),
                "unique_valid_sd": _finite_number(
                    row["unique_valid_sd"],
                    f"{location}.unique_valid_sd",
                    minimum=0.0,
                ),
            }
        if seen != expected_cells:
            raise ValueError(
                f"{path}: tables.{backbone}.rows missing required cells: "
                f"{sorted(expected_cells - seen)!r}"
            )
        tables[backbone] = backbone_table

    payload = {
        "format": 1,
        "kind": "protein_fitness_reference_tables",
        "metadata": _table_metadata(),
        "tables": tables,
    }
    _validate_complete_tables(payload, str(path))
    return payload


def _table_rows(payload: Mapping[str, Any], backbone: str):
    table = payload["tables"][backbone]
    for budget in TABLE_BUDGETS:
        for policy in _expected_policies(budget):
            yield budget, policy, table[str(budget)][policy]


def render_markdown(payload: Mapping[str, Any], backbone: str) -> str:
    """Return one paper-style Markdown table."""

    if backbone not in BACKBONE_ORDER:
        raise ValueError(f"unknown backbone: {backbone}")
    _validate_complete_tables(payload, "table payload")
    display_backbone = BACKBONES[backbone]["display_name"]
    lines = [
        f"**Table {TABLE_NUMBERS[backbone]}: Protein Fitness, "
        f"{display_backbone}, SMC-base** "
        f"({N_PARTICLES} particles $\\times$ {N_RUNS} runs)",
        "",
        "| T' | Policy | Time [s] | Unique Valid |",
        "|---:|:---|---:|---:|",
    ]
    budget_best = {
        budget: max(
            payload["tables"][backbone][str(budget)][policy]["unique_valid_mean"]
            for policy in _expected_policies(budget)
        )
        for budget in SPARSE_T_PRIMES
    }
    for budget, policy, cell in _table_rows(payload, backbone):
        unique = f'{cell["unique_valid_mean"]:.2f} ± {cell["unique_valid_sd"]:.2f}'
        if budget in budget_best and cell["unique_valid_mean"] == budget_best[budget]:
            unique = f"**{unique}**"
        lines.append(
            f"| {budget} | {MARKDOWN_POLICY_NAMES[policy]} | "
            f'{cell["time_seconds"]:.2f} | {unique} |'
        )
    return "\n".join(lines) + "\n"


def render_csv(payload: Mapping[str, Any], backbone: str) -> str:
    """Return one CSV table, with the mean and sample SD in one metric cell."""

    if backbone not in BACKBONE_ORDER:
        raise ValueError(f"unknown backbone: {backbone}")
    _validate_complete_tables(payload, "table payload")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["T'", "Policy", "Time [s]", "Unique Valid"])
    for budget, policy, cell in _table_rows(payload, backbone):
        writer.writerow(
            [
                budget,
                CSV_POLICY_NAMES[policy],
                f'{cell["time_seconds"]:.2f}',
                f'{cell["unique_valid_mean"]:.2f} ± {cell["unique_valid_sd"]:.2f}',
            ]
        )
    return output.getvalue()


def write_tables(payload: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, ...]:
    """Write ``table_e1.md/csv`` and ``table_e2.md/csv``."""

    _validate_complete_tables(payload, "table payload")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for backbone in BACKBONE_ORDER:
        stem = "table_" + TABLE_NUMBERS[backbone].lower().replace("-", "")
        markdown_path = output_dir / f"{stem}.md"
        csv_path = output_dir / f"{stem}.csv"
        markdown_path.write_text(render_markdown(payload, backbone), encoding="utf-8")
        csv_path.write_text(render_csv(payload, backbone), encoding="utf-8")
        written.extend((markdown_path, csv_path))
    return tuple(written)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root containing metrics/ and timing/ (default: outputs).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tables",
        help="Directory for table_e1/e2 Markdown and CSV files.",
    )
    parser.add_argument(
        "--reference",
        nargs="?",
        const="results/reference_tables.json",
        help=(
            "Render a validated reference JSON instead of aggregating raw outputs. "
            "With no path, uses results/reference_tables.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = (
        load_reference(args.reference)
        if args.reference is not None
        else aggregate_outputs(args.output_root)
    )
    paths = write_tables(payload, args.output_dir)
    print(render_markdown(payload, "mdlm"))
    print(render_markdown(payload, "udlm"), end="")
    print("Wrote: " + ", ".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
