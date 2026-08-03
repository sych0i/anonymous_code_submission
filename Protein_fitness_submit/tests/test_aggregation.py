import csv
import io
import json
import sys
import statistics
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import aggregate_tables
from scripts.experiment_spec import (
    BACKBONES,
    N_PARTICLES,
    N_RUNS,
    POLICIES,
    SPARSE_T_PRIMES,
    T,
    table_config,
)


BACKBONES_IN_ORDER = ("mdlm", "udlm")
BUDGETS = (T, *SPARSE_T_PRIMES)


def policies_for(budget):
    return ("full",) if budget == T else POLICIES


def table_metadata():
    return {
        "experiment": "protein_fitness",
        "algo": "smc_base",
        "T": T,
        "N": N_PARTICLES,
        "repeats": N_RUNS,
        "backbones": list(BACKBONES_IN_ORDER),
        "budgets": list(BUDGETS),
    }


def reference_payload():
    tables = {}
    for backbone_index, backbone in enumerate(BACKBONES_IN_ORDER):
        rows = []
        for budget in BUDGETS:
            for policy_index, policy in enumerate(policies_for(budget)):
                rows.append(
                    {
                        "budget": budget,
                        "policy": policy,
                        "time_mean": budget + policy_index / 10,
                        "unique_valid_mean": backbone_index + policy_index + 1.0,
                        "unique_valid_sd": policy_index / 4 + 0.5,
                    }
                )
        tables[backbone] = {
            "display_name": BACKBONES[backbone]["display_name"],
            "rows": rows,
        }
    return {
        "format": 1,
        "kind": "protein_fitness_reference_tables",
        "metadata": table_metadata(),
        "tables": tables,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_raw_outputs(root):
    for backbone in BACKBONES_IN_ORDER:
        for budget in BUDGETS:
            config = table_config(backbone, budget)
            metric_policies = {}
            timing_policies = {}
            for policy_index, policy in enumerate(policies_for(budget)):
                runs = []
                for offset in range(N_RUNS):
                    runs.append(
                        {
                            "seed": config["evaluation_seed"] + offset,
                            # Deliberately differs from the timing artifact: the
                            # table must use outputs/timing, not this duplicate.
                            "elapsed_seconds": 999.0,
                            "combos": ["A" * 15] * N_PARTICLES,
                            "rewards": [0.25] * N_PARTICLES,
                            "unique_valid": offset + policy_index,
                        }
                    )
                metric_policies[policy] = {
                    "guided_set": list(range(budget)),
                    "shat": float(policy_index),
                    "runs": runs,
                }
                timing_policies[policy] = [
                    policy_index + offset / 10 for offset in range(N_RUNS)
                ]
            metric = {
                "format": 1,
                "kind": "protein_fitness_metric",
                "config": config,
                "policies": metric_policies,
            }
            timing = {
                "format": 1,
                "kind": "protein_fitness_timing",
                "config": {**config, **aggregate_tables.TIMING_CONFIG_EXTRAS},
                "timing_device": "test-gpu",
                "times": timing_policies,
            }
            write_json(root / "metrics" / f"{backbone}_b{budget}.json", metric)
            write_json(root / "timing" / f"{backbone}_b{budget}.json", timing)


def test_load_reference_and_write_paper_tables(tmp_path):
    reference_path = tmp_path / "results" / "reference_tables.json"
    write_json(reference_path, reference_payload())

    payload = aggregate_tables.load_reference(reference_path)
    markdown = aggregate_tables.render_markdown(payload, "mdlm")
    assert "Table E-1: Protein Fitness, MDLM, SMC-base" in markdown
    assert "| 128 | Full | 128.00 | 1.00 ± 0.50 |" in markdown
    assert "| 16 | Top-$\\Delta V$ |" in markdown
    # VISTA has the largest synthetic mean for both sparse budgets.
    assert "**9.00 ± 2.50**" in markdown

    paths = aggregate_tables.write_tables(payload, tmp_path / "rendered")
    assert {path.name for path in paths} == {
        "table_e1.md",
        "table_e1.csv",
        "table_e2.md",
        "table_e2.csv",
    }
    rows = list(
        csv.DictReader(io.StringIO((tmp_path / "rendered" / "table_e2.csv").read_text()))
    )
    assert len(rows) == 1 + 2 * len(POLICIES)
    assert rows[0] == {
        "T'": "128",
        "Policy": "Full",
        "Time [s]": "128.00",
        "Unique Valid": "2.00 ± 0.50",
    }


def test_aggregate_uses_sample_sd_and_dedicated_timing(tmp_path):
    write_raw_outputs(tmp_path)

    payload = aggregate_tables.aggregate_outputs(tmp_path)
    cell = payload["tables"]["mdlm"]["16"]["interval1"]
    assert cell["unique_valid_mean"] == statistics.mean(range(N_RUNS))
    assert cell["unique_valid_sd"] == statistics.stdev(range(N_RUNS))
    assert cell["time_seconds"] == statistics.mean(
        offset / 10 for offset in range(N_RUNS)
    )
    assert cell["time_seconds"] != 999.0


def test_rejects_missing_required_policy(tmp_path):
    write_raw_outputs(tmp_path)
    path = tmp_path / "metrics" / "mdlm_b16.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["policies"]["vista"]
    write_json(path, payload)

    with pytest.raises(ValueError, match=r"policies keys mismatch.*vista"):
        aggregate_tables.aggregate_outputs(tmp_path)


def test_rejects_incompatible_metadata(tmp_path):
    write_raw_outputs(tmp_path)
    path = tmp_path / "timing" / "udlm_b32.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["N"] = 31
    write_json(path, payload)

    with pytest.raises(ValueError, match="config metadata mismatch"):
        aggregate_tables.aggregate_outputs(tmp_path)


def test_reference_rejects_duplicate_and_missing_cell(tmp_path):
    payload = reference_payload()
    payload["tables"]["mdlm"]["rows"][-1] = dict(
        payload["tables"]["mdlm"]["rows"][-2]
    )
    path = tmp_path / "reference_tables.json"
    write_json(path, payload)

    with pytest.raises(ValueError, match="duplicate table cell"):
        aggregate_tables.load_reference(path)
