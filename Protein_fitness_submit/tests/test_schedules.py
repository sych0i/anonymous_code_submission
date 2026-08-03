import json
import math
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from sampling.schedules import build_schedule
from sampling.vista import evaluate_schedule, time_weighted_lambda


REFERENCE = PACKAGE_ROOT / "results" / "reference_schedules.json"


def test_all_reference_schedules_and_scores():
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert payload["format"] == 1
    assert payload["kind"] == "protein_fitness_schedule_regression"
    T = payload["metadata"]["T"]
    for backbone, model in payload["backbones"].items():
        vhat = model["vhat"]
        lam = None if model["vista_k"] is None else time_weighted_lambda(T, model["vista_k"])
        for budget_text, policies in model["budgets"].items():
            budget = int(budget_text)
            for policy, expected in policies.items():
                policy_lam = lam if policy == "vista" else None
                actual = build_schedule(
                    policy, T, budget, V_hat=vhat, lam=policy_lam)
                assert sorted(actual) == expected["guided_set"], (
                    backbone, budget, policy)
                score = evaluate_schedule(vhat, T, actual, lam=lam)
                assert math.isclose(score, expected["shat"], rel_tol=1e-10, abs_tol=1e-7)


def test_full_schedule_contains_every_reverse_step():
    assert build_schedule("full", 128, 128) == set(range(128))
