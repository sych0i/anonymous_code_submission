"""Run the fixed MDLM/UDLM Protein Fitness SMC-base experiment.

The table configuration lives in :mod:`experiment_spec`.  Results are written
incrementally after every seed, so the command can safely be restarted.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.chdir(PACKAGE_ROOT)

from problem.protein_reward import ProteinOracleReward
from sampling.schedules import build_schedule
from sampling.smc import SMC_Base
from sampling.vista import estimate_warmup_values, evaluate_schedule, time_weighted_lambda
from util.diversity_metrics import num_unique_high_reward
from util.seed import set_seed

try:
    from .experiment_spec import (
        BACKBONES, ESS_THRESHOLD, N_PARTICLES, N_ROLLOUTS, N_RUNS,
        N_WARMUP_RUNS, PARTIAL_RESAMPLE, POLICIES, REWARD_THRESHOLD,
        SIMILARITY_THRESHOLD, T, T_PRIMES, WARMUP_SEED, table_config,
    )
except ImportError:
    from experiment_spec import (
        BACKBONES, ESS_THRESHOLD, N_PARTICLES, N_ROLLOUTS, N_RUNS,
        N_WARMUP_RUNS, PARTIAL_RESAMPLE, POLICIES, REWARD_THRESHOLD,
        SIMILARITY_THRESHOLD, T, T_PRIMES, WARMUP_SEED, table_config,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def checkpoint_path(backbone):
    return PACKAGE_ROOT / "checkpoints" / backbone / "TrpB" / "best_model.ckpt"


def warmup_metadata(backbone, checkpoint_sha256):
    model = BACKBONES[backbone]
    return {
        "format": 1,
        "backbone": backbone,
        "algo": "smc_base",
        "checkpoint_sha256": checkpoint_sha256,
        "T": T,
        "N": N_PARTICLES,
        "J": N_ROLLOUTS,
        "M": N_WARMUP_RUNS,
        "alpha": model["alpha"],
        "ess_threshold": ESS_THRESHOLD,
        "partial_resample": PARTIAL_RESAMPLE,
        "warmup_seed": WARMUP_SEED,
    }


def load_or_estimate_vhat(path, algo, backbone, checkpoint_sha256):
    expected = warmup_metadata(backbone, checkpoint_sha256)
    path = Path(path)
    if path.exists():
        payload = torch.load(path, map_location=algo.device, weights_only=False)
        if payload.get("metadata") != expected:
            raise ValueError(f"incompatible Vhat cache: {path}")
        print(f"[warmup] loaded {path}", flush=True)
        return payload["Vhat"].to(algo.device)

    set_seed(WARMUP_SEED)
    start = time.perf_counter()
    vhat = estimate_warmup_values(
        algo, T, N_PARTICLES, num_warmup_runs=N_WARMUP_RUNS, verbose=False)
    elapsed = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save({"metadata": expected, "Vhat": vhat.cpu()}, temporary)
    os.replace(temporary, path)
    print(f"[warmup] generated {path} in {elapsed:.2f}s", flush=True)
    return vhat


def make_runtime(backbone, device):
    data_config = OmegaConf.load(PACKAGE_ROOT / "configs" / "data" / "TrpB.yaml")
    model_config = OmegaConf.load(PACKAGE_ROOT / "configs" / "model" / f"{backbone}.yaml")
    set_seed(42)
    net = instantiate(
        model_config.model,
        model_name=f"{backbone}/TrpB",
        seq_len=data_config.seq_len,
        num_steps=T,
        device=device,
        _recursive_=False,
    )
    oracle = ProteinOracleReward(data_config=data_config, device=device)
    algo = SMC_Base(
        net=net,
        forward_op=oracle,
        data_config=data_config,
        alpha=BACKBONES[backbone]["alpha"],
        num_rollout_samples=N_ROLLOUTS,
        ess_threshold=ESS_THRESHOLD,
        partial_resample=PARTIAL_RESAMPLE,
        device=device,
    )
    return algo, oracle


def guided_sets(backbone, budget, vhat):
    k = BACKBONES[backbone]["vista_k"]
    lam = None if k is None else time_weighted_lambda(T, k)
    result = {}
    policies = ("full",) if budget == T else POLICIES
    for policy in policies:
        policy_lam = lam if policy == "vista" else None
        steps = build_schedule(policy, T, budget, V_hat=vhat, lam=policy_lam)
        if len(steps) != budget:
            raise RuntimeError(
                f"{policy} produced {len(steps)} steps for T'={budget}, expected {budget}")
        result[policy] = {
            "guided_set": sorted(int(step) for step in steps),
            "shat": float(evaluate_schedule(vhat, T, steps, lam=lam)),
            "runs": [],
        }
    return result


def new_metric_artifact(backbone, budget, checkpoint_sha256, vhat, device):
    return {
        "format": 1,
        "kind": "protein_fitness_metric",
        "config": table_config(backbone, budget),
        "checkpoint_sha256": checkpoint_sha256,
        "vhat": [float(value) for value in vhat.detach().cpu()],
        "runtime": {
            "device_type": device.type,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "policies": guided_sets(backbone, budget, vhat),
    }


def validate_resume(payload, backbone, budget, checkpoint_sha256):
    if payload.get("format") != 1 or payload.get("kind") != "protein_fitness_metric":
        raise ValueError("unsupported metric artifact")
    if payload.get("config") != table_config(backbone, budget):
        raise ValueError("existing metric artifact has incompatible configuration")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("existing metric artifact used a different checkpoint")
    expected_policies = set(table_config(backbone, budget)["policies"])
    if set(payload.get("policies", {})) != expected_policies:
        raise ValueError("existing metric artifact has an incompatible policy set")


def run_seed(algo, oracle, guided_set, seed, device):
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    _, sequences = algo.inference(
        num_samples=N_PARTICLES,
        verbose=False,
        detokenize=True,
        guided_set=set(guided_set),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    combos = ["".join(sequence[index] for index in algo.residues) for sequence in sequences]
    rewards = oracle(combos).detach().cpu().numpy()
    unique_valid = num_unique_high_reward(
        combos,
        rewards,
        reward_threshold=REWARD_THRESHOLD,
        sim_threshold=SIMILARITY_THRESHOLD,
    )
    return {
        "seed": seed,
        "elapsed_seconds": elapsed,
        "combos": combos,
        "rewards": [float(value) for value in rewards],
        "unique_valid": int(unique_valid),
    }


def run_budget(backbone, budget, algo, oracle, vhat, checkpoint_sha256, output_root, device):
    path = Path(output_root) / "metrics" / f"{backbone}_b{budget}.json"
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_resume(payload, backbone, budget, checkpoint_sha256)
        print(f"[resume] {path}", flush=True)
    else:
        payload = new_metric_artifact(backbone, budget, checkpoint_sha256, vhat, device)
        atomic_json_dump(payload, path)

    base_seed = BACKBONES[backbone]["evaluation_seed"]
    for policy in table_config(backbone, budget)["policies"]:
        row = payload["policies"][policy]
        runs = row["runs"]
        if len(runs) > N_RUNS:
            raise ValueError(f"too many completed runs for {policy}")
        expected_completed = list(range(base_seed, base_seed + len(runs)))
        if [item["seed"] for item in runs] != expected_completed:
            raise ValueError(f"non-contiguous resume state for {policy}")
        for offset in range(len(runs), N_RUNS):
            seed = base_seed + offset
            result = run_seed(algo, oracle, row["guided_set"], seed, device)
            row["runs"].append(result)
            atomic_json_dump(payload, path)
            print(
                f"[{backbone} T'={budget} {policy}] "
                f"run={offset + 1}/{N_RUNS} seed={seed} "
                f"unique_valid={result['unique_valid']} "
                f"elapsed={result['elapsed_seconds']:.3f}s",
                flush=True,
            )
    print(f"[done] {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--budgets", type=int, nargs="+", choices=T_PRIMES, default=list(T_PRIMES))
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--vhat-path")
    parser.add_argument("--warmup-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the table-scale reproduction requires an NVIDIA CUDA GPU")
    device = torch.device("cuda")
    checkpoint = checkpoint_path(args.backbone)
    checkpoint_sha256 = sha256_file(checkpoint)
    algo, oracle = make_runtime(args.backbone, device)
    vhat_path = args.vhat_path or str(
        Path(args.output_root) / "vhat" / f"{args.backbone}.pt")
    vhat = load_or_estimate_vhat(
        vhat_path, algo, args.backbone, checkpoint_sha256)
    if args.warmup_only:
        return
    for budget in args.budgets:
        run_budget(
            args.backbone,
            budget,
            algo,
            oracle,
            vhat,
            checkpoint_sha256,
            args.output_root,
            device,
        )


if __name__ == "__main__":
    main()
