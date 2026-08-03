"""CUDA-synchronized timing benchmark for the reported Protein Fitness rows."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.chdir(PACKAGE_ROOT)

from util.seed import set_seed

try:
    from .experiment_spec import BACKBONES, N_PARTICLES, N_RUNS, T_PRIMES, table_config
    from .run_protein_fitness import (
        atomic_json_dump, checkpoint_path, guided_sets, make_runtime, sha256_file,
        warmup_metadata,
    )
except ImportError:
    from experiment_spec import BACKBONES, N_PARTICLES, N_RUNS, T_PRIMES, table_config
    from run_protein_fitness import (
        atomic_json_dump, checkpoint_path, guided_sets, make_runtime, sha256_file,
        warmup_metadata,
    )


BENCHMARK_WARMUPS = 1
PRIMING_SEED = 90000
ORDER_SEED = 20260802


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_vhat(path, device, backbone, checkpoint_sha256):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("metadata") != warmup_metadata(backbone, checkpoint_sha256):
        raise ValueError(f"incompatible Vhat cache: {path}")
    vhat = payload.get("Vhat")
    if not isinstance(vhat, torch.Tensor) or tuple(vhat.shape) != (129,):
        raise ValueError("Vhat cache must contain a length-129 tensor")
    return vhat.to(device)


def run_inference(algo, guided_set, seed, device, measured):
    set_seed(seed)
    synchronize(device)
    start = time.perf_counter()
    algo.inference(
        num_samples=N_PARTICLES,
        verbose=False,
        detokenize=True,
        guided_set=set(guided_set),
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed if measured else None


def benchmark_config(backbone, budget):
    config = table_config(backbone, budget)
    config.update({
        "benchmark_warmups": BENCHMARK_WARMUPS,
        "priming_seed": PRIMING_SEED,
        "order_seed": ORDER_SEED,
        "timed_scope": "SMC_Base.inference(detokenize=True), CUDA synchronized",
    })
    return config


def new_artifact(backbone, budget, checkpoint_sha256, schedules, device):
    return {
        "format": 1,
        "kind": "protein_fitness_timing",
        "config": benchmark_config(backbone, budget),
        "checkpoint_sha256": checkpoint_sha256,
        "runtime": {
            "device_type": device.type,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "guided_sets": {
            name: value["guided_set"] for name, value in schedules.items()
        },
        "times": {name: [] for name in schedules},
    }


def validate_resume(payload, backbone, budget, checkpoint_sha256, schedules):
    if payload.get("format") != 1 or payload.get("kind") != "protein_fitness_timing":
        raise ValueError("unsupported timing artifact")
    if payload.get("config") != benchmark_config(backbone, budget):
        raise ValueError("existing timing artifact has incompatible configuration")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("existing timing artifact used a different checkpoint")
    expected = {name: row["guided_set"] for name, row in schedules.items()}
    if payload.get("guided_sets") != expected or set(payload.get("times", {})) != set(expected):
        raise ValueError("existing timing artifact has incompatible schedules")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--budget", type=int, choices=T_PRIMES, required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--vhat-path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("controlled timing requires an NVIDIA CUDA GPU")
    device = torch.device("cuda")
    checkpoint_sha256 = sha256_file(checkpoint_path(args.backbone))
    algo, _ = make_runtime(args.backbone, device)
    vhat_path = args.vhat_path or str(
        Path(args.output_root) / "vhat" / f"{args.backbone}.pt")
    vhat = load_vhat(vhat_path, device, args.backbone, checkpoint_sha256)
    schedules = guided_sets(args.backbone, args.budget, vhat)

    output = Path(args.output_root) / "timing" / f"{args.backbone}_b{args.budget}.json"
    if output.exists():
        with open(output, encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_resume(payload, args.backbone, args.budget, checkpoint_sha256, schedules)
        print(f"[resume] {output}", flush=True)
    else:
        payload = new_artifact(
            args.backbone, args.budget, checkpoint_sha256, schedules, device)
        atomic_json_dump(payload, output)

    names = list(schedules)
    for warmup_index in range(BENCHMARK_WARMUPS):
        for policy_index, name in enumerate(names):
            run_inference(
                algo,
                schedules[name]["guided_set"],
                PRIMING_SEED + warmup_index * len(names) + policy_index,
                device,
                measured=False,
            )
            print(f"[prime] {args.backbone} T'={args.budget} {name}", flush=True)

    order_rng = np.random.RandomState(ORDER_SEED + args.budget)
    orders = [order_rng.permutation(names).tolist() for _ in range(N_RUNS)]
    evaluation_seed = BACKBONES[args.backbone]["evaluation_seed"]
    for repeat_index, order in enumerate(orders):
        for name in order:
            completed = len(payload["times"][name])
            if completed > repeat_index:
                continue
            if completed != repeat_index:
                raise ValueError(f"non-contiguous timing state for {name}")
            elapsed = run_inference(
                algo,
                schedules[name]["guided_set"],
                evaluation_seed + repeat_index,
                device,
                measured=True,
            )
            payload["times"][name].append(elapsed)
            atomic_json_dump(payload, output)
            print(
                f"[timed] {args.backbone} T'={args.budget} {name} "
                f"run={repeat_index + 1}/{N_RUNS} elapsed={elapsed:.4f}s",
                flush=True,
            )
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
