"""Run paper-indexed VISTA/TDS schedule experiments on pretrained MNIST models.

This is an inference-only runner.  It keeps the DDPM respacing fixed at ``T``
and changes only the set of reverse transitions at which SMC guidance is
active.  Three full-guidance runs are used by default to estimate
``[V_hat_0, ..., V_hat_T]``; those warmup samples are stored separately and are
never included in policy metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torchvision.utils import make_grid


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:  # Package import, e.g. ``from image_exp import run_vista``.
    from .image_diffusion import dist_util
    from .image_diffusion.resnet import get_model as get_resnet_model
    from .image_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
        select_args,
    )
    from .image_diffusion.vista_smc import sample_vista_smc
    from .vista_metrics import compute_vista_metrics
    from .vista_schedules import (
        POLICIES,
        build_schedule_map,
        empirical_utd,
        execution_order,
        normalize_policy_name,
        policy_weight_power,
        schedule_objective,
        timestep_weights,
    )
except ImportError:  # Direct execution: ``python image_exp/run_vista.py``.
    from image_diffusion import dist_util
    from image_diffusion.resnet import get_model as get_resnet_model
    from image_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
        select_args,
    )
    from image_diffusion.vista_smc import sample_vista_smc
    from vista_metrics import compute_vista_metrics
    from vista_schedules import (
        POLICIES,
        build_schedule_map,
        empirical_utd,
        execution_order,
        normalize_policy_name,
        policy_weight_power,
        schedule_objective,
        timestep_weights,
    )


RUNNER_SCHEMA_VERSION = 12
RAW_EVALUATION_SCHEMA_VERSION = 3
DEFAULT_MODEL_CONFIG = (
    SCRIPT_DIR / "image_confs" / "mnist_model_and_diffusion_conf.yml"
)
DEFAULT_TASK_CONFIG = SCRIPT_DIR / "image_confs" / "task_class_cond_gen_conf.yml"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "vista"
Scalar = int | float


@dataclass
class RuntimeComponents:
    diffusion: Any
    model: Callable[..., torch.Tensor]
    classifier: Callable[[torch.Tensor], torch.Tensor]
    model_kwargs: Mapping[str, Any]
    metadata: Mapping[str, Any]


class ScaledMNISTClassifier(nn.Module):
    """Wrap the checkpoint classifier's expected [0, 1] input scaling."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.network((samples + 1.0) / 2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full and sparse VISTA/TDS guidance schedules with the "
            "pretrained unconditional MNIST DDPM and classifier."
        )
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
        help="MNIST model/diffusion YAML.",
    )
    parser.add_argument(
        "--task-config",
        default=str(DEFAULT_TASK_CONFIG),
        help="Class-conditional task YAML containing the classifier checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory. A configuration-hashed experiment directory is used.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit experiment directory name.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed per-seed artifacts when the manifest matches.",
    )
    parser.add_argument(
        "--save-grids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a PNG grid for every warmup and evaluation run.",
    )
    parser.add_argument(
        "--plot-vhat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Plot all warmup value curves and their mean to "
            "warmup/value_curve.png."
        ),
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Estimate/save the warmup value curve, then skip all policy runs.",
    )
    parser.add_argument(
        "--warmup-from",
        default=None,
        metavar="PATH",
        help=(
            "Reuse value_means from an existing warmup summary.json (or an "
            "experiment summary.json containing a warmup section)."
        ),
    )
    parser.add_argument(
        "--full-baseline-only",
        action="store_true",
        help=(
            "Run only the full-guidance evaluation baseline and omit bootstrap "
            "warmup sampling. The saved per-seed value_means can later be "
            "reused to construct V-hat without sampling those seeds twice."
        ),
    )

    parser.add_argument("--target-class", "--target", type=int, default=4)
    parser.add_argument("--total-steps", "--T", dest="total_steps", type=int, default=100)
    parser.add_argument(
        "--guidance-steps",
        "--T-prime",
        dest="guidance_steps",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--num-particles", "--N", dest="num_particles", type=int, default=20
    )
    parser.add_argument(
        "--num-rollouts",
        "--J",
        dest="num_rollouts",
        type=int,
        default=1,
        help=(
            "Clean-state samples used by value estimation. Continuous "
            "one_shot is the deterministic pred_xstart delta and requires "
            "J=1. one_shot_gaussian draws J>=3 direct clean candidates "
            "without an ancestral reverse chain."
        ),
    )
    parser.add_argument(
        "--warmup-runs", "--M", dest="warmup_runs", type=int, default=3
    )
    parser.add_argument(
        "--eval-runs",
        type=int,
        default=50,
        help="Independent evaluation seeds per policy.",
    )
    parser.add_argument(
        "--eval-start-index",
        type=int,
        default=0,
        help=(
            "Global index of the first evaluation run. Use disjoint ranges to "
            "shard an experiment; seed = --seed + global run index."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup-seed-offset",
        type=int,
        default=100_000,
        help="Warmup seed i is seed + this offset + i.",
    )

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ess-threshold", type=float, default=0.95)
    parser.add_argument(
        "--partial-resample",
        type=int,
        default=None,
        help="Particles in adaptive partial resampling; default N//2.",
    )
    parser.add_argument(
        "--final-transition",
        choices=("stochastic", "legacy_deterministic"),
        default="legacy_deterministic",
        help=(
            "Use the deterministic DDPM x_1 -> x_0 transition by default. "
            "'stochastic' is retained for explicit ablations."
        ),
    )
    parser.add_argument(
        "--value-estimation",
        choices=("one_shot", "one_shot_gaussian", "ancestral"),
        default="one_shot",
        help=(
            "'one_shot' mirrors the paper's direct x0 prediction using this "
            "DDPM's pred_xstart as a delta approximation and is the default. "
            "'one_shot_gaussian' draws J>=3 direct x0 candidates around "
            "pred_xstart with a unit-prior Gaussian posterior variance. "
            "'ancestral' runs full reverse chains and is retained only as a "
            "diagnostic ablation. The q proposal uses the deterministic TDS "
            "classifier potential in both modes."
        ),
    )
    parser.add_argument(
        "--clip-denoised",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--clip-twisted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Clip the TDS-shifted pred_xstart to [-1,1] before recomputing "
            "the proposal mean, matching the upstream implementation."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "'auto' uses the first visible CUDA device, then CPU. Select an "
            "unused physical GPU with CUDA_VISIBLE_DEVICES before launching."
        ),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(POLICIES),
        help=(
            "Policies separated by spaces and/or commas. Canonical policies: "
            + ", ".join(POLICIES)
        ),
    )

    parser.add_argument("--valid-logprob-threshold", type=float, default=-0.1)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    return parser


def parse_policy_list(raw_policies: str | Sequence[str]) -> list[str]:
    """Parse comma/space-separated aliases into unique canonical policy names."""

    if isinstance(raw_policies, str):
        raw_items: Sequence[str] = [raw_policies]
    else:
        raw_items = raw_policies
    tokens: list[str] = []
    for raw_item in raw_items:
        tokens.extend(piece.strip() for piece in str(raw_item).split(","))
    tokens = [token for token in tokens if token]
    if not tokens:
        raise ValueError("at least one policy must be selected")

    policies: list[str] = []
    for token in tokens:
        canonical = normalize_policy_name(token)
        if canonical in policies:
            raise ValueError(
                f"duplicate policy after alias normalization: {canonical}"
            )
        policies.append(canonical)
    return policies


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate and normalize parsed arguments in place."""

    if not 0 <= args.target_class < 10:
        raise ValueError("target_class must lie in [0, 9]")
    if args.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 1 <= args.guidance_steps <= args.total_steps:
        raise ValueError("guidance_steps must satisfy 1 <= T' <= T")
    for field in ("num_particles", "num_rollouts", "warmup_runs", "eval_runs"):
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.value_estimation == "one_shot" and args.num_rollouts != 1:
        raise ValueError(
            "continuous one_shot exposes only one pred_xstart point estimate; "
            "pass --J 1"
        )
    if (
        args.value_estimation == "one_shot_gaussian"
        and args.num_rollouts < 3
    ):
        raise ValueError(
            "continuous one_shot_gaussian requires at least three direct "
            "clean-state samples; pass --J 3 or larger"
        )
    if args.eval_start_index < 0:
        raise ValueError("eval_start_index must be non-negative")
    if args.alpha <= 0 or not math.isfinite(args.alpha):
        raise ValueError("alpha must be positive and finite")
    if not 0.0 <= args.ess_threshold <= 1.0:
        raise ValueError("ess_threshold must lie in [0, 1]")
    if args.partial_resample is None:
        args.partial_resample = args.num_particles // 2
    if not 0 <= args.partial_resample <= args.num_particles:
        raise ValueError("partial_resample must lie in [0, num_particles]")
    if args.classifier_batch_size <= 0:
        raise ValueError("classifier_batch_size must be positive")
    for field in ("foreground_threshold", "duplicate_threshold"):
        value = float(getattr(args, field))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be finite and lie in [0, 1]")
    if math.isnan(float(args.valid_logprob_threshold)):
        raise ValueError("valid_logprob_threshold must not be NaN")
    if args.warmup_from is not None:
        warmup_source = Path(args.warmup_from).expanduser()
        if not warmup_source.is_file():
            raise FileNotFoundError(
                f"warmup source summary does not exist: {warmup_source}"
            )
        args.warmup_from = str(warmup_source.resolve())

    args.policies = parse_policy_list(args.policies)
    if args.full_baseline_only:
        if args.policies != ["full"]:
            raise ValueError(
                "full_baseline_only requires exactly --policies full"
            )
        if args.warmup_only:
            raise ValueError(
                "full_baseline_only cannot be combined with --warmup-only"
            )
        if args.warmup_from is not None:
            raise ValueError(
                "full_baseline_only cannot be combined with --warmup-from"
            )
        if args.eval_runs < args.warmup_runs:
            raise ValueError(
                "full_baseline_only needs eval_runs >= warmup_runs so its first "
                "full-step runs can estimate V-hat"
            )
    if args.run_name is not None:
        if (
            args.run_name in ("", ".", "..")
            or Path(args.run_name).name != args.run_name
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name)
        ):
            raise ValueError(
                "run_name must be a single safe path component containing only "
                "letters, digits, '.', '_' or '-'"
            )
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {requested!r} requested but unavailable")
        index = 0 if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(
                f"CUDA index {index} is outside the {torch.cuda.device_count()} "
                "visible devices"
            )
        return torch.device(f"cuda:{index}")
    return device


def _resolve_existing_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path).expanduser()
    candidates = (
        path,
        Path.cwd() / path,
        SCRIPT_DIR / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return path.resolve()


def _resolve_checkpoint_path(raw_path: str, *, config_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    candidates = (
        path,
        Path.cwd() / path,
        SCRIPT_DIR / path,
        config_path.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint does not exist: {raw_path}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


@lru_cache(maxsize=None)
def _sha256_file(path_text: str) -> str:
    """Hash one resolved file without loading large checkpoints into memory."""

    path = Path(path_text).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint code, configs, and checkpoints that affect an experiment."""

    model_config_path = _resolve_existing_path(args.model_config)
    task_config_path = _resolve_existing_path(args.task_config)
    model_config = _read_yaml(model_config_path)
    task_config = _read_yaml(task_config_path)
    model_checkpoint = _resolve_checkpoint_path(
        str(model_config["model_path"]), config_path=model_config_path
    )
    classifier_checkpoint = _resolve_checkpoint_path(
        str(task_config["classifier_path"]), config_path=task_config_path
    )

    source_paths = (
        Path(__file__).resolve(),
        SCRIPT_DIR / "vista_schedules.py",
        SCRIPT_DIR / "vista_metrics.py",
        SCRIPT_DIR / "image_diffusion" / "vista_smc.py",
        SCRIPT_DIR / "image_diffusion" / "operators.py",
        SCRIPT_DIR / "image_diffusion" / "script_util.py",
        SCRIPT_DIR / "image_diffusion" / "smc_diffusion.py",
        SCRIPT_DIR / "image_diffusion" / "resnet.py",
    )
    source_hashes = {
        str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(str(path.resolve()))
        for path in source_paths
    }
    serialized_sources = json.dumps(
        source_hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "implementation_sha256": hashlib.sha256(serialized_sources).hexdigest(),
        "source_sha256": source_hashes,
        "model_config_sha256": _sha256_file(str(model_config_path)),
        "task_config_sha256": _sha256_file(str(task_config_path)),
        "model_checkpoint": str(model_checkpoint),
        "model_checkpoint_sha256": _sha256_file(str(model_checkpoint)),
        "classifier_checkpoint": str(classifier_checkpoint),
        "classifier_checkpoint_sha256": _sha256_file(
            str(classifier_checkpoint)
        ),
    }


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def load_runtime_components(
    args: argparse.Namespace, device: torch.device
) -> RuntimeComponents:
    """Load pretrained inference components; this function never trains."""

    model_config_path = _resolve_existing_path(args.model_config)
    task_config_path = _resolve_existing_path(args.task_config)
    model_config = model_and_diffusion_defaults()
    model_config.update(_read_yaml(model_config_path))
    task_config = _read_yaml(task_config_path)

    model_config["timestep_respacing"] = args.total_steps
    # VISTA owns proposal construction. The diffusion object only supplies p.
    model_config["sampler"] = "unconditional"
    model, diffusion = create_model_and_diffusion(
        **select_args(model_config, model_and_diffusion_defaults().keys())
    )

    model_checkpoint = _resolve_checkpoint_path(
        str(model_config["model_path"]), config_path=model_config_path
    )
    state_dict = dist_util.load_state_dict(
        str(model_checkpoint), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.to(device)
    if model_config.get("use_fp16"):
        model.convert_to_fp16()
    _freeze(model)

    classifier_checkpoint = _resolve_checkpoint_path(
        str(task_config["classifier_path"]), config_path=task_config_path
    )
    classifier_info = torch.load(classifier_checkpoint, map_location="cpu")
    if not isinstance(classifier_info, dict) or "state_dict" not in classifier_info:
        raise ValueError(
            f"invalid classifier checkpoint structure: {classifier_checkpoint}"
        )
    classifier_network = get_resnet_model(
        arch=classifier_info["arch"], num_classes=10
    )
    # The historical checkpoint was saved from nn.DataParallel. Loading the
    # underlying module avoids accidentally spanning all visible GPUs.
    classifier_state = {
        key.removeprefix("module."): value
        for key, value in classifier_info["state_dict"].items()
    }
    classifier_network.load_state_dict(classifier_state)
    classifier = ScaledMNISTClassifier(classifier_network).to(device)
    _freeze(classifier)

    diffusion.t_truncate = 0
    diffusion.task = "class_cond_gen"
    if int(diffusion.T) != args.total_steps:
        raise RuntimeError(
            f"respaced diffusion has T={diffusion.T}; expected {args.total_steps}"
        )
    timestep_map = list(getattr(diffusion, "timestep_map", []))
    if timestep_map and len(timestep_map) != args.total_steps:
        raise RuntimeError("diffusion timestep_map length does not match T")

    return RuntimeComponents(
        diffusion=diffusion,
        model=model,
        classifier=classifier,
        model_kwargs={},
        metadata={
            "model_config": str(model_config_path),
            "task_config": str(task_config_path),
            "model_checkpoint": str(model_checkpoint),
            "classifier_checkpoint": str(classifier_checkpoint),
            "model_checkpoint_sha256": _sha256_file(str(model_checkpoint)),
            "classifier_checkpoint_sha256": _sha256_file(
                str(classifier_checkpoint)
            ),
            "timestep_map": timestep_map,
            "training_performed": False,
        },
    )


def average_value_curves(
    curves: Sequence[Any], expected_length: int
) -> list[float]:
    """Average finite full-guidance curves and return native floats."""

    if not curves:
        raise ValueError("at least one warmup value curve is required")
    converted: list[torch.Tensor] = []
    for index, curve in enumerate(curves):
        tensor = torch.as_tensor(curve, dtype=torch.float64).detach().cpu()
        if tensor.ndim != 1 or tensor.shape[0] != expected_length:
            raise ValueError(
                f"warmup curve {index} must have shape ({expected_length},), "
                f"got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            bad = torch.nonzero(~torch.isfinite(tensor), as_tuple=False).flatten()
            raise ValueError(
                f"warmup curve {index} has non-finite entries at {bad.tolist()}"
            )
        converted.append(tensor)
    mean_curve = torch.stack(converted, dim=0).mean(dim=0)
    return [float(value) for value in mean_curve.tolist()]


def aggregate_numeric_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Scalar | None]]:
    """Aggregate scalar fields using the mean and sample standard deviation."""

    keys = sorted({key for record in records for key in record})
    aggregate: dict[str, dict[str, Scalar | None]] = {}
    for key in keys:
        finite_values: list[float] = []
        for record in records:
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                finite_values.append(numeric)
        if not finite_values:
            continue
        mean = float(np.mean(finite_values))
        sample_std = (
            float(np.std(finite_values, ddof=1))
            if len(finite_values) >= 2
            else None
        )
        aggregate[key] = {
            "mean": mean,
            "sample_std": sample_std,
            "finite_runs": len(finite_values),
        }
    return aggregate


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return repr(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _plot_value_curves(
    path: Path,
    curves: Sequence[Any],
    averaged_values: Sequence[float],
    *,
    total_steps: int,
) -> None:
    """Plot warmup curves in reverse-process execution order, T -> 0."""

    # Matplotlib is deliberately imported lazily so importing the runner does
    # not initialize a GUI backend on headless compute nodes.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    curve_array = np.asarray(
        [
            average_value_curves([curve], total_steps + 1)
            for curve in curves
        ],
        dtype=np.float64,
    )
    mean_array = np.asarray(
        average_value_curves([averaged_values], total_steps + 1),
        dtype=np.float64,
    )
    paper_timesteps = np.arange(total_steps, -1, -1, dtype=np.int64)

    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    try:
        for index, curve in enumerate(curve_array):
            axis.plot(
                paper_timesteps,
                curve[::-1],
                color="tab:blue",
                linewidth=1.0,
                alpha=0.3,
                label="Warmup runs" if index == 0 else None,
            )
        axis.plot(
            paper_timesteps,
            mean_array[::-1],
            color="black",
            linewidth=2.2,
            label=r"Mean $\widehat{V}_t$",
        )
        axis.set_xlim(total_steps, 0)
        axis.set_xlabel(r"Paper timestep $t$ (reverse process: $T \rightarrow 0$)")
        axis.set_ylabel(r"$\widehat{V}_t$")
        axis.set_title("Full-guidance warmup value estimates")
        axis.grid(True, alpha=0.25)
        axis.legend()

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f"{path.stem}.tmp-{os.getpid()}{path.suffix}"
        )
        figure.savefig(temporary, dpi=180, format="png")
        os.replace(temporary, path)
    finally:
        plt.close(figure)


def _save_warmup_curve_artifacts(
    warmup_root: Path,
    curves: Sequence[Any],
    averaged_values: Sequence[float],
    *,
    total_steps: int,
    plot_vhat: bool,
) -> dict[str, str | None]:
    """Save a combined curve archive and, by default, a headless PNG plot."""

    canonical_curves = np.asarray(
        [
            average_value_curves([curve], total_steps + 1)
            for curve in curves
        ],
        dtype=np.float64,
    )
    canonical_mean = np.asarray(
        average_value_curves([averaged_values], total_steps + 1),
        dtype=np.float64,
    )
    paper_timesteps = np.arange(total_steps, -1, -1, dtype=np.int64)
    combined_name = "value_curves.npz"
    _atomic_save_npz(
        warmup_root / combined_name,
        # Canonical arrays follow the sampler convention [V_hat_0, ..., V_hat_T].
        value_curves=canonical_curves,
        mean_value_means=canonical_mean,
        # These arrays are ready to consume in reverse-process execution order.
        paper_timesteps=paper_timesteps,
        value_curves_T_to_0=canonical_curves[:, ::-1],
        mean_value_curve_T_to_0=canonical_mean[::-1],
    )

    plot_name: str | None = None
    if plot_vhat:
        plot_name = "value_curve.png"
        _plot_value_curves(
            warmup_root / plot_name,
            canonical_curves,
            canonical_mean,
            total_steps=total_steps,
        )
    return {
        "combined_npz": str((warmup_root / combined_name).resolve()),
        "value_curve_plot": (
            str((warmup_root / plot_name).resolve())
            if plot_name is not None
            else None
        ),
    }


def _save_grid(path: Path, samples: torch.Tensor) -> None:
    samples = samples.detach().to(device="cpu", dtype=torch.float32)
    if samples.ndim != 4 or samples.shape[0] == 0:
        raise ValueError("grid samples must have shape (N, C, H, W) with N>0")
    nrow = max(1, math.ceil(math.sqrt(samples.shape[0])))
    grid = make_grid(
        samples,
        nrow=nrow,
        normalize=True,
        value_range=(-1.0, 1.0),
    )
    array = (
        grid.mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    if array.shape[-1] == 1:
        array = array[..., 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def classifier_log_probs(
    classifier: Callable[[torch.Tensor], torch.Tensor],
    samples: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Return the untempered, full class log-probability matrix."""

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, samples.shape[0], batch_size):
            logits = classifier(samples[start : start + batch_size])
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise ValueError("classifier must return logits with shape (N, C)")
            outputs.append(F.log_softmax(logits, dim=-1).detach().cpu())
    if not outputs:
        raise ValueError("sampler returned no samples")
    return torch.cat(outputs, dim=0)


def normalized_final_particle_weights(
    stats: Mapping[str, Any], num_particles: int
) -> torch.Tensor:
    """Return normalized CPU weights, defaulting to equal weights for mocks."""

    raw = stats.get("final_log_weights")
    if raw is None:
        return torch.full(
            (num_particles,), 1.0 / num_particles, dtype=torch.float64
        )
    log_weights = torch.as_tensor(raw, dtype=torch.float64).detach().cpu()
    if log_weights.shape != (num_particles,):
        raise ValueError(
            "sampler final_log_weights must have shape "
            f"({num_particles},), got {tuple(log_weights.shape)}"
        )
    if bool((torch.isnan(log_weights) | torch.isposinf(log_weights)).any()):
        raise FloatingPointError("sampler final_log_weights contain NaN or +Inf")
    normalizer = torch.logsumexp(log_weights, dim=0)
    if not bool(torch.isfinite(normalizer)):
        raise FloatingPointError("all sampler final weights are zero")
    return (log_weights - normalizer).exp()


def _warmup_compatibility_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return fields that can change a full-guidance V-hat estimate."""

    return {
        "provenance": _scientific_provenance(args),
        "model_config": str(_resolve_existing_path(args.model_config)),
        "task_config": str(_resolve_existing_path(args.task_config)),
        "target_class": args.target_class,
        "total_steps": args.total_steps,
        "num_particles": args.num_particles,
        "num_rollouts": args.num_rollouts,
        "warmup_runs": args.warmup_runs,
        "seed": args.seed,
        "warmup_seed_offset": args.warmup_seed_offset,
        "alpha": args.alpha,
        "ess_threshold": args.ess_threshold,
        "partial_resample": args.partial_resample,
        "final_transition": args.final_transition,
        "value_estimation": args.value_estimation,
        "value_estimator": _value_estimator_metadata(args),
        "proposal_potential": "classifier_on_pred_xstart",
        "proposal_family": (
            "upstream_vp_tds_twisted_xstart_posterior"
            if args.clip_twisted
            else "vp_tds_twisted_xstart_posterior_unclipped_ablation"
        ),
        "warmup_value_aggregation": (
            "post_resampling_normalized_weight_mean_eq11"
        ),
        "terminal_resampling": "scheduled_adaptive_partial_only",
        "partial_resampling_rule": "half_high_half_low_systematic",
        "clip_denoised": args.clip_denoised,
        "clip_twisted": args.clip_twisted,
    }


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "provenance": _scientific_provenance(args),
        "model_config": str(_resolve_existing_path(args.model_config)),
        "task_config": str(_resolve_existing_path(args.task_config)),
        "target_class": args.target_class,
        "total_steps": args.total_steps,
        "guidance_steps": args.guidance_steps,
        "num_particles": args.num_particles,
        "num_rollouts": args.num_rollouts,
        "warmup_runs": args.warmup_runs,
        "eval_runs": args.eval_runs,
        "eval_start_index": args.eval_start_index,
        "seed": args.seed,
        "warmup_seed_offset": args.warmup_seed_offset,
        "alpha": args.alpha,
        "ess_threshold": args.ess_threshold,
        "partial_resample": args.partial_resample,
        "final_transition": args.final_transition,
        "value_estimation": args.value_estimation,
        "value_estimator": _value_estimator_metadata(args),
        "proposal_potential": "classifier_on_pred_xstart",
        "proposal_family": (
            "upstream_vp_tds_twisted_xstart_posterior"
            if args.clip_twisted
            else "vp_tds_twisted_xstart_posterior_unclipped_ablation"
        ),
        "warmup_value_aggregation": (
            "post_resampling_normalized_weight_mean_eq11"
        ),
        "terminal_resampling": "scheduled_adaptive_partial_only",
        "partial_resampling_rule": "half_high_half_low_systematic",
        "clip_denoised": args.clip_denoised,
        "clip_twisted": args.clip_twisted,
        "policies": list(args.policies),
        "valid_logprob_threshold": args.valid_logprob_threshold,
        "foreground_threshold": args.foreground_threshold,
        "duplicate_threshold": args.duplicate_threshold,
        "classifier_batch_size": args.classifier_batch_size,
        "save_grids": args.save_grids,
        "plot_vhat": args.plot_vhat,
        "warmup_only": args.warmup_only,
        "warmup_from": args.warmup_from,
    }
    # This marker is needed only by the intentionally distinct no-bootstrap
    # baseline mode; ordinary runs omit it from the standard manifest.
    if args.full_baseline_only:
        config["full_baseline_only"] = True
    return config


def _value_estimator_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the actual value-estimation work without overloading ``J``."""

    if args.value_estimation == "one_shot":
        return {
            "definition": "classifier_on_one_shot_pred_xstart_delta",
            "paper_value_sampling_structure_aligned": True,
            "continuous_tds_implementation_aligned": args.clip_twisted,
            "proposal_implementation_aligned": args.clip_twisted,
            "direct_p0_given_xt_distribution": "delta_at_pred_xstart",
            "exact_p0_given_xt_available": False,
            "vista_theory_aligned": False,
            "vista_value_role": "continuous_delta_p0_given_xt_approximation",
            "supports_theoretical_utd": False,
            "num_rollouts_parameter_used": False,
            "effective_one_shot_samples": 1,
            "effective_ancestral_rollouts": 0,
            "denoiser_calls_per_value_evaluation": 1,
            "denoiser_calls_per_noisy_state": 1,
            "state_kernel_cache": "resampling_ancestor_indexed",
            "proposal_mean_shift": (
                "shift_pred_xstart_clip_then_recompute_posterior_mean"
                if args.clip_twisted
                else "shift_pred_xstart_then_recompute_posterior_mean_unclipped"
            ),
            "warmup_value_statistics_estimator": (
                "post_resampling_normalized_weight_mean_of_tweedie_value"
            ),
            "finite_j_self_weighting_bias": "not_applicable_deterministic_delta",
        }
    if args.value_estimation == "one_shot_gaussian":
        return {
            "definition": "monte_carlo_mean_reward_over_one_shot_gaussian_x0",
            "paper_value_sampling_structure_aligned": True,
            "paper_cost_model_aligned": True,
            "continuous_tds_implementation_aligned": False,
            "proposal_implementation_aligned": args.clip_twisted,
            "continuous_extension": True,
            "direct_p0_given_xt_distribution": (
                "gaussian_pred_xstart_mean_unit_prior_posterior_variance"
            ),
            "one_shot_clean_variance": (
                "1_minus_alpha_cumprod_at_noisy_state"
            ),
            "exact_p0_given_xt_available": False,
            "vista_theory_aligned": False,
            "vista_value_role": (
                "continuous_gaussian_p0_given_xt_approximation"
            ),
            "supports_theoretical_utd": False,
            "num_rollouts_parameter_used": True,
            "effective_one_shot_samples": args.num_rollouts,
            "effective_ancestral_rollouts": 0,
            "denoiser_calls_per_value_evaluation": 1,
            "classifier_reward_evaluations_per_value_evaluation": (
                args.num_rollouts
            ),
            "denoiser_calls_per_noisy_state": 1,
            "state_kernel_cache": "resampling_ancestor_indexed",
            "proposal_mean_shift": (
                "shift_pred_xstart_clip_then_recompute_posterior_mean"
                if args.clip_twisted
                else "shift_pred_xstart_then_recompute_posterior_mean_unclipped"
            ),
            "proposal_potential": "classifier_on_pred_xstart_delta",
            "warmup_value_statistics_estimator": (
                "post_resampling_normalized_weight_mean_of_same_j_sample_estimate"
            ),
            "finite_j_self_weighting_bias": (
                "possible_same_estimate_used_for_weight_and_statistic"
            ),
        }
    return {
        "definition": "monte_carlo_mean_reward_over_p_reverse_paths",
        "paper_value_sampling_structure_aligned": False,
        "paper_cost_model_aligned": False,
        "vista_value_identity_aligned": True,
        "proposal_implementation_aligned": args.clip_twisted,
        "vista_value_role": "monte_carlo_conditional_expectation",
        "supports_theoretical_utd": True,
        "num_rollouts_parameter_used": True,
        "effective_ancestral_rollouts": args.num_rollouts,
        "denoiser_calls_per_value_evaluation": None,
        "ancestral_rollouts_per_weight_query": args.num_rollouts,
        "warmup_ancestral_rollouts_per_value_statistic_query": (
            args.num_rollouts
        ),
        "warmup_value_statistics_estimator": (
            "post_resampling_normalized_weight_mean_of_weight_rollout_estimates"
        ),
        "finite_j_self_weighting_avoided": False,
    }


def utd_result_fields(
    value: float | None,
    *,
    value_estimation: str,
) -> dict[str, Any]:
    """Label a schedule score according to what its value curve estimates."""

    if value_estimation not in {
        "ancestral",
        "one_shot",
        "one_shot_gaussian",
    }:
        raise ValueError(f"unknown value estimator: {value_estimation!r}")
    if value is None:
        return {
            "empirical_utd": None,
            "proxy_utd": None,
            "utd_interpretation": "unavailable_without_warmup_values",
        }
    numeric = float(value)
    if value_estimation == "ancestral":
        return {
            "empirical_utd": numeric,
            "proxy_utd": None,
            "utd_interpretation": (
                "monte_carlo_estimate_of_continuous_analogue_of_"
                "theorem_4_3_bound"
            ),
        }
    interpretation = (
        "one_shot_gaussian_p0_proxy_not_a_proven_upper_bound"
        if value_estimation == "one_shot_gaussian"
        else "one_shot_pred_xstart_delta_proxy_not_a_proven_upper_bound"
    )
    return {
        "empirical_utd": None,
        "proxy_utd": numeric,
        "utd_interpretation": interpretation,
    }


def _experiment_name(
    args: argparse.Namespace, scientific_config: Mapping[str, Any]
) -> str:
    if args.run_name is not None:
        return args.run_name
    serialized = json.dumps(scientific_config, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()[:10]
    if args.value_estimation == "one_shot":
        estimator_tag = "TDSoneShotJ1"
    elif args.value_estimation == "one_shot_gaussian":
        estimator_tag = f"TDSoneShotGaussianJ{args.num_rollouts}"
    else:
        estimator_tag = f"ancestralJ{args.num_rollouts}"
    return (
        f"mnist_y{args.target_class}_T{args.total_steps}"
        f"_Tp{args.guidance_steps}_N{args.num_particles}"
        f"_{estimator_tag}_M{args.warmup_runs}"
        f"_eval{args.eval_start_index}-{args.eval_start_index + args.eval_runs - 1}"
        f"_seed{args.seed}_{digest}"
    )


class _DirectoryLock:
    def __init__(self, experiment_dir: Path):
        self.path = experiment_dir / ".runner.lock"
        self.acquired = False

    def __enter__(self) -> "_DirectoryLock":
        try:
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError as exc:
            raise RuntimeError(
                f"experiment is already locked: {self.path}. If no process is "
                "running, remove this stale lock file explicitly."
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.acquired = False


def _prepare_experiment(
    args: argparse.Namespace, scientific_config: Mapping[str, Any]
) -> Path:
    output_root = Path(args.output_dir).expanduser().resolve()
    experiment_dir = output_root / _experiment_name(args, scientific_config)
    manifest_path = experiment_dir / "manifest.json"

    if experiment_dir.exists() and not args.resume:
        raise FileExistsError(
            f"experiment directory already exists and --no-resume was used: "
            f"{experiment_dir}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            prior_manifest = json.load(handle)
        if prior_manifest.get("config") != _json_safe(scientific_config):
            raise RuntimeError(
                "existing experiment manifest does not match this configuration; "
                "choose a different --run-name or output directory"
            )
    else:
        unexpected = [
            path
            for path in experiment_dir.iterdir()
            if path.name != ".runner.lock"
        ]
        if unexpected:
            raise RuntimeError(
                f"refusing to adopt non-empty directory without a manifest: "
                f"{experiment_dir}"
            )
        _atomic_write_json(
            manifest_path,
            {
                "config": scientific_config,
                "created_unix_time": time.time(),
            },
        )
    return experiment_dir


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _resolve_source_artifact(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_warmup_source(
    source_path: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load and validate reusable V-hat data from a prior warmup."""

    outer = _load_json(source_path)
    nested = outer.get("warmup")
    if isinstance(nested, Mapping):
        warmup = dict(nested)
        # Top-level experiment summaries contain paths rooted at the experiment.
        artifact_base = source_path.parent
        source_config = outer.get("config")
    elif "value_means" in outer:
        warmup = outer
        artifact_base = source_path.parent
        source_config = warmup.get("warmup_config")
        if source_config is None and source_path.parent.name == "warmup":
            manifest_path = source_path.parent.parent / "manifest.json"
            if manifest_path.is_file():
                source_config = _load_json(manifest_path).get("config")
    else:
        raise ValueError(
            f"warmup source must contain value_means or a warmup section: "
            f"{source_path}"
        )

    expected_config = _warmup_compatibility_config(args)
    if source_config is None:
        source_config = warmup.get("warmup_config")
    compatibility_verified = isinstance(source_config, Mapping)
    if compatibility_verified:
        for field, expected in expected_config.items():
            if field not in source_config:
                raise ValueError(
                    f"warmup source config is missing required field {field!r}: "
                    f"{source_path}"
                )
            actual = source_config[field]
            if _json_safe(actual) != _json_safe(expected):
                raise ValueError(
                    f"warmup source is incompatible for {field}: "
                    f"source={actual!r}, requested={expected!r}"
                )

    source_m = warmup.get("M")
    if isinstance(source_m, bool) or not isinstance(source_m, int):
        raise ValueError(f"warmup source M must be an integer: {source_path}")
    if source_m != args.warmup_runs:
        raise ValueError(
            f"warmup source M={source_m} does not match requested "
            f"M={args.warmup_runs}"
        )

    expected_schedule = list(range(args.total_steps))
    if warmup.get("schedule") != expected_schedule:
        raise ValueError(
            "warmup source schedule is not the full paper-index schedule "
            f"[0, ..., {args.total_steps - 1}]"
        )
    averaged_values = average_value_curves(
        [warmup.get("value_means")], args.total_steps + 1
    )

    source_curves: list[list[float]] = []
    artifacts = warmup.get("artifacts")
    combined_path: Path | None = None
    if isinstance(artifacts, Mapping) and artifacts.get("combined_npz"):
        combined_path = _resolve_source_artifact(
            str(artifacts["combined_npz"]), base_dir=artifact_base
        )
        if not combined_path.is_file():
            raise FileNotFoundError(
                f"warmup combined curve artifact does not exist: {combined_path}"
            )
        with np.load(combined_path, allow_pickle=False) as archive:
            if "value_curves" not in archive:
                raise ValueError(
                    f"warmup combined archive lacks value_curves: {combined_path}"
                )
            archived_curves = np.asarray(archive["value_curves"], dtype=np.float64)
        if archived_curves.ndim != 2:
            raise ValueError(
                f"warmup value_curves must be a matrix: {combined_path}"
            )
        source_curves = [
            average_value_curves([curve], args.total_steps + 1)
            for curve in archived_curves
        ]
    else:
        runs = warmup.get("runs")
        if isinstance(runs, list) and runs:
            for run in runs:
                if not isinstance(run, Mapping) or not run.get("result"):
                    source_curves = []
                    break
                result_path = _resolve_source_artifact(
                    str(run["result"]), base_dir=source_path.parent / "warmup"
                    if isinstance(nested, Mapping)
                    else artifact_base
                )
                if not result_path.is_file():
                    source_curves = []
                    break
                result = _load_json(result_path)
                source_curves.append(
                    average_value_curves(
                        [result.get("value_means")], args.total_steps + 1
                    )
                )

    # Older summaries may expose only the M-run average. It remains a valid
    # schedule estimate and is plotted as the single available source curve.
    if not source_curves:
        source_curves = [averaged_values]
    elif len(source_curves) != source_m:
        raise ValueError(
            f"warmup source has {len(source_curves)} curves but declares M={source_m}"
        )

    reconstructed_mean = np.asarray(
        average_value_curves(source_curves, args.total_steps + 1),
        dtype=np.float64,
    )
    if len(source_curves) == source_m and not np.allclose(
        reconstructed_mean,
        np.asarray(averaged_values, dtype=np.float64),
        rtol=1e-7,
        atol=1e-12,
    ):
        raise ValueError(
            "warmup source value_means does not match the mean of its stored curves"
        )

    return {
        "summary_path": str(source_path),
        "warmup": warmup,
        "curves": source_curves,
        "averaged_values": averaged_values,
        "compatibility_verified": compatibility_verified,
        "combined_npz": str(combined_path) if combined_path is not None else None,
    }


def _call_sampler(
    *,
    sampler_fn: Callable[..., tuple[torch.Tensor, dict[str, Any], torch.Tensor]],
    components: RuntimeComponents,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    schedule: Iterable[int],
    collect_value_statistics: bool,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, float]:
    _seed_everything(seed)
    generator = _make_generator(device, seed)
    start = time.perf_counter()
    samples, stats, value_means = sampler_fn(
        components.diffusion,
        components.model,
        components.classifier,
        args.target_class,
        args.num_particles,
        schedule,
        model_kwargs=components.model_kwargs,
        device=device,
        value_estimation=args.value_estimation,
        num_rollouts=args.num_rollouts,
        collect_value_statistics=collect_value_statistics,
        alpha=args.alpha,
        ess_threshold=args.ess_threshold,
        partial_resample=args.partial_resample,
        final_transition=args.final_transition,
        clip_denoised=args.clip_denoised,
        clip_twisted=args.clip_twisted,
        generator=generator,
        initial_particles=None,
    )
    elapsed = time.perf_counter() - start
    if not isinstance(samples, torch.Tensor) or samples.shape[0] != args.num_particles:
        raise ValueError(
            "sampler must return a tensor with num_particles leading entries"
        )
    if not bool(torch.isfinite(samples).all()):
        raise FloatingPointError("sampler returned non-finite samples")
    return samples.detach(), stats, value_means.detach(), float(elapsed)


def _warmup_result_is_compatible(
    result: Mapping[str, Any], *, run_index: int, seed: int
) -> bool:
    return (
        result.get("kind") == "full_guidance_warmup"
        and result.get("run_index") == run_index
        and result.get("seed") == seed
    )


def _policy_result_is_compatible(
    result: Mapping[str, Any],
    *,
    policy: str,
    run_index: int,
    seed: int,
    schedule: Sequence[int],
) -> bool:
    return (
        result.get("kind") == "policy_evaluation"
        and result.get("policy") == policy
        and result.get("run_index") == run_index
        and result.get("seed") == seed
        and result.get("schedule") == list(schedule)
    )


def run_experiment(
    args: argparse.Namespace,
    *,
    components_loader: Callable[
        [argparse.Namespace, torch.device], RuntimeComponents
    ] = load_runtime_components,
    sampler_fn: Callable[
        ..., tuple[torch.Tensor, dict[str, Any], torch.Tensor]
    ] = sample_vista_smc,
) -> dict[str, Any]:
    """Execute or resume one complete warmup and policy comparison."""

    validate_args(args)
    scientific_config = _scientific_config(args)
    experiment_dir = _prepare_experiment(args, scientific_config)
    device = resolve_device(args.device)

    with _DirectoryLock(experiment_dir):
        components = components_loader(args, device)
        if int(components.diffusion.T) != args.total_steps:
            raise ValueError(
                f"components diffusion.T={components.diffusion.T}, "
                f"expected {args.total_steps}"
            )

        full_schedule = set(range(args.total_steps))
        warmup_root = experiment_dir / "warmup"
        warmup_curves: list[Any] = []
        warmup_records: list[dict[str, Any]] = []
        warmup_phase_start = time.perf_counter()
        warmup_source_record: dict[str, Any] | None = None

        if args.full_baseline_only:
            # A full schedule is independent of V-hat.  Keep its evaluation
            # seeds in the policy baseline and derive V-hat from their stored
            # value curves later, instead of drawing a duplicate bootstrap.
            averaged_values = None
            warmup_sampling_seconds = 0.0
        elif args.warmup_from is not None:
            source_path = Path(args.warmup_from).resolve()
            if source_path in (
                (warmup_root / "summary.json").resolve(),
                (experiment_dir / "summary.json").resolve(),
            ):
                raise ValueError("warmup source cannot be this experiment itself")
            source = _load_warmup_source(source_path, args=args)
            warmup_curves.extend(source["curves"])
            averaged_values = source["averaged_values"]
            source_warmup = source["warmup"]
            warmup_sampling_seconds = 0.0
            warmup_source_record = {
                "summary": source["summary_path"],
                "compatibility_verified": source["compatibility_verified"],
                "combined_npz": source["combined_npz"],
                "M": source_warmup["M"],
                "sampling_seconds_total": source_warmup.get(
                    "sampling_seconds_total"
                ),
            }
        else:
            for run_index in range(args.warmup_runs):
                seed = args.seed + args.warmup_seed_offset + run_index
                run_dir = warmup_root / f"run_{run_index:03d}_seed_{seed}"
                result_path = run_dir / "result.json"
                if result_path.exists() and args.resume:
                    result = _load_json(result_path)
                    if not _warmup_result_is_compatible(
                        result, run_index=run_index, seed=seed
                    ):
                        raise RuntimeError(
                            f"incompatible warmup result: {result_path}"
                        )
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    samples, stats, value_means, sampling_seconds = _call_sampler(
                        sampler_fn=sampler_fn,
                        components=components,
                        args=args,
                        device=device,
                        seed=seed,
                        schedule=full_schedule,
                        collect_value_statistics=True,
                    )
                    curve = average_value_curves(
                        [value_means], args.total_steps + 1
                    )
                    _atomic_save_npz(
                        run_dir / "samples.npz",
                        samples=samples.cpu().numpy(),
                        value_means=np.asarray(curve, dtype=np.float64),
                    )
                    grid_name = None
                    if args.save_grids:
                        grid_name = "grid.png"
                        _save_grid(run_dir / grid_name, samples)
                    result = {
                        "kind": "full_guidance_warmup",
                        "run_index": run_index,
                        "seed": seed,
                        "schedule": sorted(full_schedule),
                        "value_means": curve,
                        "sampling_seconds": sampling_seconds,
                        "stats": stats,
                        "artifacts": {
                            "npz": "samples.npz",
                            "grid": grid_name,
                        },
                        "included_in_policy_metrics": False,
                    }
                    _atomic_write_json(result_path, result)
                warmup_curves.append(result["value_means"])
                warmup_records.append(result)

            averaged_values = average_value_curves(
                warmup_curves, args.total_steps + 1
            )
            warmup_sampling_seconds = float(
                sum(float(record["sampling_seconds"]) for record in warmup_records)
            )

        if args.full_baseline_only:
            warmup_artifacts = {
                "combined_npz": None,
                "value_curve_plot": None,
                "value_curve_plot_order": "paper_t_T_to_0",
            }
        else:
            warmup_artifacts = _save_warmup_curve_artifacts(
                warmup_root,
                warmup_curves,
                averaged_values,
                total_steps=args.total_steps,
                plot_vhat=args.plot_vhat,
            )
        warmup_phase_wall_seconds = (
            0.0
            if args.full_baseline_only
            else time.perf_counter() - warmup_phase_start
        )
        if args.full_baseline_only:
            warmup_summary = {
                "omitted": True,
                "reason": "full_baseline_only",
                "M": 0,
                "schedule": [],
                "value_means": None,
                "sampling_seconds_total": 0.0,
                "phase_wall_seconds": 0.0,
                "sampling_reused": False,
                "source": None,
                "artifacts": warmup_artifacts,
                "runs": [],
                "samples_excluded_from_policy_metrics": True,
            }
        else:
            warmup_summary = {
                "M": args.warmup_runs,
                "schedule": sorted(full_schedule),
                "value_means": averaged_values,
                "warmup_config": _warmup_compatibility_config(args),
                "sampling_seconds_total": warmup_sampling_seconds,
                "phase_wall_seconds": warmup_phase_wall_seconds,
                "sampling_reused": args.warmup_from is not None,
                "source": warmup_source_record,
                "artifacts": warmup_artifacts,
                "runs": [
                    {
                        "run_index": record["run_index"],
                        "seed": record["seed"],
                        "sampling_seconds": record["sampling_seconds"],
                        "result": str(
                            Path(
                                f"run_{record['run_index']:03d}_seed_{record['seed']}"
                            )
                            / "result.json"
                        ),
                    }
                    for record in warmup_records
                ],
                "samples_excluded_from_policy_metrics": True,
            }
        _atomic_write_json(warmup_root / "summary.json", warmup_summary)

        if args.warmup_only:
            summary = {
                "experiment_dir": str(experiment_dir),
                "config": scientific_config,
                "device": str(device),
                "components": components.metadata,
                "warmup": warmup_summary,
                "policies": {},
                "timing": {
                    "warmup_sampling_seconds": warmup_sampling_seconds,
                    "warmup_phase_wall_seconds": warmup_phase_wall_seconds,
                    "policy_sampling_seconds_excluding_warmup": 0.0,
                    "policy_phase_wall_seconds_excluding_warmup": 0.0,
                },
                "warmup_only": True,
                "training_performed": False,
            }
            _atomic_write_json(experiment_dir / "summary.json", summary)
            return summary

        schedule_map = build_schedule_map(
            T=args.total_steps,
            T_prime=args.guidance_steps,
            values=averaged_values,
            policies=args.policies,
        )
        policy_summaries: dict[str, Any] = {}
        policy_phase_start = time.perf_counter()

        for policy, schedule_set in schedule_map.items():
            schedule = sorted(schedule_set)
            objective_weights = timestep_weights(
                args.total_steps, policy_weight_power(policy)
            )
            policy_dir = experiment_dir / "policies" / policy.lower()
            run_records: list[dict[str, Any]] = []
            for local_run_index in range(args.eval_runs):
                # The same evaluation seed is intentionally reused by every
                # policy at a given global run index. Distinct
                # --eval-start-index ranges therefore form disjoint shards.
                run_index = args.eval_start_index + local_run_index
                seed = args.seed + run_index
                run_dir = policy_dir / f"run_{run_index:03d}_seed_{seed}"
                result_path = run_dir / "result.json"
                if result_path.exists() and args.resume:
                    result = _load_json(result_path)
                    if not _policy_result_is_compatible(
                        result,
                        policy=policy,
                        run_index=run_index,
                        seed=seed,
                        schedule=schedule,
                    ):
                        raise RuntimeError(
                            f"incompatible policy result: {result_path}"
                        )
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    samples, stats, value_means, sampling_seconds = _call_sampler(
                        sampler_fn=sampler_fn,
                        components=components,
                        args=args,
                        device=device,
                        seed=seed,
                        schedule=schedule_set,
                        collect_value_statistics=(
                            args.full_baseline_only and policy == "full"
                        ),
                    )
                    evaluation_start = time.perf_counter()
                    log_probs = classifier_log_probs(
                        components.classifier,
                        samples,
                        batch_size=args.classifier_batch_size,
                    )
                    metrics = compute_vista_metrics(
                        samples.cpu(),
                        log_probs,
                        target_label=args.target_class,
                        valid_logprob_threshold=args.valid_logprob_threshold,
                        foreground_threshold=args.foreground_threshold,
                        duplicate_threshold=args.duplicate_threshold,
                    )
                    final_weights = normalized_final_particle_weights(
                        stats, args.num_particles
                    )
                    valid_mask = (
                        log_probs[:, args.target_class]
                        > args.valid_logprob_threshold
                    )
                    metrics["weighted_valid_probability"] = float(
                        final_weights[valid_mask].sum().item()
                    )
                    metrics["final_weight_ess"] = float(
                        final_weights.square().sum().reciprocal().item()
                    )
                    evaluation_seconds = time.perf_counter() - evaluation_start

                    _atomic_save_npz(
                        run_dir / "samples.npz",
                        samples=samples.cpu().numpy(),
                        classifier_log_probs=log_probs.numpy(),
                        final_normalized_weights=final_weights.numpy(),
                        value_means=value_means.cpu().numpy(),
                    )
                    grid_name = None
                    if args.save_grids:
                        grid_name = "grid.png"
                        _save_grid(run_dir / grid_name, samples)
                    result = {
                        "kind": "policy_evaluation",
                        "policy": policy,
                        "run_index": run_index,
                        "seed": seed,
                        "schedule": schedule,
                        "schedule_execution_order": execution_order(schedule),
                        "metrics": metrics,
                        "metric_config": {
                            "target_class": args.target_class,
                            "valid_logprob_threshold": (
                                args.valid_logprob_threshold
                            ),
                            "validity_comparison": "strict_greater_than",
                            "foreground_threshold": args.foreground_threshold,
                            "foreground_comparison": (
                                "greater_than_or_equal"
                            ),
                            "duplicate_threshold": args.duplicate_threshold,
                            "duplicate_comparison": (
                                "greater_than_or_equal"
                            ),
                            "unique_valid_weight_treatment": (
                                "unweighted_retained_particles_after_scheduled_"
                                "adaptive_partial_resampling"
                            ),
                            "weighted_valid_probability": (
                                "normalized_final_particle_weight_mass"
                            ),
                            "extra_terminal_full_resampling": False,
                        },
                        "raw_evaluation": {
                            "schema_version": RAW_EVALUATION_SCHEMA_VERSION,
                            "sample_range": "minus_one_to_one",
                            "target_class": args.target_class,
                            "classifier_log_probs_semantics": (
                                "full_untempered_natural_log_softmax"
                            ),
                            "arrays": {
                                "samples": "samples",
                                "classifier_log_probs": (
                                    "classifier_log_probs"
                                ),
                                "final_normalized_weights": (
                                    "final_normalized_weights"
                                ),
                                "value_means": "value_means",
                            },
                            "value_statistics_collected": bool(
                                stats.get("value_statistics_collected", False)
                            ),
                            "value_statistics_estimator": stats.get(
                                "value_statistics_estimator", "unknown"
                            ),
                            "supports_threshold_reevaluation": True,
                        },
                        "timing": {
                            "sampling_seconds": sampling_seconds,
                            "evaluation_seconds": evaluation_seconds,
                            "seconds_excluding_warmup": (
                                sampling_seconds + evaluation_seconds
                            ),
                        },
                        "stats": stats,
                        "artifacts": {
                            "npz": "samples.npz",
                            "raw_evaluation_npz": "samples.npz",
                            "grid": grid_name,
                        },
                        "warmup_samples_included": False,
                    }
                    _atomic_write_json(result_path, result)
                run_records.append(result)

            metric_records = [record["metrics"] for record in run_records]
            timing_records = [record["timing"] for record in run_records]
            policy_summary = {
                "policy": policy,
                "schedule": schedule,
                "schedule_execution_order": execution_order(schedule),
                "active_steps": len(schedule),
                "timestep_weight": (
                    {"formula": "(1-t/T)^k", "power": policy_weight_power(policy)}
                    if policy_weight_power(policy) is not None
                    else {"formula": "1", "power": 0}
                ),
                "empirical_objective_definition": (
                    "sum_{t not_in G} lambda(t) V_hat_{ceil_G(t)}"
                ),
                "empirical_objective": (
                    schedule_objective(
                        args.total_steps,
                        schedule,
                        averaged_values,
                        weights=objective_weights,
                    )
                    if averaged_values is not None
                    else None
                ),
                **utd_result_fields(
                    (
                        empirical_utd(
                            args.total_steps,
                            schedule,
                            averaged_values,
                            weights=objective_weights,
                        )
                        if averaged_values is not None
                        else None
                    ),
                    value_estimation=args.value_estimation,
                ),
                "aggregate_metrics": aggregate_numeric_records(metric_records),
                "aggregate_timing_excluding_warmup": aggregate_numeric_records(
                    timing_records
                ),
                "warmup_seconds_excluded": warmup_sampling_seconds,
                "runs": [
                    {
                        "run_index": record["run_index"],
                        "seed": record["seed"],
                        "metrics": record["metrics"],
                        "timing": record["timing"],
                        "result": str(
                            Path(
                                f"run_{record['run_index']:03d}"
                                f"_seed_{record['seed']}"
                            )
                            / "result.json"
                        ),
                    }
                    for record in run_records
                ],
            }
            _atomic_write_json(policy_dir / "summary.json", policy_summary)
            policy_summaries[policy] = policy_summary

        policy_phase_wall_seconds = time.perf_counter() - policy_phase_start
        if args.full_baseline_only:
            # Full guidance is the baseline and supplies the shared V-hat:
            # retain its first M fixed-seed curves, but never mix its samples
            # into a sparse policy's metrics.
            full_runs = policy_summaries["full"]["runs"][: args.warmup_runs]
            full_curves = []
            for record in full_runs:
                raw_path = (
                    experiment_dir / "policies" / "full" / record["result"]
                ).parent / "samples.npz"
                with np.load(raw_path) as archive:
                    full_curves.append(archive["value_means"])
            averaged_values = average_value_curves(
                full_curves, args.total_steps + 1
            )
            warmup_summary = {
                "M": args.warmup_runs,
                "schedule": sorted(full_schedule),
                "value_means": averaged_values,
                "warmup_config": _warmup_compatibility_config(args),
                "sampling_seconds_total": float(
                    sum(run["timing"]["sampling_seconds"] for run in full_runs)
                ),
                "phase_wall_seconds": policy_phase_wall_seconds,
                "sampling_reused": True,
                "source": "first_M_full_baseline_evaluation_runs",
                "artifacts": {"combined_npz": None, "value_curve_plot": None},
                # The raw curves live under policies/full rather than warmup/;
                # leave runs empty so a later --warmup-from uses value_means
                # directly instead of resolving them relative to warmup/.
                "runs": [],
                "full_baseline_runs_used_for_value_estimation": full_runs,
                "samples_excluded_from_sparse_policy_metrics": True,
            }
            _atomic_write_json(warmup_root / "summary.json", warmup_summary)
        policy_sampling_seconds = float(
            sum(
                float(run["timing"]["sampling_seconds"])
                for summary in policy_summaries.values()
                for run in summary["runs"]
            )
        )
        summary = {
            "experiment_dir": str(experiment_dir),
            "config": scientific_config,
            "device": str(device),
            "components": components.metadata,
            "warmup": warmup_summary,
            "policies": policy_summaries,
            "timing": {
                "warmup_sampling_seconds": warmup_sampling_seconds,
                "warmup_phase_wall_seconds": warmup_phase_wall_seconds,
                "policy_sampling_seconds_excluding_warmup": policy_sampling_seconds,
                "policy_phase_wall_seconds_excluding_warmup": (
                    policy_phase_wall_seconds
                ),
            },
            "warmup_only": False,
            "training_performed": False,
        }
        _atomic_write_json(experiment_dir / "summary.json", summary)
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        summary = run_experiment(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(summary["experiment_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
