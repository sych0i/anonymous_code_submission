#!/usr/bin/env python3
"""Run smc_binarized_mnist_vista.ipynb repeatedly with config overrides.

The local environment does not need jupyter/nbconvert. This script extracts
code cells from the notebook, skips IPython-only magic lines, overrides the
configuration cell, and executes each sweep item in a fresh Python subprocess.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


VALID_MODEL_TYPES = ("mdm", "remdm", "udm")
VALID_PROPOSAL_MODES = ("base", "grad")
CONFIG_CELL_INDEX = 3
VISUALIZATION_CELL_MARKER = "# Optional reward preview."
CONFIG_KEYS = {
    "notebook",
    "model_types",
    "proposal_modes",
    "guidance_steps",
    "cuda_device_index",
    "warmup_seeds",
    "eval_seeds",
    "log_dir",
    "continue_on_error",
    "dry_run",
    "skip_visualization",
}


@dataclass(frozen=True)
class SweepItem:
    model_type: str
    proposal_mode: str
    n_guidance_steps: int


def parse_csv_items(raw_values: Iterable[str], cast=str):
    values = []
    for raw in raw_values:
        for item in raw.split(","):
            item = item.strip()
            if item:
                values.append(cast(item))
    return values


def parse_optional_int(raw: str) -> int | None:
    raw = raw.strip().lower()
    if raw in {"none", "null", "-1"}:
        return None
    return int(raw)


def parse_seed_expr(raw: str) -> str:
    """Normalize a seed spec to safe Python source.

    Accepted forms:
      100:103       -> range(100, 103)
      0:50:2        -> range(0, 50, 2)
      0,1,2         -> [0, 1, 2]
    """

    raw = raw.strip()
    if not raw:
        raise ValueError("seed spec must not be empty")

    if ":" in raw:
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"invalid seed range: {raw!r}")
        nums = [int(part) for part in parts]
        return f"range({', '.join(str(num) for num in nums)})"

    nums = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not nums:
        raise ValueError(f"invalid seed list: {raw!r}")
    return repr(nums)


def normalize_config_key(key: str) -> str:
    return key.replace("-", "_")


def parse_simple_yaml_scalar(raw: str):
    raw = raw.strip()
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [parse_simple_yaml_scalar(part) for part in inner.split(",")]
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return ast.literal_eval(raw)
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_simple_yaml(source: str) -> dict:
    config = {}
    for line_number, raw_line in enumerate(source.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line {line_number}: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty config key on line {line_number}")
        config[key] = parse_simple_yaml_scalar(value)
    return config


def load_config_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = parse_simple_yaml(f.read())
    if not isinstance(config, dict):
        raise ValueError("config file must contain a mapping at the top level")

    defaults = {}
    unknown_keys = []
    for raw_key, value in config.items():
        key = normalize_config_key(str(raw_key))
        if key not in CONFIG_KEYS:
            unknown_keys.append(raw_key)
            continue
        defaults[key] = value

    if unknown_keys:
        valid = ", ".join(sorted(CONFIG_KEYS))
        raise ValueError(f"unknown config key(s): {unknown_keys}. Valid keys: {valid}")

    return defaults


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run smc_binarized_mnist_vista.ipynb for many "
            "MODEL_TYPE / PROPOSAL_MODE / N_GUIDANCE_STEPS combinations."
        )
    )
    parser.add_argument(
        "--config",
        help="YAML config file containing the same options as this CLI.",
    )
    parser.add_argument(
        "--notebook",
        default=str(script_dir / "smc_binarized_mnist_vista.ipynb"),
        help="Notebook to execute. Defaults to scripts/smc_binarized_mnist_vista.ipynb.",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["udm"],
        help="Comma or space separated model types. Choices: mdm, remdm, udm.",
    )
    parser.add_argument(
        "--proposal-modes",
        nargs="+",
        default=["grad"],
        help="Comma or space separated proposal modes. Choices: base, grad.",
    )
    parser.add_argument(
        "--guidance-steps",
        nargs="+",
        default=["25"],
        help="Comma or space separated N_GUIDANCE_STEPS values, e.g. 5,10,25,50.",
    )
    parser.add_argument(
        "--cuda-device-index",
        default="3",
        help="CUDA device index, or 'none' for torch's default CUDA device.",
    )
    parser.add_argument(
        "--warmup-seeds",
        default="100:103",
        help="Warmup seeds as start:stop[:step] or comma list. Default: 100:103.",
    )
    parser.add_argument(
        "--eval-seeds",
        default="0:50",
        help="Eval seeds as start:stop[:step] or comma list. Default: 0:50.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(script_dir / "sweep_logs"),
        help="Directory for sweep stdout/stderr logs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining combinations if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned combinations without running experiments.",
    )
    parser.add_argument(
        "--skip-visualization",
        dest="skip_visualization",
        action="store_true",
        default=True,
        help="Skip the notebook's reward preview plot cell. Enabled by default.",
    )
    parser.add_argument(
        "--with-visualization",
        dest="skip_visualization",
        action="store_false",
        help="Run visualization cells too.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[str], list[int], int | None, str, str]:
    model_types = parse_csv_items([str(value) for value in args.model_types], str)
    proposal_modes = parse_csv_items([str(value) for value in args.proposal_modes], str)
    guidance_steps = parse_csv_items([str(value) for value in args.guidance_steps], int)
    cuda_device_index = parse_optional_int(str(args.cuda_device_index))
    warmup_seed_expr = parse_seed_expr(str(args.warmup_seeds))
    eval_seed_expr = parse_seed_expr(str(args.eval_seeds))

    invalid_models = sorted(set(model_types) - set(VALID_MODEL_TYPES))
    invalid_modes = sorted(set(proposal_modes) - set(VALID_PROPOSAL_MODES))
    if invalid_models:
        raise ValueError(f"invalid model type(s): {invalid_models}")
    if invalid_modes:
        raise ValueError(f"invalid proposal mode(s): {invalid_modes}")
    if not guidance_steps or any(step <= 0 for step in guidance_steps):
        raise ValueError("guidance steps must be positive integers")

    return model_types, proposal_modes, guidance_steps, cuda_device_index, warmup_seed_expr, eval_seed_expr


def remove_ipython_only_lines(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def should_skip_visualization_cell(source: str, skip_visualization: bool) -> bool:
    return skip_visualization and source.lstrip().startswith(VISUALIZATION_CELL_MARKER)


def patch_config_cell(
    source: str,
    item: SweepItem,
    cuda_device_index: int | None,
    warmup_seed_expr: str,
    eval_seed_expr: str,
) -> str:
    replacements = {
        "MODEL_TYPE": repr(item.model_type),
        "PROPOSAL_MODE": repr(item.proposal_mode),
        "N_GUIDANCE_STEPS": str(item.n_guidance_steps),
        "CUDA_DEVICE_INDEX": "None" if cuda_device_index is None else str(cuda_device_index),
        "WARMUP_SEEDS": warmup_seed_expr,
        "EVAL_SEEDS": eval_seed_expr,
    }
    pattern = re.compile(r"^(\s*)(" + "|".join(re.escape(key) for key in replacements) + r")\s*=.*$")

    patched_lines = []
    seen = set()
    for line in source.splitlines():
        match = pattern.match(line)
        if match:
            indent, name = match.groups()
            patched_lines.append(f"{indent}{name} = {replacements[name]}")
            seen.add(name)
        else:
            patched_lines.append(line)

    missing = sorted(set(replacements) - seen)
    if missing:
        raise RuntimeError(f"configuration assignment(s) not found in notebook: {missing}")

    return "\n".join(patched_lines) + "\n"


def execute_notebook_once(
    notebook_path: Path,
    item: SweepItem,
    cuda_device_index: int | None,
    warmup_seed_expr: str,
    eval_seed_expr: str,
    skip_visualization: bool,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook_path = notebook_path.resolve()
    project_root = notebook_path.parent.parent
    os.chdir(project_root)

    with notebook_path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
    }

    print(
        "[single] start "
        f"MODEL_TYPE={item.model_type} "
        f"PROPOSAL_MODE={item.proposal_mode} "
        f"N_GUIDANCE_STEPS={item.n_guidance_steps}",
        flush=True,
    )

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if should_skip_visualization_cell(source, skip_visualization):
            print(f"[single] skip visualization cell {cell_index}", flush=True)
            continue

        source = remove_ipython_only_lines(source)
        if cell_index == CONFIG_CELL_INDEX:
            source = patch_config_cell(
                source=source,
                item=item,
                cuda_device_index=cuda_device_index,
                warmup_seed_expr=warmup_seed_expr,
                eval_seed_expr=eval_seed_expr,
            )

        code = compile(source, f"{notebook_path}:cell_{cell_index}", "exec")
        print(f"[single] execute cell {cell_index}", flush=True)
        exec(code, namespace)

    warmup_output = namespace.get("warmup_output_file")
    tstep_output = namespace.get("tstep_output_file")
    print("[single] complete", flush=True)
    if warmup_output is not None:
        print(f"[single] warmup_output_file={warmup_output}", flush=True)
    if tstep_output is not None:
        print(f"[single] tstep_output_file={tstep_output}", flush=True)


def stream_child(cmd: list[str], cwd: Path, log_file) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()

    return process.wait()


def build_single_cmd(args: argparse.Namespace, item: SweepItem) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--single",
        "--notebook",
        str(Path(args.notebook).resolve()),
        "--model-types",
        item.model_type,
        "--proposal-modes",
        item.proposal_mode,
        "--guidance-steps",
        str(item.n_guidance_steps),
        "--cuda-device-index",
        str(args.cuda_device_index),
        "--warmup-seeds",
        str(args.warmup_seeds),
        "--eval-seeds",
        str(args.eval_seeds),
    ]
    cmd.append("--skip-visualization" if args.skip_visualization else "--with-visualization")
    return cmd


def run_sweep(args: argparse.Namespace) -> int:
    model_types, proposal_modes, guidance_steps, _, _, _ = validate_args(args)
    items = [
        SweepItem(model_type=model_type, proposal_mode=proposal_mode, n_guidance_steps=n_steps)
        for model_type, proposal_mode, n_steps in itertools.product(
            model_types,
            proposal_modes,
            guidance_steps,
        )
    ]

    print(f"Planned {len(items)} run(s):")
    for index, item in enumerate(items, 1):
        print(
            f"  {index:02d}. MODEL_TYPE={item.model_type} "
            f"PROPOSAL_MODE={item.proposal_mode} "
            f"N_GUIDANCE_STEPS={item.n_guidance_steps}"
        )

    if args.dry_run:
        return 0

    notebook_path = Path(args.notebook).resolve()
    project_root = notebook_path.parent.parent
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"smc_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    failures: list[tuple[SweepItem, int]] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        header = (
            f"[sweep] log_path={log_path}\n"
            f"[sweep] notebook={notebook_path}\n"
            f"[sweep] started_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        print(header, end="", flush=True)
        log_file.write(header)
        log_file.flush()

        for index, item in enumerate(items, 1):
            banner = (
                "\n"
                + "=" * 80
                + "\n"
                + f"[sweep] run {index}/{len(items)} "
                + f"MODEL_TYPE={item.model_type} "
                + f"PROPOSAL_MODE={item.proposal_mode} "
                + f"N_GUIDANCE_STEPS={item.n_guidance_steps}\n"
                + "=" * 80
                + "\n"
            )
            print(banner, end="", flush=True)
            log_file.write(banner)
            log_file.flush()

            return_code = stream_child(build_single_cmd(args, item), cwd=project_root, log_file=log_file)
            if return_code != 0:
                failures.append((item, return_code))
                message = f"[sweep] failed with return_code={return_code}\n"
                print(message, end="", flush=True)
                log_file.write(message)
                log_file.flush()
                if not args.continue_on_error:
                    break

        footer = f"\n[sweep] finished_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if failures:
            footer += "[sweep] failures:\n"
            for item, return_code in failures:
                footer += (
                    f"  MODEL_TYPE={item.model_type} "
                    f"PROPOSAL_MODE={item.proposal_mode} "
                    f"N_GUIDANCE_STEPS={item.n_guidance_steps} "
                    f"return_code={return_code}\n"
                )
        else:
            footer += "[sweep] all runs completed successfully\n"
        print(footer, end="", flush=True)
        log_file.write(footer)

    print(f"[sweep] full log: {log_path}", flush=True)
    return 1 if failures else 0


def run_single(args: argparse.Namespace) -> int:
    model_types, proposal_modes, guidance_steps, cuda_device_index, warmup_seed_expr, eval_seed_expr = validate_args(args)
    if len(model_types) != 1 or len(proposal_modes) != 1 or len(guidance_steps) != 1:
        raise ValueError("--single requires exactly one model type, proposal mode, and guidance step")

    execute_notebook_once(
        notebook_path=Path(args.notebook),
        item=SweepItem(model_types[0], proposal_modes[0], guidance_steps[0]),
        cuda_device_index=cuda_device_index,
        warmup_seed_expr=warmup_seed_expr,
        eval_seed_expr=eval_seed_expr,
        skip_visualization=args.skip_visualization,
    )
    return 0


def main() -> int:
    parser = build_parser()
    try:
        config_args, _ = parser.parse_known_args()
        if config_args.config:
            parser.set_defaults(**load_config_defaults(config_args.config))
        args = parser.parse_args()
        if args.single:
            return run_single(args)
        return run_sweep(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
