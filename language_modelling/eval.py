import copy
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import time
import numpy as np

# Make sure we can import local helper modules (including flash_attn_compat)
_lm_dir = os.path.dirname(os.path.abspath(__file__))
if _lm_dir not in sys.path:
    sys.path.insert(0, _lm_dir)

# If flash-attn CUDA extension cannot load on this machine (glibc mismatch),
# install an SDPA-based shim before importing mdlm model code.
import flash_attn_compat

flash_attn_compat.ensure()

# Ensure mdlm package (submodule) is importable regardless of Hydra chdir
sys.path.insert(0, os.path.join(_lm_dir, "mdlm"))

import torch
import hydra
from tqdm import tqdm
import lightning
from transformers import AutoTokenizer

import smc.inference as inference
from model_revisions import pretrained_kwargs
from evaluation.mdlm_to_eval_format import get_possible_prompts, process_file
from warmup_trace_utils import (
    dp_timestep_list_vista,
    ensure_boundary_series,
    interval_steps,
    load_warmup_samples,
    resolve_path,
    top_delta_value_steps,
    top_value_steps,
    uniform_steps,
)


def _flatten_floats(vals):
    if vals is None:
        return []
    if isinstance(vals, (int, float)):
        return [float(vals)]
    out = []
    for v in vals:
        if isinstance(v, (list, tuple)):
            out.extend(_flatten_floats(v))
        else:
            out.append(float(v))
    return out


def mean_exp_alpha_r_series_from_trace_jsonl(trace_path: str) -> list[float]:
    """
    Per SMC step, return the particle mean of rollout value estimates.

    ``reward_aggregated`` is supported only for legacy traces.
    """
    series: list[float] = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            values = _flatten_floats(row.get("value_estimates"))
            if values:
                series.append(sum(values) / len(values))
                continue
            scale_cur = float(row.get("scale_cur", 5.0))
            flat = _flatten_floats(row.get("reward_aggregated", []))
            if not flat:
                series.append(float("nan"))
            else:
                series.append(sum(math.exp(scale_cur * v) for v in flat) / len(flat))
    return series[::-1]


def _samples_from_inference(text_samples, toxicity_scores, prompt_text, config):
    if config.run_all.save_all:
        return [
            {"prompt": prompt_text, "toxicity_score": score.item(), "text": txt}
            for txt, score in zip(text_samples, toxicity_scores)
        ]
    highest_toxicity_score = toxicity_scores.max()
    highest_index = toxicity_scores.argmax()
    return [
        {
            "prompt": prompt_text,
            "toxicity_score": highest_toxicity_score.item(),
            "text": text_samples[highest_index],
        }
    ]


def merge_abc_ssdlm_by_strategy(work_dir: str, strategy: str, prompt_order: list[str], out_path: str) -> None:
    """Merge abc_ssdlm_gen_p*_r*_{strategy}.jsonl into one JSONL (one row per prompt, all gens pooled)."""
    pat = os.path.join(work_dir, f"abc_ssdlm_gen_p*_r*_{strategy}.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No files matching {pat}")
    by_prompt: dict[str, dict] = {}
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                key = d["context_string"]
                if key not in by_prompt:
                    by_prompt[key] = copy.deepcopy(d)
                    by_prompt[key]["string"] = list(d["string"])
                else:
                    by_prompt[key]["string"].extend(d["string"])
    with open(out_path, "w") as f:
        for p in prompt_order:
            if p in by_prompt:
                f.write(json.dumps(by_prompt[p], ensure_ascii=False) + "\n")


def run_evaluate(generations_file: str, output_file: str, num_particles: int) -> None:
    eval_script = os.path.join(_lm_dir, "evaluation", "evaluate.py")
    metrics = (
        "ppl#gpt2-xl,cola,dist-n,toxic,toxic_ext,"
        "unique_edit#0.05"
    )
    cmd = [
        sys.executable,
        eval_script,
        "--generations_file",
        generations_file,
        "--metrics",
        metrics,
        "--output_file",
        output_file,
        "--num-particles",
        str(num_particles),
    ]
    print(f"Running evaluation: {' '.join(cmd)}")
    env = os.environ.copy()
    eval_cuda = env.get("EVAL_CUDA_VISIBLE_DEVICES")
    if eval_cuda:
        env["CUDA_VISIBLE_DEVICES"] = eval_cuda
    subprocess.run(cmd, check=True, env=env)


def run_inference_with_timing(
    config,
    guidance_steps,
    timing_log_path: str,
    prompt_idx: int,
    run_idx: int,
    warmup: bool = False,
):
    """Run inference and record only the SMC pipeline wall time."""
    tag = os.environ.get("SMC_TRACE_TAG", "")
    text_samples, toxicity_scores, elapsed = inference.main(
        config, guidance_steps
    )
    record = {
        "tag": tag,
        "prompt_idx": prompt_idx,
        "run_idx": run_idx,
        "seconds": round(elapsed, 6),
        "num_inference_steps": int(config.smc.num_inference_steps),
        "num_particles": int(config.smc.num_particles),
        "batch_p": int(config.smc.batch_p),
        "proposal_type": str(config.smc.proposal_type),
    }
    os.makedirs(os.path.dirname(os.path.abspath(timing_log_path)), exist_ok=True)
    with open(timing_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[inference timing] {tag}: {elapsed:.2f}s")
    return text_samples, toxicity_scores, elapsed


_STRATEGY_SUFFIXES = frozenset(
    {
        "allstep",
        "vista",
        "topv",
        "topdv",
        "uniformstep",
        "interval1",
        "interval2",
        "interval3",
        "interval4",
        "interval5",
    }
)
_ALL_STRATEGIES = (
    "allstep",
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
_DEFAULT_STRATEGIES = _ALL_STRATEGIES
_STRATEGY_ALIASES: dict[str, str] = {
    "all": "allstep",
    "full": "allstep",
    "fullstep": "allstep",
    "full step": "allstep",
    "first": "interval1",
    "firstkstep": "interval1",
    "middle": "interval3",
    "middlekstep": "interval3",
    "last": "interval5",
    "lastkstep": "interval5",
    "top_v": "topv",
    "top v": "topv",
    "top_dv": "topdv",
    "top dv": "topdv",
}


def _selected_strategies(config) -> set[str]:
    raw = getattr(config.run_all, "strategies", None)
    if raw is None:
        return set(_DEFAULT_STRATEGIES)
    if isinstance(raw, str):
        strategies = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        strategies = [str(part) for part in raw]
    selected = {_STRATEGY_ALIASES.get(strategy, strategy) for strategy in strategies}
    unknown = selected - set(_ALL_STRATEGIES)
    if unknown:
        raise ValueError(
            f"Unknown run_all.strategies entries: {sorted(unknown)}. "
            f"Expected one or more of {list(_ALL_STRATEGIES)}"
        )
    return selected


def _schedule_warmup_samples(config, warmup_samples):
    raw_tags = getattr(config.run_all, "eval_schedule_warmup_tags", None)
    if raw_tags is not None:
        requested_tags = (
            [part.strip() for part in raw_tags.split(",") if part.strip()]
            if isinstance(raw_tags, str)
            else [str(tag) for tag in raw_tags]
        )
        by_tag = {sample.tag: sample for sample in warmup_samples}
        missing = [tag for tag in requested_tags if tag not in by_tag]
        if missing:
            raise ValueError(
                f"run_all.eval_schedule_warmup_tags not found in manifest: {missing}"
            )
        if len(set(requested_tags)) != len(requested_tags):
            raise ValueError("run_all.eval_schedule_warmup_tags contains duplicates")
        if not requested_tags:
            raise ValueError("run_all.eval_schedule_warmup_tags must not be empty")
        return [by_tag[tag] for tag in requested_tags]

    schedule_warmup_size_cfg = getattr(
        config.run_all, "eval_schedule_warmup_samples", 3
    )
    schedule_warmup_size = (
        int(schedule_warmup_size_cfg)
        if schedule_warmup_size_cfg is not None
        else len(warmup_samples)
    )
    if schedule_warmup_size <= 0 or schedule_warmup_size > len(warmup_samples):
        raise ValueError(
            "run_all.eval_schedule_warmup_samples must be in "
            f"[1, {len(warmup_samples)}], got {schedule_warmup_size}"
        )
    return warmup_samples[:schedule_warmup_size]


def _timing_stats_from_seconds(times: list[float]) -> dict:
    if not times:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    mean = sum(times) / len(times)
    if len(times) < 2:
        std = 0.0
    else:
        std = statistics.stdev(times)
    return {"n": len(times), "mean": mean, "std": std}


def aggregate_inference_timing_by_strategy(timing_log_path: str) -> dict[str, dict]:
    """Bucket `inference_timing.jsonl` by tag suffix."""
    buckets = {s: [] for s in _STRATEGY_SUFFIXES}
    if os.path.isfile(timing_log_path):
        with open(timing_log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tag = row.get("tag", "")
                suf = tag.rsplit("_", 1)[-1]
                if suf in buckets:
                    buckets[suf].append(float(row["seconds"]))
    return {s: _timing_stats_from_seconds(times) for s, times in buckets.items()}


def append_inference_timing_to_eval_results(eval_results_path: str, strategy: str, stats: dict) -> None:
    """Append human-readable SMC wall-time summary (inference.main) to metrics file."""
    lines = [
        "\n",
        f"--- SMC inference wall time ({strategy}, mean/std over inference.main calls) ---\n",
    ]
    if stats["n"] == 0:
        lines.append("runs: 0 (no matching records in inference_timing.jsonl)\n")
    else:
        lines.append(f"runs: {stats['n']}\n")
        lines.append(f"mean_seconds_per_run: {stats['mean']:.3f}\n")
        lines.append(f"std_seconds_per_run: {stats['std']:.3f}\n")
    with open(eval_results_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


# def _guidance_steps_first_k(num_inference_steps: int, k: int) -> set[int]:
#     """First k diffusion steps in execution order → largest k timestep indices (T-k .. T-1)."""
#     k = min(k, num_inference_steps)
#     return set(range(num_inference_steps - k, num_inference_steps))


# def _guidance_steps_last_k(num_inference_steps: int, k: int) -> set[int]:
#     """Last k steps before finish → smallest k timestep indices (0 .. k-1)."""
#     k = min(k, num_inference_steps)
#     return set(range(k))


def write_text_and_abc(tag: str, samples: list[dict], prompts_full: list[str], tokenizer, max_len: int) -> None:
    path_text = f"text_samples_{tag}.jsonl"
    with open(path_text, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    _, prompt_to_data = process_file(
        file=path_text,
        prompts=prompts_full,
        expected_per=None,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    ssd_path = f"abc_ssdlm_gen_{tag}.jsonl"
    with open(ssd_path, "w") as f:
        for _, data in prompt_to_data.items():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def calculate_skipped_guidance_sum(T, T_prime, series):
    """
    T: 전체 타임스텝 수 (0, 1, ..., T-1) [cite: 80]
    T_prime: 가이드가 수행되는 시점의 집합 (list or set) [cite: 80]
    V_hat_series: warmup 시뮬레이션을 통해 얻은 각 시점별 V_hat 값들 (리스트/딕셔너리) [cite: 139]
    """
    total_sum = 0
    # 논문의 정의에 따라 T_prime_cup_T를 구성 (T' U {T}) 
    T_prime_cup_T = sorted(list(T_prime) + [T])
    
    # 가이드가 생략된 시점 (T - T') 탐색 
    all_timesteps = set(range(T))
    skipped_timesteps = all_timesteps - set(T_prime) # t \in T - T' [cite: 133]
    
    for t in skipped_timesteps:
        # ceil(t) 찾기: t보다 크거나 같은 T_prime_cup_T의 최소 원소 
        # (DP logic에서 V_ceil(t)를 계산하는 핵심 부분) [cite: 140, 551]
        ceil_t = min([tp for tp in T_prime_cup_T if tp >= t])
        
        # 해당 시점의 V_hat 값을 누적 
        total_sum += series[ceil_t]
        
    return total_sum


def _inference_unit_seed(base_seed: int, prompt_idx: int, run_idx: int) -> int:
    """동일 (base_seed, prompt_idx, run_idx)에 대해 전략마다 같은 시작 RNG를 쓰기 위한 정수 시드."""
    x = (int(base_seed) * 1_000_003 + prompt_idx * 1_009 + run_idx * 65_537) % (2**31 - 1)
    return x if x != 0 else 1


def _set_rng_before_strategy_inference(config, prompt_idx: int, run_idx: int) -> None:
    """각 전략 실행 직전에 호출: 같은 (prompt, run)이면 전략 간 동일 시드."""
    if config.run_all.seed is None:
        return
    unit_seed = _inference_unit_seed(int(config.run_all.seed), prompt_idx, run_idx)
    lightning.seed_everything(unit_seed, workers=True)


def _config_path(config, original_cwd: str, name: str) -> str | None:
    return resolve_path(getattr(config.run_all, name, None), original_cwd)


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(config):
    # Resolve relative checkpoint path (Hydra chdir breaks relative paths)
    original_cwd = hydra.utils.get_original_cwd()
    if config.ft_model.ckpt_path and not os.path.isabs(config.ft_model.ckpt_path):
        config.ft_model.ckpt_path = os.path.join(original_cwd, config.ft_model.ckpt_path)

    # Read all prompts from the prompt file
    prompt_file = resolve_path(str(config.run_all.prompt_file), original_cwd)
    assert prompt_file is not None
    with open(prompt_file, 'r') as f:
        prompts_from_file = [json.loads(l) for l in f]
        prompts_from_file = [p["context_string"] for p in prompts_from_file]

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-large", **pretrained_kwargs("roberta-large")
    )
    prompts_full = get_possible_prompts(prompt_file)
    max_len = 1000
    timing_log_path = os.path.join(os.getcwd(), "inference_timing.jsonl")
    append_timing = bool(getattr(config.run_all, "append_timing_log", False))
    os.makedirs(os.path.dirname(os.path.abspath(timing_log_path)), exist_ok=True)
    with open(timing_log_path, "a" if append_timing else "w", encoding="utf-8") as f:
        pass  # fresh log unless appending
    # grad/reverse with guidance_steps=None runs guidance at every diffusion timestep.
    guidance_steps = None
    T = int(config.smc.num_inference_steps)
    T_ = int(config.run_all.num_guidance_steps)
    if not (0 < T_ <= T):
        raise ValueError(
            f"run_all.num_guidance_steps must satisfy 1 <= T' <= T where T=num_inference_steps={T}, got {T_}"
        )
    selected_strategies = _selected_strategies(config)

    # Static schedules are available without all-step value traces.
    trace_to_dp_start_t = time.perf_counter()
    schedule_by_strategy = {
        "allstep": list(range(T)),
        "uniformstep": uniform_steps(T, T_),
        **{
            f"interval{interval}": interval_steps(T, T_, interval)
            for interval in range(1, 6)
        },
    }
    warmup_trace_manifest = None
    warmup_trace_dir = None
    schedule_warmup_samples = []
    mean_series = None
    value_schedule_strategies = {"vista", "topv", "topdv"}
    if selected_strategies & value_schedule_strategies:
        warmup_trace_manifest = _config_path(
            config, original_cwd, "warmup_trace_manifest"
        ) or _config_path(config, original_cwd, "allstep_cache_manifest")
        warmup_trace_dir = _config_path(
            config, original_cwd, "warmup_trace_dir"
        ) or _config_path(config, original_cwd, "allstep_cache_dir")
        warmup_trace_glob = str(
            getattr(
                config.run_all,
                "warmup_trace_glob",
                "reward_trace_*allstep*.jsonl",
            )
        )
        warmup_samples = load_warmup_samples(
            trace_manifest=warmup_trace_manifest,
            trace_dir=warmup_trace_dir,
            trace_glob=warmup_trace_glob,
        )
        schedule_warmup_samples = _schedule_warmup_samples(
            config, warmup_samples
        )
        series_matrix = np.asarray(
            [
                ensure_boundary_series(sample.series, T)
                for sample in schedule_warmup_samples
            ],
            dtype=float,
        )
        if series_matrix.ndim != 2 or series_matrix.shape[1] <= T:
            raise ValueError(
                f"Warmup series shape {series_matrix.shape} is incompatible with T={T}"
            )
        mean_series = np.mean(series_matrix, axis=0)
        schedule_by_strategy.update(
            {
                "vista": dp_timestep_list_vista(mean_series, T, T_),
                "topv": top_value_steps(mean_series, T, T_),
                "topdv": top_delta_value_steps(mean_series, T, T_),
            }
        )
    trace_to_dp_elapsed_s = time.perf_counter() - trace_to_dp_start_t

    # 결과 timesteps는 논문의 T' 집합이 됨
    with open(f"dp_timestep.txt", "w") as f:
        for strategy, steps in schedule_by_strategy.items():
            f.write(f"{strategy}_steps: {sorted(steps)}\n")
        f.write(f"\ntrace_to_dp_elapsed_s: {trace_to_dp_elapsed_s:.6f}")
        f.write(f"\nwarmup_trace_manifest: {warmup_trace_manifest}")
        f.write(f"\nwarmup_trace_dir: {warmup_trace_dir}")
        f.write(
            "\neval_schedule_warmup_tags: "
            f"{[sample.tag for sample in schedule_warmup_samples]}"
        )
        if mean_series is not None:
            for strategy, steps in schedule_by_strategy.items():
                res = calculate_skipped_guidance_sum(T, steps, mean_series)
                f.write(f"\n{strategy}_score: {res}")

    # actual test: 전략 실행 직전에만 시드 고정 → 동일 (prompt, run)에서 전략 간 공정 비교
    if config.run_all.seed is not None and getattr(config.run_all, "cudnn_deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start_idx = int(getattr(config.run_all, "start_prompt_idx", 0))
    end_raw = getattr(config.run_all, "end_prompt_idx", None)
    default_eval_prompts = int(getattr(config.run_all, "eval_num_prompts", 15))
    end_idx = start_idx + default_eval_prompts if end_raw is None else int(end_raw)
    start_idx = max(0, min(start_idx, len(prompts_from_file)))
    end_idx = max(start_idx, min(end_idx, len(prompts_from_file)))
    runs_per_prompt = int(getattr(config.run_all, "runs_per_prompt", 1))
    start_run_idx = int(getattr(config.run_all, "start_run_idx", 0))
    end_run_idx = getattr(config.run_all, "end_run_idx", None)
    end_run_idx = runs_per_prompt if end_run_idx is None else int(end_run_idx)
    start_run_idx = max(0, min(start_run_idx, runs_per_prompt))
    end_run_idx = max(start_run_idx, min(end_run_idx, runs_per_prompt))
    guidance_steps_by_strategy = {
        **{strategy: set(steps) for strategy, steps in schedule_by_strategy.items()},
    }

    for prompt_idx in range(start_idx, end_idx):
        prompt_text = prompts_from_file[prompt_idx]
        for run_idx in tqdm(list(range(start_run_idx, end_run_idx))):
            for strategy in _ALL_STRATEGIES:
                if strategy not in selected_strategies:
                    continue
                tag = f"p{prompt_idx}_r{run_idx}_{strategy}"
                if bool(getattr(config.run_all, "skip_existing_samples", False)):
                    text_path = os.path.join(os.getcwd(), f"text_samples_{tag}.jsonl")
                    abc_path = os.path.join(os.getcwd(), f"abc_ssdlm_gen_{tag}.jsonl")
                    if os.path.isfile(text_path) and os.path.isfile(abc_path):
                        print(f"[eval] skip existing: {tag}")
                        continue
                os.environ["SMC_TRACE_TAG"] = tag
                guidance_steps = set(guidance_steps_by_strategy[strategy])
                config.smc.prompt_text = prompt_text
                print(f"Running {strategy} SMC for prompt: {prompt_text}")
                _set_rng_before_strategy_inference(config, prompt_idx, run_idx)
                text_samples, toxicity_scores, _ = run_inference_with_timing(
                    config, guidance_steps, timing_log_path, prompt_idx, run_idx
                )
                samples = _samples_from_inference(text_samples, toxicity_scores, prompt_text, config)
                write_text_and_abc(
                    tag, samples, prompts_full, tokenizer, max_len
                )

    os.environ.pop("SMC_TRACE_TAG", None)

    cwd = os.getcwd()
    prompt_order = get_possible_prompts(prompt_file)
    timing_by_strategy = aggregate_inference_timing_by_strategy(timing_log_path)
    for strategy in _ALL_STRATEGIES:
        if strategy not in selected_strategies:
            continue
        merged = os.path.join(cwd, f"abc_ssdlm_gen_merged_{strategy}.jsonl")
        eval_results_path = os.path.join(cwd, f"eval_results_{strategy}.txt")
        merge_abc_ssdlm_by_strategy(cwd, strategy, prompt_order, merged)
        run_evaluate(
            merged,
            f"eval_results_{strategy}.txt",
            int(config.smc.num_particles),
        )
        append_inference_timing_to_eval_results(
            eval_results_path, strategy, timing_by_strategy[strategy]
        )


if __name__ == '__main__':
    main()
    
