"""
Generate reusable all-step guidance traces.

This script only samples with guidance enabled at every diffusion timestep.
It writes reward traces, text samples, SSDL-M evaluation JSONL files, timing,
series arrays, and a manifest that can be reused by eval.py and
eval_warmup_sensitivity.py.

Typical usage:

  LD_LIBRARY_PATH= .venv/bin/python sample_allstep_traces.py \
    run_all.allstep_max_samples=10 \
    run_all.runs_per_prompt=1

Then reuse the saved manifest:

  LD_LIBRARY_PATH= .venv/bin/python eval_warmup_sensitivity.py \
    run_all.warmup_trace_manifest=outputs/.../allstep_samples_manifest.json

  LD_LIBRARY_PATH= .venv/bin/python eval.py \
    run_all.warmup_trace_manifest=outputs/.../allstep_samples_manifest.json \
    run_all.allstep_cache_manifest=outputs/.../allstep_samples_manifest.json
"""

from __future__ import annotations

import csv
import json
import os
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from model_revisions import pretrained_kwargs

_lm_dir = os.path.dirname(os.path.abspath(__file__))
if _lm_dir not in sys.path:
    sys.path.insert(0, _lm_dir)

from eval import (  # noqa: E402
    _inference_unit_seed,
    _samples_from_inference,
    _set_rng_before_strategy_inference,
    aggregate_inference_timing_by_strategy,
    append_inference_timing_to_eval_results,
    merge_abc_ssdlm_by_strategy,
    run_evaluate,
    run_inference_with_timing,
    write_text_and_abc,
)
from evaluation.mdlm_to_eval_format import get_possible_prompts  # noqa: E402
from warmup_trace_utils import (  # noqa: E402
    ensure_boundary_series,
    mean_exp_alpha_r_series_from_trace_jsonl,
    resolve_path,
)


def _read_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, encoding="utf-8") as f:
        return [json.loads(line)["context_string"] for line in f if line.strip()]


def _allstep_max_samples(config, num_selected_units: int) -> int | None:
    raw = getattr(config.run_all, "allstep_max_samples", "auto")
    if raw is None:
        return None
    if str(raw).lower() == "auto":
        return num_selected_units
    return int(raw)


def _write_samples_csv(path: str, records: list[dict]) -> None:
    keys = [
        "sample_id",
        "prompt_idx",
        "run_idx",
        "tag",
        "trace_path",
        "text_path",
        "abc_path",
        "seconds",
        "unit_seed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in keys})


def _load_existing_records(manifest_path: str) -> list[dict]:
    if not os.path.isfile(manifest_path):
        return []
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest.get("samples", [])
    if not isinstance(records, list):
        raise ValueError(f"{manifest_path} has no list-valued 'samples' field")
    return records


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(config) -> None:
    if config.ft_model.ckpt_path and not os.path.isabs(config.ft_model.ckpt_path):
        config.ft_model.ckpt_path = os.path.join(
            hydra.utils.get_original_cwd(),
            config.ft_model.ckpt_path,
        )

    original_cwd = hydra.utils.get_original_cwd()
    prompt_file = resolve_path(str(config.run_all.prompt_file), original_cwd)
    assert prompt_file is not None
    prompts = _read_prompts(prompt_file)
    prompts_full = get_possible_prompts(prompt_file)
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-large", **pretrained_kwargs("roberta-large")
    )

    T = int(config.smc.num_inference_steps)
    guidance_steps = set(range(T))
    append_existing = bool(getattr(config.run_all, "append_timing_log", False))
    timing_log_path = os.path.join(os.getcwd(), "inference_timing.jsonl")
    with open(timing_log_path, "a" if append_existing else "w", encoding="utf-8"):
        pass

    start_idx = int(getattr(config.run_all, "start_prompt_idx", 0))
    end_raw = getattr(config.run_all, "end_prompt_idx", None)
    default_eval_prompts = int(getattr(config.run_all, "eval_num_prompts", 15))
    end_idx = start_idx + default_eval_prompts if end_raw is None else int(end_raw)
    start_idx = max(0, min(start_idx, len(prompts)))
    end_idx = max(start_idx, min(end_idx, len(prompts)))
    runs_per_prompt = int(config.run_all.runs_per_prompt)
    start_run_idx = int(getattr(config.run_all, "start_run_idx", 0))
    end_run_idx = getattr(config.run_all, "end_run_idx", None)
    end_run_idx = runs_per_prompt if end_run_idx is None else int(end_run_idx)
    start_run_idx = max(0, min(start_run_idx, runs_per_prompt))
    end_run_idx = max(start_run_idx, min(end_run_idx, runs_per_prompt))
    prompt_indices = range(start_idx, end_idx)
    run_indices = range(start_run_idx, end_run_idx)
    max_samples = _allstep_max_samples(config, len(prompt_indices) * len(run_indices))
    max_len = int(getattr(config.run_all, "max_eval_len", 1000))
    base_seed = getattr(config.run_all, "seed", None)

    records: list[dict] = _load_existing_records("allstep_samples_manifest.json") if append_existing else []
    for record in records:
        if record.get("series") is not None:
            record["series"] = ensure_boundary_series(record["series"], T)
    series_matrix: list[list[float]] = [
        [float(x) for x in record["series"]]
        for record in records
        if record.get("series") is not None
    ]
    existing_tags = {str(record.get("tag")) for record in records}
    sample_id = max([int(record.get("sample_id", -1)) for record in records], default=-1) + 1
    generated = 0
    for run_idx in run_indices:
        for prompt_idx in prompt_indices:
            if max_samples is not None and generated >= max_samples:
                break
            prompt_text = prompts[prompt_idx]
            tag = f"p{prompt_idx}_r{run_idx}_allstep"
            if tag in existing_tags:
                print(f"[allstep sampling] skip existing: {tag}")
                continue
            os.environ["SMC_TRACE_TAG"] = tag
            config.smc.prompt_text = prompt_text
            _set_rng_before_strategy_inference(config, prompt_idx, run_idx)
            print(
                f"[allstep sampling] sample {sample_id + 1}"
                f"{'' if max_samples is None else f'/{max_samples}'}: {tag}"
            )
            text_samples, toxicity_scores, seconds = run_inference_with_timing(
                config,
                guidance_steps,
                timing_log_path,
                prompt_idx,
                run_idx,
                warmup=True,
            )

            samples = _samples_from_inference(text_samples, toxicity_scores, prompt_text, config)
            write_text_and_abc(tag, samples, prompts_full, tokenizer, max_len)

            trace_path = f"reward_trace_{tag}.jsonl"
            text_path = f"text_samples_{tag}.jsonl"
            abc_path = f"abc_ssdlm_gen_{tag}.jsonl"
            series = ensure_boundary_series(
                mean_exp_alpha_r_series_from_trace_jsonl(trace_path),
                T,
            )
            series_matrix.append(series)
            records.append(
                {
                    "sample_id": sample_id,
                    "prompt_idx": prompt_idx,
                    "run_idx": run_idx,
                    "repeat_idx": run_idx,
                    "tag": tag,
                    "prompt_text": prompt_text,
                    "trace_path": trace_path,
                    "text_path": text_path,
                    "abc_path": abc_path,
                    "seconds": seconds,
                    "unit_seed": None
                    if base_seed is None
                    else _inference_unit_seed(int(base_seed), prompt_idx, run_idx),
                    "series": series,
                }
            )
            sample_id += 1
            generated += 1
        if max_samples is not None and generated >= max_samples:
            break

    os.environ.pop("SMC_TRACE_TAG", None)
    np.save("allstep_series.npy", np.asarray(series_matrix, dtype=float))
    _write_samples_csv("allstep_samples.csv", records)

    manifest = {
        "trace_schema_version": 2,
        "created_in": os.getcwd(),
        "prompt_file": prompt_file,
        "timing_log_path": "inference_timing.jsonl",
        "num_samples": len(records),
        "T": T,
        # This script always guides at every timestep, independently of the
        # sparse-budget default in run_all.num_guidance_steps.
        "T_prime": T,
        "start_prompt_idx": start_idx,
        "end_prompt_idx": end_idx,
        "runs_per_prompt": runs_per_prompt,
        "start_run_idx": start_run_idx,
        "end_run_idx": end_run_idx,
        "allstep_max_samples": max_samples,
        "seed": base_seed,
        "samples": records,
        "config": OmegaConf.to_container(config, resolve=False),
    }
    with open("allstep_samples_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if bool(getattr(config.run_all, "evaluate_allstep", False)):
        merged_path = os.path.join(os.getcwd(), "abc_ssdlm_gen_merged_allstep.jsonl")
        eval_results_path = os.path.join(os.getcwd(), "eval_results_allstep.txt")
        merge_abc_ssdlm_by_strategy(
            os.getcwd(), "allstep", prompts_full, merged_path
        )
        run_evaluate(merged_path, eval_results_path, int(config.smc.num_particles))
        timing_stats = aggregate_inference_timing_by_strategy(timing_log_path)
        append_inference_timing_to_eval_results(
            eval_results_path, "allstep", timing_stats["allstep"]
        )

    print("[allstep sampling] wrote:")
    print("  allstep_samples_manifest.json")
    print("  allstep_samples.csv")
    print("  allstep_series.npy")
    print("  inference_timing.jsonl")


if __name__ == "__main__":
    main()
