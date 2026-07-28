"""Evaluate VISTA and retained sparse baselines from full-guidance traces."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import dataloader_gosai
from eval_common import (
    DEFAULT_EVAL_METHODS,
    append_metrics_row,
    append_timing_row,
    build_schedule_map,
    calculate_skipped_guidance_sum,
    cfg_select,
    configure_logger,
    eval_get_metrics,
    find_allstep_traces,
    guidance_count,
    load_components,
    mean_series_from_traces,
    paper_T,
    parse_method_list,
    plot_reward_trace,
    plot_value_trace,
    run_smc_once,
    save_mean_series,
    save_samples,
    validate_guidance_count,
    validate_mean_series_length,
    write_metrics_summary,
    write_schedule_json,
    write_schedule_scores_csv,
)


def _resolve_run_dir(raw_path: object) -> Path:
    if raw_path in (None, "", "null", "None"):
        raise ValueError(
            "eval.allstep_run_dir is required; run eval_allstep.py first"
        )
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path(get_original_cwd()).resolve() / path
    return path.resolve()


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "allstep_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing all-step manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_provenance(
    config: DictConfig,
    manifest: dict,
    traces: list[Path],
) -> None:
    expected = {
        "T": paper_T(config),
        "num_inference_steps": paper_T(config),
        "seed_base": int(cfg_select(config, "eval_allstep.seed_base", 0)),
        "proposal_type": "reduced_SMC_grad",
        "propagation_model": "pretrained",
        "reward_type": "atac",
        "phi": int(cfg_select(config, "smc.phi")),
        "num_particles": int(cfg_select(config, "smc.num_particles")),
        "resampling_frequency": int(
            cfg_select(config, "smc.resampling.frequency")
        ),
    }
    for field, current in expected.items():
        recorded = manifest.get(field)
        if recorded is not None and str(recorded) != str(current):
            raise ValueError(
                f"All-step manifest {field}={recorded!r}, "
                f"but current configuration uses {current!r}"
            )
    for field, current in {
        "kl_weight": float(cfg_select(config, "smc.kl_weight")),
        "tau": float(cfg_select(config, "smc.tau")),
        "resampling_ess_threshold": float(
            cfg_select(config, "smc.resampling.ess_threshold")
        ),
    }.items():
        recorded = manifest.get(field)
        if recorded is not None and abs(float(recorded) - current) > 1.0e-12:
            raise ValueError(
                f"All-step manifest {field}={recorded}, "
                f"but current configuration uses {current}"
            )
    manifest_runs = manifest.get("num_runs")
    if manifest_runs is not None and int(manifest_runs) != len(traces):
        raise ValueError(
            f"Manifest records {manifest_runs} traces, found {len(traces)}"
        )


def _save_policy_schedules(
    out_dir: Path,
    *,
    T: int,
    T_prime: int,
    schedules: dict[str, set[int]],
    scores: dict[str, float],
) -> None:
    payload = {
        "T": T,
        "T_prime": T_prime,
        "particle_aggregation": "weighted",
        "schedule_scores": {
            method: float(score) for method, score in scores.items()
        },
        "schedules": {
            method: sorted(int(timestep) for timestep in steps)
            for method, steps in schedules.items()
        },
    }
    (out_dir / f"policy_schedules_T{T_prime}.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(config: DictConfig) -> None:
    out_dir = Path.cwd()
    OmegaConf.save(config.smc, out_dir / "smc_config.yaml", resolve=True)
    logger = configure_logger("eval", "eval.log")
    allstep_dir = _resolve_run_dir(
        cfg_select(config, "eval.allstep_run_dir", None)
    )
    manifest = _load_manifest(allstep_dir)
    traces = find_allstep_traces(allstep_dir)
    _validate_provenance(config, manifest, traces)

    T = paper_T(config)
    T_prime = guidance_count(config)
    validate_guidance_count(T, T_prime)
    warmup_runs = int(cfg_select(config, "schedule.warmup_runs", 3))
    if warmup_runs < 1 or warmup_runs > len(traces):
        raise ValueError(
            f"schedule.warmup_runs={warmup_runs}, found {len(traces)} traces"
        )
    warmup_traces = traces[:warmup_runs]
    mean_series = mean_series_from_traces(
        warmup_traces,
        gamma=float(cfg_select(config, "smc.gamma", 1.0)),
    )
    validate_mean_series_length(T, mean_series)
    save_mean_series(out_dir, mean_series)
    plot_value_trace(mean_series, out_dir / "value_trace_raw.png")

    schedules = build_schedule_map(T, T_prime, mean_series)
    scores = {
        method: calculate_skipped_guidance_sum(
            T,
            steps,
            mean_series,
        )
        for method, steps in schedules.items()
    }
    trace_names = [path.name for path in warmup_traces]
    write_schedule_json(
        out_dir / f"vista_schedule_T{T_prime}.json",
        name="vista",
        T=T,
        T_prime=T_prime,
        timesteps=sorted(schedules["vista"]),
        source_run_dir=allstep_dir,
        scores=scores,
        trace_files=trace_names,
        mean_series=mean_series,
    )
    write_schedule_scores_csv(
        out_dir / "schedule_scores.csv",
        scores=scores,
        T=T,
        T_prime=T_prime,
        trace_files=trace_names,
    )
    _save_policy_schedules(
        out_dir,
        T=T,
        T_prime=T_prime,
        schedules=schedules,
        scores=scores,
    )

    methods = parse_method_list(
        cfg_select(config, "eval.methods", None),
        default=DEFAULT_EVAL_METHODS,
    )
    unknown = [method for method in methods if method not in schedules]
    if unknown:
        raise ValueError(
            f"Unknown methods {unknown}; available={list(schedules)}"
        )
    for method in methods:
        if len(schedules[method]) != T_prime:
            raise ValueError(
                f"{method} has {len(schedules[method])} steps, expected {T_prime}"
            )
        logger.info(
            "[%s] guidance steps=%s schedule_score=%.6e",
            method,
            sorted(schedules[method]),
            scores[method],
        )

    num_runs = int(cfg_select(config, "eval.num_runs", 30))
    seed_base = int(cfg_select(config, "eval.seed_base", 0))
    if seed_base != int(cfg_select(config, "eval_allstep.seed_base", 0)):
        raise ValueError("eval.seed_base must match eval_allstep.seed_base")
    components = load_components(config, logger=logger)
    metrics_path = out_dir / "eval_metrics_per_run.csv"
    timing_path = out_dir / "inference_timing.csv"
    metric_rows: list[dict[str, object]] = []

    for itr in range(num_runs):
        for method in methods:
            tag = f"itr{itr}_{method}"
            samples, elapsed_s, trace_path = run_smc_once(
                components,
                config,
                guidance_steps=schedules[method],
                tag=tag,
                seed=seed_base + itr,
            )
            append_timing_row(
                timing_path,
                {
                    "tag": tag,
                    "method": method,
                    "itr": itr,
                    "elapsed_s": elapsed_s,
                },
            )
            if bool(cfg_select(config, "eval.plot_reward_traces", False)):
                plot_reward_trace(
                    trace_path,
                    out_dir / f"reward_trace_{tag}.png",
                    logger,
                )
            if bool(cfg_select(config, "eval.save_samples", True)):
                save_samples(samples, tag, out_dir)
            sequences = dataloader_gosai.batch_dna_detokenize(
                samples.detach().cpu().numpy()
            )
            metrics = eval_get_metrics(sequences, components)
            row = {
                "method": method,
                "itr": itr,
                "elapsed_s": elapsed_s,
                **metrics,
            }
            metric_rows.append(row)
            append_metrics_row(metrics_path, row)
            logger.info(
                "[%s] %.3fs atac=%.3f unique_atac_count=%.0f",
                tag,
                elapsed_s,
                metrics["atac"],
                metrics["unique_atac_count"],
            )

    write_metrics_summary(out_dir / "eval_metrics_summary.csv", metric_rows)
    logger.info(
        "completed T'=%d, %d methods × %d runs under %s",
        T_prime,
        len(methods),
        num_runs,
        out_dir,
    )


if __name__ == "__main__":
    main()
