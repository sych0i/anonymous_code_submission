"""Generate the 30 full-guidance traces used to estimate value schedules."""

from __future__ import annotations

import json
import re
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import dataloader_gosai
from eval_common import (
    allstep_schedule,
    append_metrics_row,
    append_timing_row,
    cfg_select,
    configure_logger,
    eval_get_metrics,
    find_allstep_traces,
    load_components,
    mean_series_from_traces,
    paper_T,
    plot_reward_trace,
    run_smc_once,
    save_mean_series,
    save_samples,
    write_metrics_summary,
)


def _trace_iteration(path: Path) -> int:
    match = re.search(r"reward_trace_itr(\d+)_allsteps\.jsonl$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse iteration from {path}")
    return int(match.group(1))


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(config: DictConfig) -> None:
    out_dir = Path.cwd()
    OmegaConf.save(config.smc, out_dir / "smc_config.yaml", resolve=True)
    logger = configure_logger("eval_allstep", "eval_allstep.log")
    start_itr = int(cfg_select(config, "eval_allstep.start_itr", 0))
    num_runs = int(cfg_select(config, "eval_allstep.num_runs", 30))
    end_itr = start_itr + num_runs
    seed_base = int(cfg_select(config, "eval_allstep.seed_base", 0))
    T = paper_T(config)
    guidance_steps = allstep_schedule(T)
    logger.info(
        "full guidance: iterations [%d,%d), seeds [%d,%d], T=%d",
        start_itr,
        end_itr,
        seed_base + start_itr,
        seed_base + end_itr - 1,
        T,
    )
    components = load_components(
        config,
        logger=logger,
        reproduce_allstep_numerics=True,
        unique_hamming_threshold=float(
            cfg_select(
                config,
                "eval_allstep.unique_hamming_threshold",
                0.0,
            )
        ),
    )
    timing_path = out_dir / "inference_timing.csv"
    metrics_path = out_dir / "eval_metrics_per_run.csv"
    metric_rows: list[dict[str, object]] = []

    for itr in range(start_itr, end_itr):
        tag = f"itr{itr}_allsteps"
        samples, elapsed_s, trace_path = run_smc_once(
            components,
            config,
            guidance_steps=guidance_steps,
            tag=tag,
            seed=seed_base + itr,
        )
        append_timing_row(
            timing_path,
            {
                "tag": tag,
                "method": "allsteps",
                "itr": itr,
                "elapsed_s": elapsed_s,
            },
        )
        # if bool(cfg_select(config, "eval_allstep.plot_reward_traces", True)):
        #     plot_reward_trace(
        #         trace_path,
        #         out_dir / f"reward_trace_{tag}.png",
        #         logger,
        #     )
        if bool(cfg_select(config, "eval_allstep.save_samples", True)):
            save_samples(samples, tag, out_dir)
        sequences = dataloader_gosai.batch_dna_detokenize(
            samples.detach().cpu().numpy()
        )
        metrics = eval_get_metrics(sequences, components)
        row = {
            "method": "allsteps",
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

    trace_paths = find_allstep_traces(out_dir)
    trace_iterations = [_trace_iteration(path) for path in trace_paths]
    expected_iterations = list(range(start_itr, end_itr))
    if trace_iterations != expected_iterations:
        raise ValueError(
            "All-step traces are not the expected contiguous run: "
            f"{trace_iterations} != {expected_iterations}"
        )
    mean_series = mean_series_from_traces(
        trace_paths,
        gamma=float(cfg_select(config, "smc.gamma", 1.0)),
    )
    save_mean_series(out_dir, mean_series)
    write_metrics_summary(out_dir / "eval_metrics_summary.csv", metric_rows)

    manifest = {
        "script": "eval_allstep.py",
        "trace_schema_version": 2,
        "num_runs": len(trace_paths),
        "start_itr": start_itr,
        "end_itr": end_itr,
        "seed_base": seed_base,
        "T": T,
        "num_inference_steps": T,
        "num_guidance_steps": len(guidance_steps),
        "gamma": float(cfg_select(config, "smc.gamma", 1.0)),
        "particle_aggregation": "weighted",
        "proposal_type": "reduced_SMC_grad",
        "propagation_model": "pretrained",
        "kl_weight": float(cfg_select(config, "smc.kl_weight")),
        "phi": int(cfg_select(config, "smc.phi")),
        "tau": float(cfg_select(config, "smc.tau")),
        "num_particles": int(cfg_select(config, "smc.num_particles")),
        "resampling_frequency": int(
            cfg_select(config, "smc.resampling.frequency")
        ),
        "resampling_ess_threshold": float(
            cfg_select(config, "smc.resampling.ess_threshold")
        ),
        "reward_type": "atac",
        "unique_hamming_threshold": float(
            cfg_select(
                config,
                "eval_allstep.unique_hamming_threshold",
                0.0,
            )
        ),
        "guidance_steps": sorted(guidance_steps),
        "mean_series_state_timesteps": list(range(T + 1)),
        "trace_files": [path.name for path in trace_paths],
        "mean_series_files": ["mean_series.npy", "mean_series.csv"],
    }
    (out_dir / "allstep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %d full-guidance traces to %s", len(trace_paths), out_dir)


if __name__ == "__main__":
    main()
