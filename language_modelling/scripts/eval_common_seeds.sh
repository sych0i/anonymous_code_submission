#!/bin/bash

# Shared seed override for eval.py and sample_allstep_traces.py.
# Override SMC_EVAL_SEED before sourcing this file if a run needs a different
# deterministic seed bank.
# Set SMC_ALLSTEP_MANIFEST or SMC_ALLSTEP_DIR to reuse saved all-step samples.

SMC_EVAL_SEED="${SMC_EVAL_SEED:-42}"

COMMON_SEED_ARGS=(
    "run_all.seed=${SMC_EVAL_SEED}"
)

COMMON_ALLSTEP_ARGS=()
if [[ -n "${SMC_ALLSTEP_MANIFEST:-}" ]]; then
    COMMON_ALLSTEP_ARGS+=(
        "run_all.warmup_trace_manifest=${SMC_ALLSTEP_MANIFEST}"
        "run_all.allstep_cache_manifest=${SMC_ALLSTEP_MANIFEST}"
    )
fi
if [[ -n "${SMC_ALLSTEP_DIR:-}" ]]; then
    COMMON_ALLSTEP_ARGS+=(
        "run_all.warmup_trace_dir=${SMC_ALLSTEP_DIR}"
        "run_all.allstep_cache_dir=${SMC_ALLSTEP_DIR}"
    )
fi
