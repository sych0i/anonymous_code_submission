#!/bin/bash
set -euo pipefail

# Reuse one all-step manifest and run sparse policies for a sweep of T' values.
# A T' directory is skipped when every expected policy result already exists.
# Usage:
#   nohup scripts/run_eval_tprime_sweep.sh > logs/eval_tprime_sweep.log 2>&1 &
# Optional:
#   SMC_ALLSTEP_MANIFEST=/path/to/allstep_samples_manifest.json scripts/run_eval_tprime_sweep.sh
#   RUN_DATE=20260703 scripts/run_eval_tprime_sweep.sh 5 10 15
#   EXTRA_EVAL_ARGS='run_all.start_run_idx=2 run_all.end_run_idx=3' scripts/run_eval_tprime_sweep.sh
#   ALLOW_OVERWRITE=1 scripts/run_eval_tprime_sweep.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export LD_LIBRARY_PATH=
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/grad_8_0.2_1_4/${RUN_DATE}}"
SMC_ALLSTEP_MANIFEST="${SMC_ALLSTEP_MANIFEST:-}"
LOG_BASENAME="${LOG_BASENAME:-eval.log}"
export SMC_ALLSTEP_MANIFEST
STRATEGIES=(
    vista topv topdv uniformstep
    interval1 interval2 interval3 interval4 interval5
)

if [[ ! -f "${SMC_ALLSTEP_MANIFEST}" ]]; then
    echo "Set SMC_ALLSTEP_MANIFEST to a manifest produced by the current sample_allstep_traces.py." >&2
    echo "Missing SMC_ALLSTEP_MANIFEST: ${SMC_ALLSTEP_MANIFEST:-<unset>}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/eval_common_seeds.sh"

if [[ "$#" -gt 0 ]]; then
    TPRIMES=("$@")
else
    TPRIMES=(5 10 15)
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_EVAL_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "${EXTRA_EVAL_ARGS}"
fi

if [[ "${OUTPUT_ROOT}" = /* ]]; then
    OUTPUT_BASE="${OUTPUT_ROOT}"
else
    OUTPUT_BASE="${PWD}/${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_BASE}"

echo "[tprime sweep] manifest=${SMC_ALLSTEP_MANIFEST}"
echo "[tprime sweep] output_base=${OUTPUT_BASE}"
echo "[tprime sweep] evaluator_cuda=${EVAL_CUDA_VISIBLE_DEVICES}"
echo "[tprime sweep] generation_cuda=${CUDA_DEVICE}"
echo "[tprime sweep] extra_args=${EXTRA_ARGS[*]:-}"
echo "[tprime sweep] values=${TPRIMES[*]}"

is_complete_tprime_dir() {
    local out_dir="$1"
    local strategy
    [[ -s "${out_dir}/dp_timestep.txt" ]] || return 1
    [[ -s "${out_dir}/inference_timing.jsonl" ]] || return 1
    for strategy in "${STRATEGIES[@]}"; do
        [[ -s "${out_dir}/eval_results_${strategy}.txt" ]] || return 1
        [[ -s "${out_dir}/eval_results_${strategy}.txt.ppl-gpt2-xl" ]] || return 1
        [[ -s "${out_dir}/abc_ssdlm_gen_merged_${strategy}.jsonl" ]] || return 1
        [[ "$(find "${out_dir}" -maxdepth 1 -type f -name "text_samples_p*_r*_${strategy}.jsonl" | wc -l)" -eq 30 ]] || return 1
        [[ "$(find "${out_dir}" -maxdepth 1 -type f -name "abc_ssdlm_gen_p*_r*_${strategy}.jsonl" | wc -l)" -eq 30 ]] || return 1
        [[ "$(find "${out_dir}" -maxdepth 1 -type f -name "reward_trace_p*_r*_${strategy}.jsonl" | wc -l)" -eq 30 ]] || return 1
    done
}

for tprime in "${TPRIMES[@]}"; do
    OUT_DIR="${OUTPUT_BASE}/Tprime${tprime}"
    LOG_PATH="${OUT_DIR}/${LOG_BASENAME}"

    if [[ -d "${OUT_DIR}" ]] && is_complete_tprime_dir "${OUT_DIR}"; then
        echo "[tprime sweep] skip completed T'=${tprime}: ${OUT_DIR}"
        continue
    fi
    if [[ -d "${OUT_DIR}" ]] && [[ -z "${ALLOW_OVERWRITE:-}" ]] && [[ -n "$(find "${OUT_DIR}" -mindepth 1 -print -quit)" ]]; then
        echo "[tprime sweep] ${OUT_DIR} is incomplete and non-empty; set ALLOW_OVERWRITE=1 to rerun it." >&2
        exit 1
    fi

    mkdir -p "${OUT_DIR}"
    echo "[tprime sweep] start T'=${tprime} -> ${OUT_DIR}"
    "${PYTHON_BIN}" eval.py \
        "${COMMON_SEED_ARGS[@]}" \
        "${COMMON_ALLSTEP_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        "cuda_device=${CUDA_DEVICE}" \
        "run_all.eval_schedule_warmup_tags=[p0_r0_allstep,p1_r0_allstep,p2_r0_allstep]" \
        "run_all.strategies=[vista,topv,topdv,uniformstep,interval1,interval2,interval3,interval4,interval5]" \
        "run_all.num_guidance_steps=${tprime}" \
        "hydra.run.dir=${OUT_DIR}" \
        "hydra.job.chdir=true" \
        > "${LOG_PATH}" 2>&1
    if ! is_complete_tprime_dir "${OUT_DIR}"; then
        echo "[tprime sweep] T'=${tprime} exited without complete results" >&2
        exit 1
    fi
    echo "[tprime sweep] done T'=${tprime}; log=${LOG_PATH}"
done

echo "[tprime sweep] all done"
