#!/bin/bash
set -euo pipefail

# Paper toxic-text experiment: run the full-step baseline once, estimate each
# schedule from its p0r0/p1r0/p2r0 traces, then sweep sparse-guidance budgets.
# Sparse policies: VISTA, Top-V, Top-dV, Uniform, and five interval windows.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export LD_LIBRARY_PATH=
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/paper_toxic/${RUN_STAMP}}"
ALLSTEP_DIR="${ALLSTEP_DIR:-${RUN_ROOT}/allstep}"
TPRIMES=(5 10 15)

manifest_in_dir() {
    if [[ "$1" = /* ]]; then
        printf '%s/allstep_samples_manifest.json\n' "$1"
    else
        printf '%s/%s/allstep_samples_manifest.json\n' "${PWD}" "$1"
    fi
}

MANIFEST_PATH="${SMC_ALLSTEP_MANIFEST:-}"
if [[ -z "${MANIFEST_PATH}" ]]; then
    ALLSTEP_MANIFEST="$(manifest_in_dir "${ALLSTEP_DIR}")"
    if [[ -f "${ALLSTEP_MANIFEST}" ]]; then
        MANIFEST_PATH="${ALLSTEP_MANIFEST}"
    fi
fi

if [[ -n "${MANIFEST_PATH}" && -f "${MANIFEST_PATH}" ]]; then
    MANIFEST_PATH="$(realpath "${MANIFEST_PATH}")"
    echo "[paper toxic] reuse all-step manifest: ${MANIFEST_PATH}"
else
    "${PYTHON_BIN}" sample_allstep_traces.py \
        smc.proposal_type=grad \
        "cuda_device=${CUDA_DEVICE}" \
        ft_model.ckpt_path=null \
        run_all.runs_per_prompt=2 \
        run_all.eval_num_prompts=15 \
        run_all.allstep_max_samples=auto \
        run_all.evaluate_allstep=true \
        hydra.run.dir="${ALLSTEP_DIR}" \
        hydra.job.chdir=true
    MANIFEST_PATH="$(manifest_in_dir "${ALLSTEP_DIR}")"
fi

SMC_ALLSTEP_MANIFEST="${MANIFEST_PATH}" \
OUTPUT_ROOT="${RUN_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
CUDA_DEVICE="${CUDA_DEVICE}" \
bash scripts/run_eval_tprime_sweep.sh "${TPRIMES[@]}"

echo "Paper toxic-text results: ${RUN_ROOT}"
