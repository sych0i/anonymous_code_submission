#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ALLSTEP_DIR="${ALLSTEP_DIR:-$ROOT/logs_eval/reproduction/allstep}"
RUN_ROOT="${RUN_ROOT:-$ROOT/logs_eval/reproduction/weighted_raw_M3_seed0_hamming0p005_atac_only}"
TPRIMES_RAW="${TPRIMES:-8 16 24}"
GPUS_RAW="${GPUS:-0 1 2}"
read -r -a TPRIMES_LIST <<< "${TPRIMES_RAW//,/ }"
read -r -a GPUS_LIST <<< "${GPUS_RAW//,/ }"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON (run bash env.sh)" >&2
  exit 1
fi
if [[ ! -f "$ALLSTEP_DIR/allstep_manifest.json" ]]; then
  echo "Missing all-step result: $ALLSTEP_DIR/allstep_manifest.json" >&2
  exit 1
fi
if ((${#TPRIMES_LIST[@]} != ${#GPUS_LIST[@]})); then
  echo "TPRIMES and GPUS must have the same number of entries" >&2
  exit 1
fi

declare -A SEEN_BUDGETS=()
declare -A GPU_BUDGETS=()
for index in "${!TPRIMES_LIST[@]}"; do
  budget="${TPRIMES_LIST[$index]}"
  gpu="${GPUS_LIST[$index]}"
  if [[ ! "$budget" =~ ^[0-9]+$ ]] || ((budget < 1 || budget > 128)); then
    echo "Invalid T' value: $budget" >&2
    exit 1
  fi
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index: $gpu" >&2
    exit 1
  fi
  if [[ -n "${SEEN_BUDGETS[$budget]:-}" ]]; then
    echo "Each T' value must be unique" >&2
    exit 1
  fi
  SEEN_BUDGETS[$budget]=1
  GPU_BUDGETS[$gpu]="${GPU_BUDGETS[$gpu]:-} $budget"
done

mkdir -p "$RUN_ROOT/queues" "$ROOT/.cache/matplotlib"
export DRAKES_DATA_ROOT="$ROOT/model_weights/data_and_model"
export XDG_CACHE_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=

run_one() {
  local budget="$1"
  local gpu="$2"
  local out_dir="$RUN_ROOT/Tprime${budget}"
  local log_path="$RUN_ROOT/queues/Tprime${budget}.log"
  if [[ -d "$out_dir" ]] &&
     [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output directory is not empty: $out_dir" >&2
    return 1
  fi
  mkdir -p "$out_dir"
  "$PYTHON" -u eval.py \
    "cuda_device=$gpu" \
    "eval.allstep_run_dir=$ALLSTEP_DIR" \
    "run_all.num_guidance_steps=$budget" \
    "hydra.run.dir=$out_dir" \
    >"$log_path" 2>&1
}

run_queue() {
  local gpu="$1"
  shift
  local budget
  for budget in "$@"; do
    run_one "$budget" "$gpu"
  done
}

declare -a PIDS=()
for gpu in "${!GPU_BUDGETS[@]}"; do
  read -r -a budgets <<< "${GPU_BUDGETS[$gpu]}"
  run_queue "$gpu" "${budgets[@]}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
if ((status != 0)); then
  echo "One or more sparse evaluations failed; inspect $RUN_ROOT/queues" >&2
  exit "$status"
fi
echo "Completed all sparse evaluations under $RUN_ROOT"
