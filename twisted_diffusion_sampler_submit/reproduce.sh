#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3.10}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/eval30_warmup3_budgets_6_12}"
WARMUP_MODE="${WARMUP_MODE:-reuse}"
GPU_IDS_RAW="${GPU_IDS:-${GPU_ID:-0}}"
GPU_IDS_RAW="${GPU_IDS_RAW//,/ }"
read -r -a GPUS <<< "$GPU_IDS_RAW"
BUDGETS_RAW="${BUDGETS:-6 12}"
read -r -a BUDGETS <<< "$BUDGETS_RAW"
WARMUP_M="${WARMUP_M:-3}"
EVAL_RUNS="${EVAL_RUNS:-30}"
CHECK_REFERENCE="${CHECK_REFERENCE:-1}"

if [[ "${#GPUS[@]}" -ne 1 && "${#GPUS[@]}" -ne 3 ]]; then
  echo "GPU_IDS must contain one or three physical GPU IDs" >&2
  exit 2
fi
if [[ "$WARMUP_M" -ne 3 ]]; then
  echo "This submission fixes the target experiment to M=3 warmup runs" >&2
  exit 2
fi

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

MODEL_CONFIG="$ROOT/image_exp/image_confs/mnist_model_and_diffusion_conf.yml"
TASK_CONFIG="$ROOT/image_exp/image_confs/task_class_cond_gen_conf.yml"

case "$WARMUP_MODE" in
  reuse)
    if [[ -n "${WARMUP_FROM:-}" ]]; then
      WARMUP="$WARMUP_FROM"
    else
      "$PYTHON" "$ROOT/scripts/prepare_portable_warmup.py"
      WARMUP="$ROOT/artifacts/shared_warmup/summary.json"
    fi
    ;;
  fullstep|fresh)
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" "$ROOT/scripts/run_deterministic.py" \
      --model-config "$MODEL_CONFIG" \
      --task-config "$TASK_CONFIG" \
      --output-dir "$OUTPUT_DIR" \
      --run-name "fullstep_m${WARMUP_M}" \
      --T 100 --T-prime 12 --N 20 --J 1 --M "$WARMUP_M" --target 4 \
      --alpha 2 --ess-threshold 0.98 --partial-resample 20 \
      --value-estimation one_shot \
      --final-transition legacy_deterministic \
      --clip-twisted --no-clip-denoised \
      --seed 0 --warmup-seed-offset 100000 \
      --full-baseline-only --policies full --eval-runs "$WARMUP_M" --eval-start-index 0 \
      --no-save-grids --no-plot-vhat \
      --resume --device cuda:0
    WARMUP="$OUTPUT_DIR/fullstep_m${WARMUP_M}/warmup/summary.json"
    ;;
  *)
    echo "WARMUP_MODE must be 'reuse' (default) or 'fullstep'" >&2
    exit 2
    ;;
esac

COMMON=(
  "$PYTHON" "$ROOT/scripts/run_deterministic.py"
  --model-config "$MODEL_CONFIG"
  --task-config "$TASK_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --warmup-from "$WARMUP"
  --T 100 --N 20 --J 1 --M "$WARMUP_M" --target 4
  --alpha 2 --ess-threshold 0.98 --partial-resample 20
  --value-estimation one_shot
  --final-transition legacy_deterministic
  --clip-twisted --no-clip-denoised
  --seed 0 --warmup-seed-offset 100000
  --eval-runs "$EVAL_RUNS" --eval-start-index 0
  --no-save-grids --no-plot-vhat --resume --device cuda:0
)

run_group() {
  local gpu="$1"
  local group="$2"
  shift 2
  local policies=("$@")
  local budget
  for budget in "${BUDGETS[@]}"; do
    echo "Running T'=$budget group=$group on physical GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "${COMMON[@]}" \
      --T-prime "$budget" \
      --run-name "tp${budget}_${group}" \
      --policies "${policies[@]}"
  done
}

if [[ "${#GPUS[@]}" -eq 1 ]]; then
  run_group "${GPUS[0]}" i123 interval-1 interval-2 interval-3
  run_group "${GPUS[0]}" i45u interval-4 interval-5 uniform
  run_group "${GPUS[0]}" vtt vista top-V top-dV weighted-vista weighted-vista-2 weighted-vista-3 weighted-vista-20
else
  run_group "${GPUS[0]}" i123 interval-1 interval-2 interval-3 &
  pid_i123=$!
  run_group "${GPUS[1]}" i45u interval-4 interval-5 uniform &
  pid_i45u=$!
  run_group "${GPUS[2]}" vtt vista top-V top-dV weighted-vista weighted-vista-2 weighted-vista-3 weighted-vista-20 &
  pid_vtt=$!
  status=0
  wait "$pid_i123" || status=$?
  wait "$pid_i45u" || status=$?
  wait "$pid_vtt" || status=$?
  if [[ "$status" -ne 0 ]]; then
    exit "$status"
  fi
fi

SUMMARY=(
  "$PYTHON" "$ROOT/scripts/summarize_results.py"
  --input-dir "$OUTPUT_DIR"
  --budgets "${BUDGETS[@]}"
)
if [[ "$CHECK_REFERENCE" -eq 1 ]]; then
  SUMMARY+=(--expected "$ROOT/expected_results.json" --strict)
fi
"${SUMMARY[@]}"
