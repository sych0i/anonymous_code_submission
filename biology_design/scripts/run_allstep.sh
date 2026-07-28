#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/logs_eval/reproduction/allstep}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON (run bash env.sh)" >&2
  exit 1
fi
if [[ -d "$OUT_DIR" ]] &&
   [[ -n "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Output directory is not empty: $OUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$ROOT/.cache/matplotlib"
export DRAKES_DATA_ROOT="$ROOT/model_weights/data_and_model"
export XDG_CACHE_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=

"$PYTHON" -u eval_allstep.py \
  "cuda_device=$GPU" \
  "hydra.run.dir=$OUT_DIR"
