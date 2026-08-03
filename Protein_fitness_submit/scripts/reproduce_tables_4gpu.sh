#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${OUTPUT_ROOT:-$package_root/outputs}"
python_bin="${PYTHON:-python}"
gpu0="${GPU0:-0}"
gpu1="${GPU1:-1}"
gpu2="${GPU2:-2}"
gpu3="${GPU3:-3}"

cd "$package_root"
mkdir -p "$output_root/logs"

run_logged() {
  local gpu="$1"
  local log="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$@" >"$log" 2>&1
}

# A Vhat trace is shared by all three budgets of the same backbone.
run_logged "$gpu0" "$output_root/logs/mdlm_warmup.log" \
  scripts/run_protein_fitness.py --backbone mdlm --warmup-only --output-root "$output_root" &
pid_mdlm=$!
run_logged "$gpu2" "$output_root/logs/udlm_warmup.log" \
  scripts/run_protein_fitness.py --backbone udlm --warmup-only --output-root "$output_root" &
pid_udlm=$!
wait "$pid_mdlm"
wait "$pid_udlm"

# Sparse budgets are split across four devices; each backbone's full row follows its T'=32 shard.
run_logged "$gpu0" "$output_root/logs/mdlm_b16_metrics.log" \
  scripts/run_protein_fitness.py --backbone mdlm --budgets 16 --output-root "$output_root" &
pid0=$!
run_logged "$gpu1" "$output_root/logs/mdlm_b32_full_metrics.log" \
  scripts/run_protein_fitness.py --backbone mdlm --budgets 32 128 --output-root "$output_root" &
pid1=$!
run_logged "$gpu2" "$output_root/logs/udlm_b16_metrics.log" \
  scripts/run_protein_fitness.py --backbone udlm --budgets 16 --output-root "$output_root" &
pid2=$!
run_logged "$gpu3" "$output_root/logs/udlm_b32_full_metrics.log" \
  scripts/run_protein_fitness.py --backbone udlm --budgets 32 128 --output-root "$output_root" &
pid3=$!
wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"

run_logged "$gpu0" "$output_root/logs/mdlm_b16_timing.log" \
  scripts/benchmark_timing.py --backbone mdlm --budget 16 --output-root "$output_root" &
pid0=$!
run_logged "$gpu1" "$output_root/logs/mdlm_b32_timing.log" \
  scripts/benchmark_timing.py --backbone mdlm --budget 32 --output-root "$output_root" &
pid1=$!
run_logged "$gpu2" "$output_root/logs/udlm_b16_timing.log" \
  scripts/benchmark_timing.py --backbone udlm --budget 16 --output-root "$output_root" &
pid2=$!
run_logged "$gpu3" "$output_root/logs/udlm_b32_timing.log" \
  scripts/benchmark_timing.py --backbone udlm --budget 32 --output-root "$output_root" &
pid3=$!
wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"

# Full timing has one policy per backbone and is inexpensive enough to run as a final pair.
run_logged "$gpu0" "$output_root/logs/mdlm_full_timing.log" \
  scripts/benchmark_timing.py --backbone mdlm --budget 128 --output-root "$output_root" &
pid0=$!
run_logged "$gpu2" "$output_root/logs/udlm_full_timing.log" \
  scripts/benchmark_timing.py --backbone udlm --budget 128 --output-root "$output_root" &
pid2=$!
wait "$pid0"
wait "$pid2"

"$python_bin" scripts/aggregate_tables.py \
  --output-root "$output_root" --output-dir "$output_root/tables"
