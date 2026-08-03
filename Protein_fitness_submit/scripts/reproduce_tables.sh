#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${1:-0}"
output_root="${OUTPUT_ROOT:-$package_root/outputs}"
python_bin="${PYTHON:-python}"

cd "$package_root"
mkdir -p "$output_root/logs"

for backbone in mdlm udlm; do
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u scripts/run_protein_fitness.py \
    --backbone "$backbone" --warmup-only --output-root "$output_root" \
    2>&1 | tee "$output_root/logs/${backbone}_warmup.log"

  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u scripts/run_protein_fitness.py \
    --backbone "$backbone" --budgets 16 32 128 --output-root "$output_root" \
    2>&1 | tee "$output_root/logs/${backbone}_metrics.log"

  for budget in 16 32 128; do
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u scripts/benchmark_timing.py \
      --backbone "$backbone" --budget "$budget" --output-root "$output_root" \
      2>&1 | tee "$output_root/logs/${backbone}_b${budget}_timing.log"
  done
done

"$python_bin" scripts/aggregate_tables.py \
  --output-root "$output_root" --output-dir "$output_root/tables"
