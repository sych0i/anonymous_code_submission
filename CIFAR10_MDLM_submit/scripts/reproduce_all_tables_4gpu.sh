#!/usr/bin/env bash
set -euo pipefail

# Run the complete table sweep over four GPUs. Each worker writes disjoint
# seed files, so the sweep is safe to resume by rerunning this script.

mkdir -p eval_runs/tables/logs

run_worker() {
  local device="$1"
  local seeds="$2"
  local label="$3"
  SEEDS="$seeds" AGGREGATE=0 \
    bash scripts/reproduce_cifar10_mdlm.sh tables "$device" \
    > "eval_runs/tables/logs/${label}.log" 2>&1
}

run_worker cuda:0 "1 5 9" gpu0 &
pid0=$!
run_worker cuda:1 "2 6 10" gpu1 &
pid1=$!
run_worker cuda:2 "3 7" gpu2 &
pid2=$!
run_worker cuda:3 "4 8" gpu3 &
pid3=$!

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  wait "$pid" || status=1
done
if [[ "$status" != "0" ]]; then
  echo "At least one GPU worker failed; inspect eval_runs/tables/logs/." >&2
  exit "$status"
fi

python scripts/aggregate_tables.py \
  --input-root eval_runs/tables \
  --output-dir eval_runs/tables/summary
