#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/reproduce_cifar10_mdlm.sh smoke [cuda:0]
#   bash scripts/reproduce_cifar10_mdlm.sh tables [cuda:0]
#
# "smoke" runs a small end-to-end check. "tables" runs the complete 10-seed,
# T=1000, M=5, J=5, N=20 sweep used by the submitted tables.

MODE="${1:-smoke}"
DEVICE="${2:-cuda:0}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
AGGREGATE="${AGGREGATE:-1}"
BATCH_SIZE="${BATCH_SIZE:-5}"
CONFIG="artifacts/mdlm_config.yaml"
CHECKPOINT="artifacts/mdlm_best.ckpt"
CLASSIFIER="artifacts/reward_classifier_best.ckpt"

if [[ "$MODE" == "smoke" ]]; then
  python vista_smc.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --classifier-checkpoint "$CLASSIFIER" \
    --tag smoke \
    --schedules "vista,top_dv,uniform" \
    --seed 1 \
    --total-samples 1 \
    --batch-size 1 \
    --num-particles 2 \
    --num-rollout-samples 1 \
    --guidance-steps 4 \
    --num-warmup-runs 1 \
    --ess-threshold 0.95 \
    --partial-resample 1 \
    --steps 16 \
    --proposal-method base \
    --device "$DEVICE" \
    --output-dir eval_runs/smoke
elif [[ "$MODE" == "tables" ]]; then
  SCHEDULES="interval1,interval2,interval3,interval4,interval5,uniform,vista,top_v,top_dv"
  ROOT="eval_runs/tables"
  mkdir -p "$ROOT"

  for seed in $SEEDS; do
    # Full guidance is a single T'=T=1000 reference row, not a sparse
    # policy that needs to be rerun for every guidance budget.
    full_dir="$ROOT/full"
    full_comparison="$full_dir/vista_smc_seed${seed}_full_comparison.json"
    mkdir -p "$full_dir"
    if [[ ! -f "$full_comparison" ]]; then
      python vista_smc.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --classifier-checkpoint "$CLASSIFIER" \
        --tag "seed${seed}_full" \
        --schedules "full" \
        --seed "$seed" \
        --total-samples 5 \
        --batch-size "$BATCH_SIZE" \
        --num-particles 20 \
        --num-rollout-samples 5 \
        --guidance-steps 1000 \
        --num-warmup-runs 5 \
        --ess-threshold 0.95 \
        --partial-resample 10 \
        --steps 1000 \
        --proposal-method base \
        --reward-threshold 0.5 \
        --lpips-threshold 0.03 \
        --device "$DEVICE" \
        --output-dir "$full_dir"
    fi

    # The warm-up value curve is independent of T'. Compute it once at
    # the largest sparse budget, T'=25, then reuse it for the other
    # budgets for this seed.
    source_dir="$ROOT/t25"
    source="$source_dir/vista_smc_seed${seed}_t25_comparison.json"
    mkdir -p "$source_dir"
    if [[ ! -f "$source" ]]; then
      python vista_smc.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --classifier-checkpoint "$CLASSIFIER" \
        --tag "seed${seed}_t25" \
        --schedules "$SCHEDULES" \
        --seed "$seed" \
        --total-samples 5 \
        --batch-size "$BATCH_SIZE" \
        --num-particles 20 \
        --num-rollout-samples 5 \
        --guidance-steps 25 \
        --num-warmup-runs 5 \
        --ess-threshold 0.95 \
        --partial-resample 10 \
        --steps 1000 \
        --proposal-method base \
        --reward-threshold 0.5 \
        --lpips-threshold 0.03 \
        --device "$DEVICE" \
        --output-dir "$source_dir"
    fi

    for budget in 5 15; do
      out_dir="$ROOT/t${budget}"
      comparison="$out_dir/vista_smc_seed${seed}_t${budget}_comparison.json"
      mkdir -p "$out_dir"
      [[ -f "$comparison" ]] && continue
      python vista_smc.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --classifier-checkpoint "$CLASSIFIER" \
        --tag "seed${seed}_t${budget}" \
        --schedules "$SCHEDULES" \
        --seed "$seed" \
        --total-samples 5 \
        --batch-size "$BATCH_SIZE" \
        --num-particles 20 \
        --num-rollout-samples 5 \
        --guidance-steps "$budget" \
        --num-warmup-runs 5 \
        --reuse-warmup-comparison "$source" \
        --ess-threshold 0.95 \
        --partial-resample 10 \
        --steps 1000 \
        --proposal-method base \
        --reward-threshold 0.5 \
        --lpips-threshold 0.03 \
        --device "$DEVICE" \
        --output-dir "$out_dir"
    done
  done

  if [[ "$AGGREGATE" == "1" ]]; then
    python scripts/aggregate_tables.py \
      --input-root "$ROOT" \
      --output-dir "$ROOT/summary"
  fi
else
  echo "Unknown mode: $MODE (expected smoke or tables)" >&2
  exit 2
fi
