# Reference tables

The supplied first and second screenshots are identical Reward tables. Their
values are transcribed in `reference_reward.csv`. The third screenshot is in
`reference_unique_success_lpips003.csv`.

These CSVs are reference targets only. Fresh results are produced from raw
per-seed comparison JSON files by:

```bash
python scripts/aggregate_tables.py \
  --input-root eval_runs/tables \
  --output-dir eval_runs/tables/summary
```

The generated `tables.md` contains three independently computed metrics:

1. Reward.
2. Success rate.
3. Unique success count at LPIPS ≤ 0.03 and reward > 0.5.

All cells report arithmetic mean ± sample standard deviation over seeds 1–10.
Each seed-level cell is based on 100 final particles.
