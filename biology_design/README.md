# ATAC-only VISTA reproduction

This directory reproduces the final full-guidance and sparse-guidance ATAC
experiments with only the required runtime code and model assets.

The configuration is fixed to the experiment:

- pretrained Gosai discrete diffusion backbone;
- reduced gradient SMC with 128 reverse steps;
- 8 particles, `phi=4`, `tau=1`, `kl_weight=0.01`;
- partial resampling every guided step at ESS threshold 0.98;
- ATAC classifier channel 1 and ATAC passing threshold 0.5;
- exact-duplicate counting for the retained all-step metrics, and normalized
  Hamming threshold 0.005 (at most one mismatch for 200 bp) for sparse metrics;
- 30 runs with seeds `seed_base + itr = 0, ..., 29`;
- sparse budgets 8, 16, 24, 32, and 40;
- schedules estimated from the first three full-guidance traces;
- residual particle weights are always used for the value estimate.

The retained sparse policies are uniform, VISTA, Top-V, Top-dV, and five
contiguous interval baselines. Top-dV selects the `T'` largest signed
differences `V_t - V_{t+1}`.

## Setup

Python 3.10 and a CUDA 12.1-compatible NVIDIA driver are expected.

```bash
bash env.sh
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The repository already contains the only two required checkpoints. Their
SHA-256 hashes are checked by the test suite. The tests also contain the
canonical weighted first-three-trace value series and verify the weighted
estimator, retained method set, representative VISTA schedules, and VISTA and
Top-dV schedules and scores for all five sparse budgets.

## Reproduce the experiments

First generate the 30 full-guidance runs:

```bash
GPU=0 bash scripts/run_allstep.sh
```

Then run the three sparse budgets in parallel:

```bash
TPRIMES="8 16 24" \
GPUS="0 1 2" \
bash scripts/run_sparse_sweep.sh
```

Change `GPUS` to match the available devices. `TPRIMES` and `GPUS` are paired
by position; budgets assigned to the same GPU run sequentially, so one GPU can
be used with `GPUS="0 0 0"`. Outputs are written under
`logs_eval/reproduction/` by default; `OUT_DIR`, `ALLSTEP_DIR`, and `RUN_ROOT`
can override those locations.

Each output contains per-run samples, reward traces, timing, ATAC metrics, and
summary CSV files. Sparse outputs additionally contain the value series,
policy schedules, and schedule scores.

The legacy all-step folder also contained unrelated likelihood, activity,
k-mer, and motif columns. Those metrics and their multi-gigabyte oracle/data
dependencies are intentionally outside this ATAC-only reproduction. Runtime
measurements can vary with hardware. The ATAC classifier's CUDA backward is
not bitwise deterministic, so regenerated trace weights may differ at small
floating-point scale even with fixed seeds; the canonical schedule regression,
checkpoints, generated samples, and aggregate statistics provide the intended
reproduction target.
