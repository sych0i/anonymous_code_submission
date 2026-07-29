# CIFAR-10 MDLM reviewer reproduction

This is the reviewer-facing CIFAR-10 submission package for the MDLM tables in
**Value-Informed Schedule Trimming for Accelerated Sequential Monte Carlo
Guidance in Discrete Diffusion**. It contains the implementation, exact
evaluation configuration, pretrained MDLM and reward-classifier checkpoints,
unit tests, table-generation scripts, and reference values. Training logs,
generated samples, intermediate checkpoints, caches, and the CIFAR-10 archive
are intentionally excluded.

The discrete-diffusion backbone is derived from the codebase accompanying
*Simple Guidance Mechanisms for Discrete Diffusion Models*. See `LICENSE`.

## Contents

- `vista_smc.py`: VISTA schedule search and SMC-base/SMC-grad evaluation.
- `smc.py`: shared SMC utilities and hand-designed schedule definitions.
- `diffusion.py`, `models/unet.py`: CIFAR-10 MDLM implementation.
- `artifacts/mdlm_best.ckpt`: anonymized inference checkpoint (Git LFS);
  model and EMA tensors are unchanged, while optimizer, callback paths, and
  training-runtime state not required by evaluation have been removed.
- `artifacts/reward_classifier_best.ckpt`: reward classifier checkpoint.
- `artifacts/mdlm_config.yaml`: sanitized exact model configuration.
- `scripts/reproduce_cifar10_mdlm.sh`: smoke and complete table sweep.
- `scripts/aggregate_tables.py`: validates all runs and creates the tables.
- `scripts/train_cifar10_mdlm.sh`: optional training-from-scratch command.
- `results/reference_reward.csv`: values in the supplied Reward panels.
- `results/reference_unique_success_lpips003.csv`: values in the supplied
  LPIPS-based unique-success panel.

## Setup

Git LFS is required to obtain the MDLM checkpoint.

```bash
git lfs install
```

Configures Git LFS hooks for the current user so Git can resolve the large
MDLM checkpoint stored as an LFS object.

```bash
git lfs pull
```

Downloads `artifacts/mdlm_best.ckpt` instead of leaving the small LFS pointer
file in the working tree.

```bash
conda env create -f environment.yml
```

Creates the `cifar10-mdlm-submit` Conda environment with the pinned PyTorch,
CUDA, Lightning, LPIPS, and evaluation dependencies.

```bash
conda activate cifar10-mdlm-submit
```

Activates that environment so subsequent `python` and shell commands use the
reproduction dependencies.

```bash
sha256sum -c CHECKSUMS.sha256
```

Verifies that both supplied checkpoints were downloaded completely and have
not been changed.

The first LPIPS evaluation may download the public AlexNet weights used by the
`lpips` package.

## GPU requirements and execution

The supplied configuration was validated on NVIDIA RTX A6000 GPUs with 48 GB
of VRAM. CUDA is required for practical table-scale runtime. The default table
settings process five independent SMC generations concurrently, with 20
particles and five reward rollouts each. Peak memory depends on the CUDA,
PyTorch, and allocator versions.

### One GPU

Use any visible CUDA device by passing its PyTorch device name:

```bash
bash scripts/reproduce_cifar10_mdlm.sh tables cuda:0
```

Runs all three guidance budgets, nine schedules, and ten seeds sequentially on
`cuda:0`. Completed seed/budget JSON files are skipped when the command is
restarted. After all runs finish, it generates the metric tables and the
five-column policy table under
`eval_runs/tables/summary/`.

If the GPU has less memory, reduce only the generation batch size while
preserving the paper's particle and rollout counts:

```bash
BATCH_SIZE=1 bash scripts/reproduce_cifar10_mdlm.sh tables cuda:0
```

Runs the same complete experiment but processes one independent SMC generation
at a time. This reduces peak VRAM without changing the configured number of
particles, rollouts, seeds, schedules, or budgets.

`BATCH_SIZE` must divide the five generations per seed, so supported values are
1 and 5. A smaller value lowers peak VRAM and increases wall-clock time. It can
also change low-level CUDA random-number grouping, so small stochastic
differences from the reference table remain possible.

To expose one selected physical GPU and address it locally as `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=2 \
  bash scripts/reproduce_cifar10_mdlm.sh tables cuda:0
```

Makes only physical GPU 2 visible to the process. Because it becomes the first
visible device, the reproduction script addresses it as `cuda:0`.

### Four GPUs

The convenience launcher expects four visible devices, `cuda:0` through
`cuda:3`:

```bash
bash scripts/reproduce_all_tables_4gpu.sh
```

Starts four independent workers, distributes the ten seeds across
`cuda:0`–`cuda:3`, waits for every worker, and then generates the final
aggregate tables. Per-GPU stdout and stderr are saved under
`eval_runs/tables/logs/`.

This is not DDP: it starts four independent Python processes and assigns
disjoint seeds to each GPU. Therefore, no NCCL communication or cross-GPU
synchronization is required. GPU assignments are:

| Device | Seeds |
|---|---|
| `cuda:0` | 1, 5, 9 |
| `cuda:1` | 2, 6, 10 |
| `cuda:2` | 3, 7 |
| `cuda:3` | 4, 8 |

Worker logs are written to `eval_runs/tables/logs/gpu{0,1,2,3}.log`. If any
worker fails, the launcher exits without aggregating incomplete results.
Rerunning the same command is safe because every completed seed/budget JSON is
detected and skipped.

The full sweep contains 280 schedule evaluations
(10 full-guidance runs plus 3 budgets × 9 schedules × 10 seeds). On the
validated A6000 setup it should
be treated as a multi-hour job; runtime varies substantially with GPU,
filesystem, CUDA, and LPIPS cache state. Generated `.pt`, preview, log, and
JSON files require roughly 2–3 GB in addition to the environment and
checkpoint.

## Fast end-to-end check

From the repository root:

```bash
bash scripts/reproduce_cifar10_mdlm.sh smoke cuda:0
```

Loads both supplied checkpoints and executes a small 16-step VISTA run on
`cuda:0`. It verifies model loading, warm-up, schedule solving, sampling,
reward evaluation, LPIPS evaluation, and result writing without launching the
full multi-hour sweep. Outputs are written under `eval_runs/smoke/`.

## Reproduce the submitted tables

```bash
bash scripts/reproduce_cifar10_mdlm.sh tables cuda:0
```

Runs the complete table experiment on one GPU and automatically invokes the
aggregation command after every expected result file is present.

For the four-GPU layout used during development:

```bash
bash scripts/reproduce_all_tables_4gpu.sh
```

Runs the same experiment with seed-level parallelism over four GPUs. It does
not change any per-seed experimental hyperparameter.

The table mode evaluates:

- a full-guidance reference at `T' = T = 1000`;
- guidance budgets `T' = 5, 15, 25`;
- schedules `interval1`–`interval5`, `uniform`, `vista`, `top_v`, and `top_dv`;
- random seeds 1–10;
- 1,000 reverse steps, 20 particles, warm-up `M=5`, rollout count `J=5`;
- ESS threshold 0.95 and partial resampling of 10 particles;
- reward threshold 0.5 and LPIPS duplicate threshold 0.03.

Each seed produces five independent SMC generations and retains all 20 final
particles, i.e. 100 evaluated images per schedule, seed, and budget. The VISTA
warm-up curve is independent of the guidance budget, so it is computed once
per seed at `T'=25` and safely reused across the other sparse budgets. Runs
are resumable:
completed comparison JSON files are skipped.

At completion, the command runs:

```bash
python scripts/aggregate_tables.py \
  --input-root eval_runs/tables \
  --output-dir eval_runs/tables/summary
```

Reads already completed per-seed comparison JSON files without running the
diffusion model again. It rejects missing or incompatible metadata, computes
the mean and sample standard deviation across ten seeds, and writes:

- `reward.csv`
- `success_rate.csv`
- `executed_time_seconds.csv`
- `unique_success_lpips003.csv`
- `s_hat_g.csv`
- `policy_table.csv` and `policy_table.md`, with the requested columns
  `T'`, `Policy`, `executed time [s]`, `unique success`, and
  `$\hat{S}(G)$`; the first data row is the `T'=1000` full-guidance
  reference, followed by `T'=5,15,25` in the policy order `Interval 1`,
  `Interval 2`, `Interval 3`, `Interval 4`, `Interval 5`, `Uniform`,
  `VISTA-DP`, `Top V`, `Top dV`
- `tables.json`, including every per-seed value
- `tables.md`, containing the rendered metric tables

Every displayed value is the arithmetic mean and sample standard deviation
across the ten seeds. `executed time` is the generation time after schedule
selection, matching the paper's timing convention; the VISTA warm-up is not
included. Following Eq. (12) of the paper, the reported objective is the
unnormalized sum
`$\hat{S}(G)=\sum_{t\in\mathcal{T}\setminus G}\hat{V}_{\lceil t\rceil}$`.
The two Reward screenshots supplied for this package are identical; both are
reproduced by the single generated `reward.csv` table.

Reference aggregate outputs are included in `results/`. Stochastic GPU kernels
and library/platform differences can cause small numerical variation.

## Training from scratch

The supplied checkpoint is sufficient for evaluation. To retrain it:

```bash
bash scripts/train_cifar10_mdlm.sh data/cifar10 outputs/cifar10/mdlm
```

Downloads CIFAR-10 when it is absent and trains the class-conditional MDLM from
scratch into `outputs/cifar10/mdlm`. The command uses the UNet backbone,
absorbing-state diffusion, classifier-free dropout 0.1, global batch size 512,
and 300,000 optimizer steps. This is optional because the anonymized pretrained
checkpoint is already supplied.

To retrain the lightweight reward classifier after CIFAR-10 is downloaded:

```bash
python reward_classifier.py \
  --data-dir data/cifar10 \
  --output-dir outputs/cifar10/reward_classifier \
  --device cuda:0
```

Trains the lightweight clean-image CIFAR-10 classifier used to compute the SMC
reward and success metrics. It writes `best.ckpt` and `last.ckpt` under
`outputs/cifar10/reward_classifier/`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Runs all CPU unit tests, including table aggregation and metadata validation,
VISTA dynamic programming, partial resampling, value-gradient conditioning,
and proposal invariants. The test suite does not require a GPU checkpoint run.
