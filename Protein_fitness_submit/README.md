# Protein Fitness: MDLM/UDLM SMC-base reproduction

This reviewer package reproduces the Protein Fitness SMC-base results for two
discrete-diffusion backbones:

- **Table E-1:** MDLM (absorbing-state diffusion) with SMC-base;
- **Table E-2:** UDLM (uniform diffusion) with SMC-base.

The generated tables contain the full-guidance row (`T'=T=128`) and the two
requested sparse budgets, `T'=16` and `T'=32`. `T'=48`, SMC-grad, training
pipelines, optimizer states, old experiment logs, and unrelated biology tasks
are intentionally excluded.

## Package contents

- `checkpoints/{mdlm,udlm}/TrpB/best_model.ckpt`: inference-only diffusion
  checkpoints (Git LFS);
- `oracle/checkpoints/TrpB/`: the 20-member fitness-oracle ensemble (Git LFS);
- `models/`, `sampling/`, `problem/`, `oracle/`, `util/`: inference-time code;
- `scripts/run_protein_fitness.py`: resumable quality-metric runner;
- `scripts/benchmark_timing.py`: CUDA-synchronized timing runner;
- `scripts/aggregate_tables.py`: required-cell validator and table renderer;
- `scripts/reproduce_tables{,_4gpu}.sh`: one- and four-GPU entry points;
- `results/`: reference tables and schedule-regression data;
- `tests/`: CPU tests for schedules, aggregation, diversity, and anonymity.

The large diffusion checkpoints were reduced to the model `state_dict` needed
for inference. Training callbacks, absolute training paths, optimizer state,
loop state, and duplicated EMA tensors are not distributed. The package's
anonymity audit also checks source artifacts and checkpoint metadata.

## Fixed experiment configuration

| Setting | MDLM | UDLM |
|---|---:|---:|
| sampler | SMC-base | SMC-base |
| reverse steps `T` | 128 | 128 |
| reported `T'` | Full (128), 16, 32 | Full (128), 16, 32 |
| particles `N` | 32 | 32 |
| independent runs | 20 | 20 |
| value rollouts `J` | 3 | 3 |
| VISTA warm-up runs `M` | 3 | 3 |
| ESS threshold | 0.98 | 0.98 |
| resampling | partial | partial |
| reward temperature `alpha` | 0.3 | 0.2 |
| warm-up seed | 1000 | 1000 |
| evaluation seeds | 8000--8019 | 5000--5019 |
| VISTA objective | unweighted | `lambda(t)=(1-t/T)` (`k=1`) |

At `T'=16` and `T'=32`, the compared policies are Interval-1 through
Interval-5, Top-V, Top-dV, Uniform, and VISTA. The `T'=128` row is evaluated
once as Full because all policy definitions reduce to the same 128 guided
steps.

`Unique Valid` is computed independently inside each 32-particle run:

1. retain sequences whose oracle fitness is at least `0.5`;
2. connect retained 15-residue variants with Hamming distance at most one;
3. count connected components; and
4. report the mean and sample standard deviation over 20 runs.

## Setup

Git LFS is required for the supplied weights:

```bash
git lfs install
git lfs pull
cd Protein_fitness_submit
```

Create the reference Python 3.9 / CUDA 12.9 environment:

```bash
conda env create -f environment.yml
conda activate protein-fitness-submit
```

FlashAttention is optional because the model has a PyTorch fallback, but it is
recommended for the reported runtime configuration:

```bash
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

Verify every distributed weight, inspect the sanitized checkpoint metadata,
and run the CPU test suite:

```bash
python scripts/verify_package.py
python -m pytest -q
```

A short four-step GPU check loads both backbones, the oracle ensemble, and
SMC-base without starting the table-scale sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py
```

## Inspect the reference tables without running a model

```bash
python scripts/aggregate_tables.py \
  --reference results/reference_tables.json \
  --output-dir outputs/reference_tables
```

This writes `table_e1.md/csv` and `table_e2.md/csv`. The committed copies are
in `results/`. The reference JSON preserves the supplied table values; newly
generated tables are always computed from the regenerated per-run artifacts.
The supplied E-1 MDLM `T'=16` Uniform row reports an SD of `3.24`, while a
direct sample-SD recomputation from its 20 archived counts gives approximately
`3.79`. Regenerated tables use the newly generated counts rather than
substituting either reference value.

## Reproduce on one GPU

```bash
bash scripts/reproduce_tables.sh 0
```

The argument is the physical CUDA device exposed to each process. The command
runs both backbones sequentially, benchmarks all required rows, validates every
artifact, and writes the final tables under `outputs/tables/`.

Every metric JSON is saved after each seed, and every timing JSON is saved
after each measurement. Rerunning the command continues missing runs after
checking the fixed experiment configuration and checkpoint digest.

## Reproduce on four GPUs

```bash
bash scripts/reproduce_tables_4gpu.sh
```

The default physical devices are `0 1 2 3`. They can be remapped without
editing the script:

```bash
GPU0=2 GPU1=3 GPU2=4 GPU3=5 bash scripts/reproduce_tables_4gpu.sh
```

MDLM and UDLM warm-ups are generated once each. The two sparse budgets are
then split across four independent processes; the Full rows and controlled
timings follow. Logs are written to `outputs/logs/`.

## Output layout

```text
outputs/
├── vhat/{mdlm,udlm}.pt
├── metrics/{mdlm,udlm}_b{16,32,128}.json
├── timing/{mdlm,udlm}_b{16,32,128}.json
├── tables/table_e{1,2}.{md,csv}
└── logs/
```

Metric artifacts retain the generated 15-residue variants, oracle scores, and
per-run `Unique Valid` counts. The aggregator requires exactly 20 contiguous
evaluation seeds for every displayed cell. Timing artifacts are separate so
the table uses only CUDA-synchronized `SMC_Base.inference(detokenize=True)`
measurements; VISTA warm-up, schedule construction, oracle post-processing,
and aggregation are outside the timed region.

## Reproducibility notes

The reference timing environment used an NVIDIA RTX PRO 6000 Blackwell GPU,
Python 3.9.23, PyTorch 2.7.1 with CUDA 12.9, and FlashAttention 2.8.3.post1.
Wall-clock values are not portable across GPUs, CUDA kernels, process load, or
library versions. Sampling results may also vary slightly across compatible
GPU/software stacks despite fixed seeds. The intended reproduction criterion
is the same qualitative table under the fixed configuration, not bitwise
identity of every aggregate.

On the reference-class hardware, the complete one-GPU command should be
treated as a multi-hour run. The four-GPU launcher reduces wall time by running
independent budget/backbone shards; it does not use DDP or alter any per-run
experiment setting.
