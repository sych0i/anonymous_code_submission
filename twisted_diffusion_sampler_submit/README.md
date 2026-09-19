# Twisted Diffusion Sampler — reviewer submission

This folder is the self-contained reproduction bundle for
`outputs/eval30_warmup3_budgets_6_12` from the parent project. It contains only
the MNIST sampler implementation, its transitive diffusion dependencies, the
two checkpoints, the frozen `M=3` warmup value curve used by that output, and
the scripts needed to verify and rerun it. The original outputs, PDFs, dataset,
and virtual environment are intentionally excluded.

## Target experiment

| Setting | Value |
|---|---|
| target class | MNIST digit `4` |
| diffusion transitions | `T=100` (1000-step DDPM, 100-step respacing) |
| guidance budgets | `T'=6, 12` |
| policies | interval-1..5, uniform, VISTA, top-V, top-dV, weighted-VISTA powers 1..3 and 20 |
| particles / rollouts | `N=20` / `J=1` |
| guidance strength | `alpha=2` |
| warmup | dense full schedule, frozen `M=3`, seeds `100000..100002` |
| sampling | 30 seeds `0..29`, 20 samples per seed |
| ESS / partial resampling | `ESS < 0.98N` / `K=20`, half-high-half-low systematic |
| validity / duplicate thresholds | log-probability `>-0.1` / binary Jaccard `>=0.85` |

The default command writes to `outputs/eval30_warmup3_budgets_6_12/` and checks
the result against `expected_results.json`. The reference Unique Valid values
are arithmetic means of the 30 seedwise counts; they are not pooled clustering
over all 600 samples. UTD is the one-shot pred-xstart delta proxy, not a proven
theoretical upper bound.

## Setup and verification

Use Python 3.10 or newer. The pinned reference environment was Python 3.10,
PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118, and CUDA 11.8.

```bash
cd /path/to/twisted_diffusion_sampler_submit
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=/tmp/tds-mpl .venv/bin/python scripts/verify_setup.py
```

The verification is CPU-only. It checks checkpoint/config/warmup hashes, the
hash-pinned sampler implementation, and that the warmup summary is compatible
with the current checkout path. No MNIST dataset download is required for this
experiment because inference uses the bundled checkpoints only.

## Reproduce the target output

With one GPU:

```bash
GPU_ID=0 PYTHON=.venv/bin/python ./reproduce.sh
```

With three GPUs, the three policy groups run in parallel:

```bash
GPU_IDS=0,1,2 PYTHON=.venv/bin/python ./reproduce.sh
```

The script is resumable and uses the portable frozen warmup by default. Set
`OUTPUT_DIR=/some/new/path` for a clean run. Set `CHECK_REFERENCE=0` only when
running a deliberately different budget/seed configuration. `WARMUP_MODE=fullstep`
recomputes the dense warmup on GPU instead of reusing the frozen curve, but is
not needed for the exact target output.

`run_deterministic.py` enables deterministic cuDNN/cuBLAS behavior and routes
the short systematic-resampling CDF cumulative sum through CPU, removing the
remaining CUDA reduction nondeterminism that affects seedwise Unique Valid.
The hash-pinned files under `image_exp/` are left unchanged so the provenance
fingerprint in the manifests and reference file remains valid.

## Bundle layout

```text
image_exp/
  run_vista.py, vista_*.py       sampler, schedules, and metrics
  image_diffusion/               transitive DDPM/UNet/SMC implementation
  image_confs/                   model and classifier configuration
  models/                        bundled model060000.pt and resnet.pth.tar
artifacts/shared_warmup/
  value_means.json               frozen M=3 value curve data
scripts/
  prepare_portable_warmup.py     path-correct warmup summary generator
  run_deterministic.py           deterministic runner wrapper
  summarize_results.py            shard validation and aggregation
  verify_setup.py                CPU integrity check
expected_results.json             reference for the target output
reproduce.sh                      single reproduction entry point
requirements.txt                  pinned Python dependencies
```
