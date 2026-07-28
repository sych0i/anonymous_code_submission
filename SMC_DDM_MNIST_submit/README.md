# SMC_DDM_MNIST

Minimal reviewer package for reproducing the binarized-MNIST SMC sweep defined
in `configs/smc_sweep.yaml`. The YAML file is the authoritative experiment
configuration; the notebook's own default values are replaced by the runner.

## Contents

- `configs/smc_sweep.yaml`: experiment matrix, seeds, device, and runtime options
- `scripts/run_configured_smc_sweep.py`: reviewer entry point
- `scripts/run_smc_binarized_mnist_sweep.py`: YAML/CLI sweep runner
- `scripts/smc_binarized_mnist_vista.ipynb`: experiment implementation
- `datasets`, `models`, `smc`, `utils`: runtime modules used by the notebook
- `model_weights`: pretrained MDM/UDM and digit-classifier checkpoints

The runner reads and executes the notebook's code cells directly, so Jupyter
and nbconvert are not required. The package intentionally excludes training
notebooks, Gaussian-mixture experiments, cached MNIST files, previous results,
figures, logs, editor settings, virtual environments, and Git history.

## Requirements

Python 3.10 and an NVIDIA GPU are recommended. The supplied versions use
PyTorch 2.7.0 with CUDA 12.8. From the repository root, create a clean
environment and install the dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The final command recognizes the already installed PyTorch packages and
installs the remaining pinned dependencies. A compatible NVIDIA driver is
required for the CUDA build.

MNIST is downloaded automatically into `datasets/data/` on the first actual
run, so that run requires network access. The included checkpoints are loaded
from `model_weights/`; ReMDM and MDM intentionally use the same pretrained MDM
checkpoint with different diffusion transitions.

The code falls back to CPU when CUDA is unavailable, but reproducing the full
sweep on CPU is likely to be impractically slow.

## Reproduce the configured sweep

First inspect the exact plan without loading models, downloading MNIST, or
running an experiment:

```bash
python scripts/run_configured_smc_sweep.py --dry-run
```

With the supplied YAML, this must print 24 runs:

- model types: `mdm`, `remdm`, `udm`
- proposal modes: `base`, `grad`
- guidance budgets \(T'\): `6`, `12`, `18`, `24`
- matrix size: 3 models x 2 proposals x 4 budgets = 24 runs

Then launch the complete sweep:

```bash
python scripts/run_configured_smc_sweep.py
```

This command automatically loads `configs/smc_sweep.yaml`. It uses warmup
seeds 100--102 (`100:103`, stop-exclusive) and evaluation seeds 0--49
(`0:50`, stop-exclusive). Each reduced-budget run evaluates Uniform, Top dV,
Top V, VISTA, and Interval 1--5. The all-step warmup and all-step benchmark are
computed once per model/proposal pair and reused across its four budgets.

The supplied configuration requests CUDA device index 3. Set
`cuda_device_index` in `configs/smc_sweep.yaml` to the desired zero-based GPU
index, or set it to `null` to use PyTorch's default CUDA device. If the
requested index exceeds the number of visible GPUs, the notebook selects the
last visible GPU. `CUDA_VISIBLE_DEVICES` may also be used to control device
visibility.

The full run is computationally expensive: it covers 24 configurations, nine
reduced policies per configuration, 50 evaluation seeds, plus the shared
warmup and all-step evaluations. Keep the process attached to a persistent
terminal or job scheduler.

## Outputs and completion check

Results are written under `scripts/`:

```text
scripts/experiment_results_<model>_<proposal>_allstep_warmup_shared.txt
scripts/experiment_results_<model>_<proposal>_allstep.txt
scripts/experiment_results_<model>_<proposal>_tstep_T<budget>.txt
```

The combined stdout/stderr log is written to:

```text
scripts/sweep_logs/smc_sweep_<timestamp>.log
```

Existing compatible result files are treated as caches and skipped. This
allows an interrupted sweep to be resumed by running the same command again.
For an independent clean reproduction, start from a fresh copy of this
reviewer package, which contains no result or log files.

Because `continue_on_error: true`, a failed configuration does not prevent the
remaining configurations from running. The runner still exits nonzero if any
configuration failed. A successful log ends with:

```text
[sweep] all runs completed successfully
```

For a complete run, expect six all-step files, six shared-warmup files, and 24
budget-specific result files. Temporary
`*.incremental.tmp` files indicate an interrupted budget run; rerunning the
command resumes from the records in that file.

## Configuration and overrides

The shipped configuration is:

```yaml
notebook: scripts/smc_binarized_mnist_vista.ipynb
model_types: [mdm, remdm, udm]
proposal_modes: [base, grad]
guidance_steps: [6, 12, 18, 24]
cuda_device_index: 3
warmup_seeds: "100:103"
eval_seeds: "0:50"
log_dir: scripts/sweep_logs
continue_on_error: true
dry_run: false
skip_visualization: true
```

Command-line options override YAML values. For example, the following runs one
small configuration without editing the reproducibility config:

```bash
python scripts/run_configured_smc_sweep.py \
  --model-types mdm \
  --proposal-modes base \
  --guidance-steps 6 \
  --warmup-seeds 100:101 \
  --eval-seeds 0:1
```

Do not use an override when reproducing the reported full sweep.
