# Toxic-text SMC schedule trimming

This directory is the reviewer-facing reproduction package for the toxic-text
generation experiment in *Value-Informed Schedule Trimming for Accelerated
Sequential Monte Carlo Guidance in Discrete Diffusion*.

It contains the code, prompt data, all-step value traces, raw generations, and
evaluations for:

- the 100-step all-step baseline;
- sparse budgets `T' = 5, 10, 15`;
- VISTA, Top-V, Top-dV, Uniform, and five evenly positioned interval policies.

No fine-tuned model is used. The experiment uses the public pretrained MDLM
reference model. Model weights are downloaded from Hugging Face at the pinned
revisions in `model_revisions.py`; they are not duplicated in this package.

> The released generations contain intentionally toxic or offensive text.

## Package contents

| Path | Contents |
|---|---|
| `eval.py` | Sparse-policy generation and evaluation driver |
| `sample_allstep_traces.py` | All-step generation and reusable value traces |
| `warmup_trace_utils.py` | VISTA, Top-V, Top-dV, Uniform, and interval schedules |
| `smc/` | SMC inference, proposal, weighting, and resampling |
| `mdlm/` | Required MDLM source; upstream license is retained |
| `evaluation/` | PPL, CoLA, diversity, toxicity, and unique-edit metrics |
| `configs/eval.yaml` | Released experiment configuration |
| `outputs/paper_toxic/20260724_120627/` | Released allstep/Tprime5/10/15 artifacts |
| `RESULTS.md` | Compact released result table and metric definitions |
| `scripts/verify_package.py` | Cross-check schedules, scores, traces, and results |
| `scripts/summarize_results.py` | Print the compact result table |

Generated trace plots and console logs are omitted because they are
deterministically reproducible from the included JSONL traces.

## Released configuration

- prompts/runs: `15 x 2 = 30` independent `(prompt, run)` units;
- particles: `8`;
- diffusion steps: `T = 100`;
- guidance budgets: `T' in {5, 10, 15}`;
- proposal: SMC-grad;
- rollout samples per value estimate: `phi = 4`;
- KL weight: `0.2`;
- resampling frequency: every guided step, ESS threshold `0.98`;
- base seed: `42`, deterministically mixed with prompt and run indices;
- schedule warmup traces: exactly `p0_r0`, `p1_r0`, and `p2_r0` from allstep;
- evaluation grouping: one observation is the eight particles from one
  `(prompt, run)`; reported standard deviations use `ddof=1` over 30 groups.

Here, `15 x 2 = 30` and `M = 3` refer to different things. The all-step
baseline generates 30 `(prompt, run)` units so that its metrics are directly
comparable with the sparse policies. Schedule estimation then averages only
the three exact all-step traces `p0_r0_allstep`, `p1_r0_allstep`, and
`p2_r0_allstep`. The obsolete sensitivity sweep over warmup sizes
`[1, ..., 15]` is not part of this package or experiment.

The five interval starts are evenly spaced from the first execution window to
the last. Because timestep indices run in reverse execution order,
`interval1 = range(T-T', T)` and `interval5 = range(0, T')`.

## Setup

The released environment used Python 3.12, PyTorch 2.6.0 with CUDA 12.4, and a
48 GB NVIDIA GPU. A recent Linux system with an NVIDIA GPU is expected.

```bash
# cd language_modelling_submit
bash scripts/setup_venv.sh
```

The setup uses the SDPA compatibility layer in `flash_attn_compat.py`, so a
separate `flash-attn` build is not required. Network access is required once to
install packages and download these pinned public models:

| Purpose | Hugging Face model |
|---|---|
| Generation | `kuleshov-group/mdlm-owt` |
| Generation tokenizer | `gpt2` |
| MDLM initialization tokenizer | `gpt2-large` |
| SMC toxicity reward | `s-nlp/roberta_toxicity_classifier` |
| Output conversion | `roberta-large` |
| Perplexity | `gpt2-xl` |
| CoLA | `textattack/roberta-base-CoLA` |
| Toxicity evaluation | `SkolkovoInstitute/roberta_toxicity_classifier` |
| External toxicity evaluation | `textdetox/xlmr-large-toxicity-classifier` |

## Verify the release

The fast verifier does not run generation or load model weights. It recomputes
all schedules and DP scores from the three selected all-step value traces and
checks every released generation/evaluation unit:

```bash
.venv/bin/python scripts/verify_package.py
.venv/bin/python scripts/summarize_results.py
.venv/bin/python -m pytest
```

To additionally hash every released file:

```bash
.venv/bin/python scripts/verify_package.py --checksums
```

## Reproduce from scratch

The execution order is:

1. Generate and evaluate allstep for 15 prompts x 2 runs (30 units).
2. Estimate VISTA, Top-V, and Top-dV from exactly three of those saved traces.
3. Run all nine sparse policies for `T'=5,10,15`.

The wrapper performs all three stages in that order:

```bash
CUDA_DEVICE=0 \
EVAL_CUDA_VISIBLE_DEVICES=0 \
RUN_ROOT="$PWD/outputs/reproduction_full" \
bash scripts/eval_smc_grad.sh
```

`CUDA_DEVICE` is the PyTorch GPU index for generation.
`EVAL_CUDA_VISIBLE_DEVICES` selects the GPU exposed to the evaluation
subprocess. On a one-GPU machine both should be `0`.

The exact three schedule traces are named in both `configs/eval.yaml` and
`scripts/run_eval_tprime_sweep.sh`. If `eval_schedule_warmup_tags` is set, it
takes precedence over the fallback count `eval_schedule_warmup_samples=3`.

## Reuse the released all-step traces

Sparse schedules can be reproduced without repeating the expensive all-step
generation:

```bash
CUDA_DEVICE=0 \
EVAL_CUDA_VISIBLE_DEVICES=0 \
SMC_ALLSTEP_MANIFEST="$PWD/outputs/paper_toxic/20260724_120627/allstep/allstep_samples_manifest.json" \
OUTPUT_ROOT="$PWD/outputs/reproduction_sparse" \
bash scripts/run_eval_tprime_sweep.sh 5 10 15
```

The value-informed schedules always use the exact manifest tags
`p0_r0_allstep`, `p1_r0_allstep`, and `p2_r0_allstep`. Existing complete T'
directories are skipped; an incomplete non-empty directory causes a failure
unless `ALLOW_OVERWRITE=1` is explicitly set.

## Released artifacts

Each strategy directory contains:

- 30 `text_samples_*.jsonl` files, each with all eight particles;
- 30 corresponding `abc_ssdlm_gen_*.jsonl` files;
- 30 `reward_trace_*.jsonl` files;
- one merged generation file with 15 rows and 16 generations per row;
- inference timing records;
- PPL details and a human-readable metric result file.

`unique_edit#0.05 toxic` in the metric files is normalized by eight particles.
`unique_edit_count#0.05 toxic` reports the corresponding raw number of unique
predicted-toxic generations per `(prompt, run)`.

## Attribution

The MDLM source under `mdlm/` is based on
Sahoo et al., *Simple and Effective Masked Diffusion Language Models*, and
retains its Apache-2.0 license and citation metadata. Evaluation code was
adapted from FK-Diffusion-Steering and SSD-LM as noted in source headers.
