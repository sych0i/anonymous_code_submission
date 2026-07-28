#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/cifar10}"
OUTPUT_DIR="${2:-outputs/cifar10/mdlm}"

python -u -m main \
  is_vision=True \
  diffusion=absorbing_state \
  parameterization=subs \
  T=0 \
  time_conditioning=False \
  zero_recon_loss=False \
  data=cifar10 \
  data.train="$DATA_DIR" \
  data.valid="$DATA_DIR" \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=64 \
  backbone=unet \
  model=unet \
  optim.lr=2e-4 \
  lr_scheduler=constant_warmup \
  lr_scheduler.num_warmup_steps=5000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
  trainer.max_steps=300000 \
  trainer.val_check_interval=10000 \
  +trainer.check_val_every_n_epoch=null \
  training.guidance.cond_dropout=0.1 \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=True \
  sampling.steps=128 \
  wandb.name=cifar10_mdlm \
  hydra.run.dir="$OUTPUT_DIR"
