import builtins
import os
import json
import math
import time
from typing import Optional

import torch
import torch.nn.functional as F
import hydra

from rich.console import Console

from smc.pipeline import Pipeline
from smc.scheduler import ReMDMScheduler, ReMDMSchedulerWithPrompt
from smc.resampling import resample
from toxicity_classifier.scorer import ToxicityScorer
from tokenizer.utils import create_token_ids_translation_map


device: Optional[torch.device] = None

_console = Console()

toxicity_scorer: Optional[ToxicityScorer] = None
_toxicity_scorer_device: Optional[torch.device] = None
translation_custom_map = {'[PAD]': '<pad>'}
# Its intialized when both tokenizers have been loaded
translation_map = None
translation_matrix = None

def initialize_tokenizer_translation_stuff(tokenizer1, tokenizer2):
    global translation_map, translation_matrix
    assert device is not None
    if translation_matrix is not None: # check if already initialized
        return
    translation_map = create_token_ids_translation_map(tokenizer1, tokenizer2, synonyms=translation_custom_map)
    C1, C2 = len(tokenizer1), len(tokenizer2)
    translation_matrix = torch.zeros((C1, C2), device=device).float()
    for old_class, new_class in translation_map.items():
        translation_matrix[old_class, new_class] = 1.0

def toxicity_reward_fn(gpt_token_ids):
    assert toxicity_scorer is not None and device is not None
    if isinstance(gpt_token_ids[0], str):
        scores = [
            toxicity_scorer.score_text(text) for text in gpt_token_ids
        ]
        # Avoid torch.tensor(list_of_tensors): shapes like (1,) → nested .tolist() and bad plots.
        parts = []
        for s in scores:
            if isinstance(s, torch.Tensor):
                parts.append(s.reshape(-1)[0].detach().float().to(device))
            else:
                parts.append(torch.tensor(float(s), device=device, dtype=torch.float32))
        return torch.stack(parts)
    roberta_token_ids = torch.matmul(gpt_token_ids, translation_matrix) # type: ignore
    return toxicity_scorer.score_token_ids(roberta_token_ids)


def main(config, guidance_steps=None):
    global device, toxicity_scorer, _toxicity_scorer_device

    cuda_idx = int(getattr(config, "cuda_device", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = torch.device("cpu")

    proposal_type = str(config.smc.proposal_type)
    if config.ft_model.ckpt_path and proposal_type in {
        "grad",
        "locally_optimal",
        "locally-optimal",
        "SMC grad",
        "smc grad",
        "reduced_SMC_grad",
        "smc_grad",
        "SMC_grad",
        "reduced_grad",
        "reverse",
        "reduced_SMC_reverse",
        "SMC reverse",
        "smc reverse",
        "smc_reverse",
        "SMC_reverse",
        "reduced_reverse",
    }:
        raise ValueError(
            "The paper's SMC-grad/SMC-base experiments use the pretrained "
            "reference model only. Remove ft_model.ckpt_path; fine-tuned "
            "proposals require a separately implemented p/q correction."
        )

    if toxicity_scorer is None or _toxicity_scorer_device != device:
        toxicity_scorer = ToxicityScorer(device=device)
        _toxicity_scorer_device = device

    # Check model batch size is equal to smc batch size
    assert config.smc.batch_p == config.loader.eval_batch_size, "Model batch size must be equal to smc batch size"

    scheduler = ReMDMSchedulerWithPrompt(
        schedule=config.smc.remdm.schedule,
        remask_strategy=config.smc.remdm.remask_strategy,
        eta=config.smc.remdm.eta,
        mask_token_id=50257,
    )
    pipe = Pipeline(config, scheduler, device=device)

    # Intialize the translation map between GPT tokenizer and the tokenizer used in Roberta toxicity classifier
    initialize_tokenizer_translation_stuff(pipe.model.tokenizer, toxicity_scorer.tokenizer)

    if config.smc.lambda_tempering.enabled:
        lambdas = torch.cat([torch.linspace(0, 1, config.smc.lambda_tempering.one_at + 1), torch.ones(config.smc.num_inference_steps - config.smc.lambda_tempering.one_at)])
        # lambdas = torch.cat([torch.linspace(1, 1, config.smc.lambda_tempering.one_at + 1), torch.ones(config.smc.num_inference_steps - config.smc.lambda_tempering.one_at)])
    else:
        lambdas = None

    smc_started_at = time.perf_counter()
    samples, text_samples = pipe(
        prompt_text=config.smc.prompt_text,
        resample_fn=lambda log_w: resample(log_w, ess_threshold=config.smc.resampling.ess_threshold, partial=config.smc.resampling.partial),
        reward_fn=toxicity_reward_fn,
        num_particles=config.smc.num_particles,
        batch_p=config.smc.batch_p,
        resample_frequency=config.smc.resampling.frequency,
        num_inference_steps=config.smc.num_inference_steps,
        proposal_type=config.smc.proposal_type,
        use_continuous_formulation=config.smc.use_continuous_formulation,
        kl_weight=config.smc.kl_weight,
        lambdas=lambdas,
        phi=config.smc.phi,
        tau=config.smc.tau,
        guidance_steps=guidance_steps,
    )
    smc_elapsed = time.perf_counter() - smc_started_at
    builtins.print(samples.shape)

    # Plot reward trace if available (saved by Pipeline in Hydra run dir)
    try:
        tag = os.environ.get("SMC_TRACE_TAG")
        suffix = f"_{tag}" if tag else ""
        trace_path = os.path.join(os.getcwd(), f"reward_trace{suffix}.jsonl")
        if not os.path.exists(trace_path):
            trace_path = os.path.join(os.getcwd(), "reward_trace.jsonl")
        if os.path.exists(trace_path):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def _flatten_floats(vals):
                if vals is None:
                    return []
                if isinstance(vals, (int, float)):
                    return [float(vals)]
                out = []
                for v in vals:
                    if isinstance(v, (list, tuple)):
                        out.extend(_flatten_floats(v))
                    else:
                        out.append(float(v))
                return out

            steps = []
            reward_agg_avg = []
            resampled_steps = []

            with open(trace_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    steps.append(row["step"])
                    values = _flatten_floats(row.get("value_estimates"))
                    scale_cur = float(row.get("scale_cur", 1.0))
                    rewards = row.get("reward_aggregated", [])
                    flat = _flatten_floats(rewards)
                    if row.get("resampled", False):
                        resampled_steps.append(row["step"])

                    def _mean_exp_scaled(vals, sc):
                        if not vals:
                            return float("nan")
                        return sum(math.exp(sc * v) for v in vals) / len(vals)

                    if values:
                        reward_agg_avg.append(sum(values) / len(values))
                    else:
                        reward_agg_avg.append(_mean_exp_scaled(flat, scale_cur))

            if steps:
                plt.figure(figsize=(10, 4))
                plt.plot(
                    steps,
                    reward_agg_avg,
                    label="mean_p exp(scale_cur * r)",
                )
                if resampled_steps:
                    step_to_y = {s: y for s, y in zip(steps, reward_agg_avg)}
                    xs = [s for s in resampled_steps if s in step_to_y]
                    ys = [step_to_y[s] for s in xs]
                    plt.scatter(xs, ys, marker="x", s=40, linewidths=2, label="resampled")
                plt.xlabel("SMC step i")
                plt.ylabel("mean_p exp(scale_cur * r)")
                plt.title("mean over particles of exp(scale_cur * r)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                out_name = f"reward_trace{suffix}.png" if tag else "reward_trace.png"
                plt.savefig(out_name, dpi=200)
                plt.close()
    except Exception as e:
        _console.print(f"[bold yellow]Warning:[/bold yellow] failed to plot reward trace: {e}")

    toxicity_scores = torch.zeros(config.smc.num_particles)
    for i, text_sample in enumerate(text_samples):
        # Model text can contain "[/...]" etc.; Rich markup would mis-parse it.
        builtins.print("Text sample:", text_sample)
        toxicity_score = toxicity_scorer.score_text(text_sample)
        builtins.print("Toxicity score:", toxicity_score)
        builtins.print("\n")
        toxicity_scores[i] = toxicity_score
    return text_samples, toxicity_scores, smc_elapsed
