import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from smc.pipeline import Pipeline
from warmup_trace_utils import (
    load_warmup_samples,
    mean_exp_alpha_r_series_from_trace_jsonl,
)


class _Tokenizer:
    def batch_decode(self, latents):
        return [" ".join(map(str, row.tolist())) for row in latents]


class _Model:
    vocab_size = 3
    mask_index = 2
    device = torch.device("cpu")
    tokenizer = _Tokenizer()

    def _sample_prior(self, batch_size, length, prompt_ids=None):
        return torch.full((batch_size, length), self.mask_index, dtype=torch.long)

    def get_logits(self, state, time):
        if state.ndim == 2:
            state = F.one_hot(state, num_classes=self.vocab_size).float()
        signal = state[..., 0] - state[..., 1]
        return torch.stack(
            (signal, -signal, torch.full_like(signal, -1e9)),
            dim=-1,
        )


class _Scheduler:
    mask_token_id = 2

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps

    def step(self, latents, step, logits):
        return SimpleNamespace(new_latents=self._next(latents))

    def step_with_approx_guidance(
        self, latents, step, logits, approx_guidance
    ):
        batch_size = latents.shape[0]
        zeros = torch.zeros(batch_size)
        return SimpleNamespace(
            new_latents=self._next(latents),
            log_prob_proposal=zeros,
            log_prob_diffusion=zeros,
        )

    @staticmethod
    def _next(latents):
        token = torch.arange(latents.shape[0]) % 2
        return token[:, None].expand_as(latents).clone()


class PipelineSMCTest(unittest.TestCase):
    def _pipeline(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe.config = SimpleNamespace(
            model=SimpleNamespace(length=2),
            ft_model=SimpleNamespace(ckpt_path=None),
        )
        pipe.model = _Model()
        pipe.scheduler = _Scheduler()
        pipe._execution_device = torch.device("cpu")
        pipe.model_dtype = torch.float
        return pipe

    def test_guided_final_state_is_weighted_and_traced(self):
        pipe = self._pipeline()
        weight_calls = []
        reward_calls = 0

        def reward_fn(samples):
            nonlocal reward_calls
            reward_calls += 1
            return samples[..., 1].float().mean(dim=1) - 1.0

        def resample_fn(log_w):
            weight_calls.append(log_w.detach().clone())
            return torch.arange(log_w.shape[0]), False, log_w

        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                latents, _ = pipe(
                    reward_fn=reward_fn,
                    resample_fn=resample_fn,
                    num_inference_steps=2,
                    num_particles=2,
                    batch_p=2,
                    phi=2,
                    proposal_type="grad",
                    guidance_steps={0, 1},
                    kl_weight=1.0,
                    disable_progress_bar=True,
                    verbose=False,
                )
                with open("reward_trace.jsonl", encoding="utf-8") as f:
                    trace = [json.loads(line) for line in f]
            finally:
                os.chdir(old_cwd)

        self.assertEqual(latents.tolist(), [[0, 0], [1, 1]])
        self.assertEqual([row["timestep"] for row in trace], [2, 1, 0])
        self.assertTrue(all("value_estimates" in row for row in trace))
        self.assertEqual(len(weight_calls), 3)
        self.assertFalse(torch.allclose(weight_calls[-1][0], weight_calls[-1][1]))
        self.assertEqual(reward_calls, 5)

    def test_trace_reader_uses_value_estimates_directly(self):
        rows = [
            {
                "value_estimates": [2.0, 4.0],
                "reward_aggregated": [-100.0],
                "scale_cur": 5.0,
            },
            {
                "value_estimates": [6.0, 10.0],
                "reward_aggregated": [-100.0],
                "scale_cur": 5.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            self.assertEqual(
                mean_exp_alpha_r_series_from_trace_jsonl(path),
                [8.0, 3.0],
            )

    def test_grad_rejects_finetuned_reference_model(self):
        pipe = self._pipeline()
        pipe.config.ft_model.ckpt_path = "hidden/lora"
        with self.assertRaisesRegex(ValueError, "pretrained reference model"):
            pipe(
                reward_fn=lambda samples: torch.zeros(samples.shape[0]),
                resample_fn=lambda log_w: (
                    torch.arange(log_w.shape[0]),
                    False,
                    log_w,
                ),
                num_inference_steps=2,
                num_particles=2,
                batch_p=2,
                proposal_type="grad",
                disable_progress_bar=True,
                verbose=False,
            )

    def test_legacy_trace_is_rejected_for_schedule_estimation(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "reward_trace_old_allstep.jsonl")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"reward_aggregated": [-1.0], "scale_cur": 5.0}
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "legacy trace"):
                load_warmup_samples(
                    trace_manifest=None,
                    trace_dir=tmp,
                    trace_glob="*allstep.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
