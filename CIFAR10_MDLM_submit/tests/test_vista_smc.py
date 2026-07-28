import itertools
import random
import types
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

import vista_smc


class _ProposalModel:
  diffusion = 'absorbing_state'
  mask_index = 3


class GradProposalTest(unittest.TestCase):

  def test_zero_scale_is_exact_noop_and_zeros_stay_zero(self):
    base = torch.tensor([[
      [0.20, 0.30, 0.00, 0.50],
      [0.00, 1.00, 0.00, 0.00],
    ]])
    xt = torch.tensor([[3, 1]])
    grad = torch.tensor([[0.5, -0.5]])
    no_op = vista_smc._grad_twisted_probs(
      _ProposalModel(), base, grad, xt, gamma_grad=0.0)
    self.assertTrue(torch.equal(no_op, base))

    twisted = vista_smc._grad_twisted_probs(
      _ProposalModel(), base, grad, xt, gamma_grad=1.0)
    self.assertTrue(torch.allclose(
      twisted.sum(dim=-1), torch.ones_like(xt, dtype=torch.float32)))
    self.assertTrue(torch.equal(
      twisted[base == 0], torch.zeros_like(twisted[base == 0])))
    self.assertTrue(torch.equal(twisted[0, 1], base[0, 1]))

  def test_gradient_sign_moves_categorical_mean(self):
    base = torch.tensor([[[0.2, 0.3, 0.4, 0.1]]])
    xt = torch.tensor([[3]])
    positive = vista_smc._grad_twisted_probs(
      _ProposalModel(), base, torch.tensor([[0.4]]), xt, 1.0)
    negative = vista_smc._grad_twisted_probs(
      _ProposalModel(), base, torch.tensor([[-0.4]]), xt, 1.0)
    values = torch.arange(4, dtype=torch.float32)
    base_mean = (base * values).sum()
    self.assertGreater((positive * values).sum().item(), base_mean.item())
    self.assertLess((negative * values).sum().item(), base_mean.item())


class PartialResamplingTest(unittest.TestCase):

  @staticmethod
  def _inputs(shift=0.0):
    xt = torch.arange(8).view(8, 1)
    log_weights = torch.tensor([
      [0.0, -1.0, -2.0, -3.0],
      [-0.2, -0.4, -1.5, -2.0],
    ]) + shift
    values = torch.arange(8, dtype=torch.float32).view(2, 4)
    return xt, log_weights, values

  def test_partial_resampling_is_shift_invariant_and_mass_preserving(self):
    outputs = []
    for shift in (0.0, 100.0):
      torch.manual_seed(123)
      outputs.append(vista_smc._maybe_partial_resample(
        *self._inputs(shift), ess_threshold=1.1, partial_resample=2))

    xt_a, logw_a, value_a, flag_a = outputs[0]
    xt_b, logw_b, value_b, flag_b = outputs[1]
    self.assertTrue(torch.equal(xt_a, xt_b))
    self.assertTrue(torch.equal(value_a, value_b))
    self.assertTrue(torch.equal(flag_a, flag_b))
    self.assertTrue(torch.allclose(
      logw_a.softmax(dim=1), logw_b.softmax(dim=1), atol=1e-7))
    self.assertTrue(torch.allclose(
      logw_a.exp().sum(dim=1), torch.ones(2), atol=1e-6))

    # xt stores ancestor IDs, so values must follow the exact same map.
    self.assertTrue(torch.equal(
      xt_a.view(2, 4).to(value_a.dtype), value_a))

  def test_selected_subset_gets_uniform_share_of_its_mass(self):
    xt, logw, values = self._inputs()
    torch.manual_seed(7)
    _, new_logw, _, _ = vista_smc._maybe_partial_resample(
      xt, logw, values, ess_threshold=1.1, partial_resample=2)
    old_weights = logw.softmax(dim=1)
    new_weights = new_logw.exp()
    for row in range(2):
      highest = old_weights[row].argmax()
      lowest = old_weights[row].argmin()
      expected = (old_weights[row, highest] + old_weights[row, lowest]) / 2
      self.assertAlmostEqual(
        new_weights[row, highest].item(), expected.item(), places=6)
      self.assertAlmostEqual(
        new_weights[row, lowest].item(), expected.item(), places=6)

  def test_no_resampling_returns_inputs_unchanged(self):
    xt, logw, values = self._inputs()
    new_xt, new_logw, new_values, flags = (
      vista_smc._maybe_partial_resample(
        xt, logw, values, ess_threshold=0.0, partial_resample=2))
    self.assertTrue(torch.equal(new_xt, xt))
    self.assertTrue(torch.equal(new_logw, logw))
    self.assertTrue(torch.equal(new_values, values))
    self.assertFalse(flags.any())


class _DummyBackbone(nn.Module):

  def __init__(self):
    super().__init__()
    self.conditions = []

  def forward(self, x, sigma, cond, x_emb=None):
    del sigma, x_emb
    self.conditions.append(None if cond is None else cond.detach().clone())
    return torch.stack((-0.01 * x, 0.01 * x, -100.0 + 0.0 * x), dim=-1)


class _DummyGradientModel:

  def __init__(self):
    self.device = torch.device('cpu')
    self.mask_index = 2
    self.backbone = _DummyBackbone()
    self.config = types.SimpleNamespace(
      training=types.SimpleNamespace(guidance='cfg'),
      data=types.SimpleNamespace(num_classes=10),
    )

  @staticmethod
  def _process_sigma(sigma):
    return sigma

  def _subs_parameterization(self, logits, xt):
    logits = logits.clone()
    logits[..., self.mask_index] = float('-inf')
    unmasked = xt != self.mask_index
    if unmasked.any():
      logits[unmasked] = float('-inf')
      logits[unmasked, xt[unmasked]] = 0.0
    return logits.log_softmax(dim=-1)


class _DummyReward(nn.Module):

  @staticmethod
  def log_prob(images):
    mean = images.mean(dim=(1, 2, 3))
    return F.log_softmax(torch.stack((mean, -mean), dim=1), dim=1)


class ValueGradientTest(unittest.TestCase):

  def test_straight_through_gradient_is_finite_and_cfg_conditioned(self):
    torch.manual_seed(5)
    model = _DummyGradientModel()
    xt = torch.full((2, 3 * 32 * 32), model.mask_index, dtype=torch.long)
    sigma = torch.ones(2)
    gradient = vista_smc._estimate_value_grad(
      model=model,
      xt=xt,
      time_conditioning=sigma,
      classifier=_DummyReward(),
      target_class=0,
      alpha=1.0,
      num_rollout_samples=3,
      gumbel_tau=1.0,
      grad_batch_size=1,
    )
    self.assertEqual(gradient.shape, xt.shape)
    self.assertTrue(torch.isfinite(gradient).all())
    self.assertGreater(gradient.mean().item(), 0.0)
    self.assertEqual(len(model.backbone.conditions), 2)
    for condition in model.backbone.conditions:
      self.assertTrue(torch.equal(condition, torch.tensor([10])))


class VistaDynamicProgrammingTest(unittest.TestCase):

  @staticmethod
  def _score(schedule, values, prior):
    selected = set(schedule)
    total = 0.0
    for t in range(len(values)):
      if t in selected:
        continue
      next_guided = min((s for s in selected if s >= t), default=None)
      total += prior if next_guided is None else values[next_guided]
    return total

  def test_dp_matches_exhaustive_search(self):
    rng = random.Random(123)
    for total_steps in range(2, 8):
      for budget in range(1, total_steps):
        for _ in range(10):
          values = [rng.random() for _ in range(total_steps)]
          prior = rng.random()
          actual = vista_smc.solve_vista_schedule(
            values, prior, budget)
          optimum = max(
            self._score(candidate, values, prior)
            for candidate in itertools.combinations(
              range(total_steps), budget))
          self.assertAlmostEqual(
            vista_smc.empirical_schedule_objective(
              values, prior, actual),
            self._score(actual, values, prior),
            places=12)
          self.assertAlmostEqual(
            self._score(actual, values, prior), optimum, places=12)


if __name__ == '__main__':
  unittest.main()
