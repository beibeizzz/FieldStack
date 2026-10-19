"""Behavioral checks for candidate-conditioned compression."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history import TargetTailPool, compress_history
from model import ModelConfig, TailScope
from make_demo import demo_schema


class HistoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)
        self.pool = TargetTailPool(8)
        self.events = torch.randn(3, 12, 8)
        self.target = torch.randn(3, 8)
        self.mask = torch.zeros(3, 12, dtype=torch.bool)
        self.mask[0] = True
        self.mask[1, 8:] = True

    def activate_scores(self):
        with torch.no_grad():
            self.pool.score[-1].weight.normal_(std=0.5)

    def test_initialization_matches_mean(self):
        actual, mask = self.pool(self.events, self.mask, self.target, 4, 3)
        expected, expected_mask = compress_history(self.events, self.mask, 4, 3)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_target_changes_only_summaries(self):
        self.activate_scores()
        first, mask = self.pool(self.events, self.mask, self.target, 4, 2)
        second, _ = self.pool(self.events, self.mask, -self.target, 4, 2)
        torch.testing.assert_close(first[:, :4], second[:, :4], atol=0, rtol=0)
        self.assertGreater((first[2, 4:] - second[2, 4:]).abs().max().item(), 1e-5)
        self.assertTrue(mask[0].all())
        self.assertEqual(first[0, 4:].abs().sum().item(), 0)

    def test_padding_is_excluded_from_scores_and_gradients(self):
        self.activate_scores()
        events = self.events.clone().requires_grad_()
        changed = events.detach().clone()
        changed[self.mask] = 10000
        first, _ = self.pool(events, self.mask, self.target, 4, 2)
        second, _ = self.pool(changed, self.mask, self.target, 4, 2)
        torch.testing.assert_close(first[:, 4:], second[:, 4:], atol=0, rtol=0)
        first[:, 4:].square().sum().backward()
        self.assertEqual(events.grad[self.mask].abs().sum().item(), 0)

    def test_tail_bins_keep_separate_information(self):
        self.activate_scores()
        changed = self.events.clone()
        changed[2, 8:] += 2
        first, _ = self.pool(self.events, self.mask, self.target, 4, 2)
        second, _ = self.pool(changed, self.mask, self.target, 4, 2)
        torch.testing.assert_close(first[:, :5], second[:, :5], atol=0, rtol=0)
        self.assertGreater((first[2, 5] - second[2, 5]).abs().sum().item(), 0)

    def test_short_histories_and_disabled_summaries(self):
        for length, count in [(2, 4), (5, 4), (12, 0)]:
            out, mask = self.pool(self.events[:, :length], self.mask[:, :length], self.target, 4, count)
            self.assertEqual(out.shape, (3, min(length, 4) + count, 8))
            self.assertTrue(torch.isfinite(out).all())
            if length <= 4:
                self.assertTrue(mask[:, length:].all())

    def test_scorer_gate_and_candidate_receive_gradients(self):
        self.activate_scores()
        target = self.target.clone().requires_grad_()
        out, _ = self.pool(self.events, self.mask, target, 4, 2)
        out[:, 4:].square().sum().backward()
        for layer in [self.pool.score[0], self.pool.score[-1], self.pool.gate]:
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertGreater(target.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.pool.parameters()))

    def test_matched_initialization_of_common_model(self):
        config = ModelConfig(d_model=16, emb_dim=8)
        torch.manual_seed(42)
        mean = TailScope(demo_schema(), replace(config, tail_pooling="mean"))
        torch.manual_seed(42)
        target = TailScope(demo_schema(), config)
        for name, value in mean.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
