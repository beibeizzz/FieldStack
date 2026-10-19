"""Behavioral checks for topology, padding, gradients, and data alignment."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blocks import RoutingBackbone
from dataset import ParquetBatches, load_schema, move_to_device
from make_demo import demo_rows, demo_schema, write_demo
from model import TailScope, ModelConfig, compress_history


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="tailscope_test_")
        cls.root = write_demo(cls.temp.name, 24)
        cls.schema = load_schema(cls.root / "schema.json")
        cls.config = ModelConfig.load(cls.root / "config.json")
        cls.data = ParquetBatches(cls.root / "train", cls.schema, 4, cls.config.seq_max_lens)
        cls.batch = cls.data.convert(demo_rows(4, cls.schema))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        torch.manual_seed(17)

    def test_topology_edges(self):
        backbone = RoutingBackbone(16, 4, 1, num_memory_tokens=1,
            num_domain_memory_tokens=1, local_window=1, recent_window=2,
            seq_domains=["seq_a", "seq_b"])
        # Global, memA, memB, qA, qB, user, item, six A events, six B events.
        ids = torch.tensor([7, 9, 10, 13, 14, 0, 1] + [3] * 6 + [4] * 6)
        allow = backbone._build_domain_memory_mask(ids) == 0
        self.assertTrue(allow[0].all())
        self.assertEqual(set(allow[1].nonzero().flatten().tolist()), {0, 1, 2, 3, 7, 8, 9, 10, 11, 12})
        self.assertEqual(set(allow[9].nonzero().flatten().tolist()), {0, 1, 3, 8, 9, 10})
        self.assertTrue(allow[3, 7:13].all())
        self.assertTrue(allow[3, 13:15].all())
        self.assertFalse(allow[3, 15:].any())
        self.assertTrue(allow[5, :7].all())
        self.assertEqual(set(allow[5, 7:].nonzero().flatten().tolist()), {0, 1, 6, 7})

    def test_query_to_sequence_is_explicit(self):
        backbone = RoutingBackbone(16, 4, 1, dropout=0,
            num_memory_tokens=1, num_domain_memory_tokens=1,
            local_window=0, recent_window=1, seq_domains=["seq_a", "seq_b"])
        tokens = torch.randn(1, 6, 16, requires_grad=True)
        types = torch.tensor([[8, 8, 0, 1, 3, 4]])
        topology = torch.tensor([[13, 14, 0, 1, 3, 4]])
        out, _ = backbone(tokens, types, topology_ids=topology)
        (out[0, 3 + 4] * torch.arange(16.)).sum().backward()
        self.assertGreater(tokens.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(tokens.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(tokens.grad[0, 2].abs().sum().item(), 0)
        self.assertEqual(tokens.grad[0, 5].abs().sum().item(), 0)

    def test_tail_summary_masks_and_values(self):
        tokens = torch.arange(8.).reshape(1, 8, 1)
        mask = torch.tensor([[False, False, False, False, True, True, True, True]])
        out, out_mask = compress_history(tokens, mask, 2, 3)
        torch.testing.assert_close(out.flatten(), torch.tensor([0., 1., 2.5, 0., 0.]))
        self.assertEqual(out_mask.tolist(), [[False, False, False, True, True]])

    def test_padding_invariance(self):
        model = TailScope(self.schema, self.config).eval()
        changed = copy.deepcopy(self.batch)
        for domain in model.domains:
            mask = changed["seq_mask"][domain]
            changed["seq"][domain][mask] = 7
            changed["time_buckets"][domain][mask] = 10
        with torch.no_grad():
            torch.testing.assert_close(model(self.batch), model(changed), atol=1e-6, rtol=1e-6)

    def test_empty_history_and_gradient(self):
        model = TailScope(self.schema, self.config)
        logits = model(self.batch)
        self.assertTrue(torch.isfinite(logits).all())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, self.batch["label"])
        loss.backward()
        for name in ("cross_user_view", "cross_item_view"):
            self.assertGreater(getattr(model.query, name).attn.W_v.weight.grad.abs().sum().item(), 0)
        for domain in model.domains:
            self.assertGreater(model.backbone.domain_memory_tokens[domain].grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_ablation_paths(self):
        for changes in ({"query_mode": "mean"}, {"topology": "full"},
                        {"domain_memory": 0}, {"tail_summaries": 0}, {"tail_pooling": "mean"}):
            model = TailScope(self.schema, replace(self.config, **changes))
            out = model(self.batch)
            self.assertEqual(out.shape, (4,))
            self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()

    def test_without_item_dense_schema(self):
        schema = demo_schema(without_item_dense=True)
        loader = copy.copy(self.data)
        loader.schema = schema
        batch = loader.convert(demo_rows(4, schema))
        self.assertEqual(batch["item_dense"].shape, (4, 0))
        self.assertTrue(torch.isfinite(TailScope(schema, self.config)(batch)).all())

    def test_data_alignment_and_id_bounds(self):
        rows = demo_rows(1, self.schema)
        rows[0]["user_int_feats_1"] = 99999
        rows[0]["user_dense_feats_61"] = [float("nan"), 1, 2, 3]
        for spec in self.schema["seq"].values():
            prefix = spec["prefix"]
            rows[0][f"{prefix}_11"] = [1, 2, 3]
            rows[0][f"{prefix}_12"] = [4, 5, 6]
            rows[0][f"{prefix}_13"] = [100, 300, 200]
        batch = self.data.convert(rows)
        self.assertEqual(batch["user_int"][0, 0].item(), 0)
        self.assertEqual(batch["user_dense"][0, 0].item(), 0)
        self.assertEqual(batch["seq"]["seq_a"][0, :3].tolist(), [[2, 5], [3, 6], [1, 4]])
        self.assertEqual((~batch["seq_mask"]["seq_a"]).sum().item(), 3)

    def test_temporal_partition(self):
        cutoff = 1750000000 + 8 * 60
        left = ParquetBatches(self.root / "train", self.schema, 4, self.config.seq_max_lens, timestamp_max=cutoff)
        right = ParquetBatches(self.root / "train", self.schema, 4, self.config.seq_max_lens, timestamp_min=cutoff)
        a = [t for b in left for t in b["timestamp"].tolist()]
        b = [t for batch in right for t in batch["timestamp"].tolist()]
        self.assertEqual(len(a) + len(b), 16)
        self.assertLess(max(a), min(b))

    def test_checkpoint_roundtrip(self):
        model = TailScope(self.schema, self.config).eval()
        path = self.root / "unit_checkpoint.pt"
        torch.save(model.state_dict(), path)
        restored = TailScope(self.schema, self.config).eval()
        restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(model(self.batch), restored(self.batch), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward(self):
        model = TailScope(self.schema, self.config).cuda()
        batch = move_to_device(self.batch, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), batch["label"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
