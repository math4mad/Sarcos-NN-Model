"""Unit tests for the band/truncation logic (stdlib unittest, no new dependencies).

These guard the definitions the whole study rests on: if ``band_indices`` is wrong,
every number in the README means nothing.

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from sarcos_svd import lowrank
from sarcos_svd.data import SplitConfig, block_split
from sarcos_svd.lowrank import band_indices, truncate_matrix, truncate_state
from sarcos_svd.model import ModelConfig


class TestBands(unittest.TestCase):
    def test_band_partitions_are_disjoint_and_complete(self):
        # at full width every band is the identity
        for band in lowrank.BANDS:
            self.assertEqual(list(band_indices(8, 8, band)), list(range(8)))

    def test_leading_is_the_top_directions(self):
        self.assertEqual(list(band_indices(8, 3, "leading")), [0, 1, 2])

    def test_trailing_is_the_minor_directions(self):
        self.assertEqual(list(band_indices(8, 3, "trailing")), [5, 6, 7])

    def test_middle_is_centred_and_never_touches_the_extremes_when_it_can_avoid_them(self):
        idx = band_indices(8, 2, "middle")
        self.assertEqual(list(idx), [3, 4])
        self.assertNotIn(0, idx)
        self.assertNotIn(7, idx)

    def test_rank_is_capped_at_matrix_rank(self):
        self.assertEqual(len(band_indices(5, 99, "leading")), 5)

    def test_unknown_band_raises(self):
        with self.assertRaises(ValueError):
            band_indices(4, 2, "sideways")


class TestTruncation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.w = rng.normal(size=(16, 40))

    def test_full_rank_truncation_is_a_noop(self):
        w_r, rep = truncate_matrix(self.w, min(self.w.shape), "leading")
        self.assertLess(float(np.linalg.norm(w_r - self.w)), 1e-10)
        self.assertAlmostEqual(rep["retained_energy"], 1.0, places=9)

    def test_energy_is_ordered_leading_greater_than_middle_greater_than_trailing(self):
        rng = np.random.default_rng(3)
        w = rng.normal(size=(16, 16)) @ np.diag(np.linspace(8.0, 0.5, 16)) @ rng.normal(size=(16, 16))
        # a decaying spectrum: keeping r directions must cost energy in band order
        r = 4
        e = {band: truncate_matrix(w, r, band)[1]["retained_energy"] for band in lowrank.BANDS}
        self.assertGreater(e["leading"], e["middle"])
        self.assertGreater(e["middle"], e["trailing"])

    def test_rank_one_keeps_exactly_one_singular_direction(self):
        w_r, rep = truncate_matrix(self.w, 1, "trailing")
        s = np.linalg.svd(w_r, compute_uv=False)
        self.assertLess(s[1] / s[0], 1e-10)  # numerically rank 1
        self.assertEqual(rep["sigma_indices"], [min(self.w.shape) - 1])

    def test_biases_pass_through_and_ranks_are_required(self):
        state = {"layers.0.weight": self.w.astype(np.float32), "layers.0.bias": np.ones(16, np.float32)}
        with self.assertRaises(ValueError):
            truncate_state(state, {}, "leading")
        new, summary = truncate_state(state, {"layers.0.weight": 4}, "leading")
        self.assertTrue(np.array_equal(new["layers.0.bias"], state["layers.0.bias"]))
        self.assertEqual(summary["band"], "leading")

    def test_constrained_state_is_refused_not_silently_misread(self):
        state = {"layers.0.u": np.zeros((4, 2)), "layers.0.v": np.zeros((2, 6))}
        with self.assertRaises(ValueError):
            truncate_state(state, {}, "leading")


class TestSplit(unittest.TestCase):
    def test_blocks_never_straddle_splits_and_sizes_follow_fractions(self):
        rows = np.arange(1000 * 3, dtype=float).reshape(1000, 3)
        cfg = SplitConfig(seed=7, block_size=50, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        idx, manifest = block_split(rows, cfg)
        self.assertEqual(len(set(idx["train"]) | set(idx["val"]) | set(idx["test"])), 1000)
        self.assertEqual(len(set(idx["train"]) & set(idx["test"])), 0)
        self.assertEqual(manifest["n_blocks"], 20)
        self.assertEqual(manifest["sizes"]["test"], 100)  # 2 of 20 blocks
        self.assertEqual(manifest["sizes"]["val"], 100)
        self.assertEqual(manifest["sizes"]["train"], 800)
        # every block lies entirely on one side of the split
        blocks = {name: {int(i) // cfg.block_size for i in sel} for name, sel in idx.items()}
        self.assertEqual(sum(len(v) for v in blocks.values()), manifest["n_blocks"])
        self.assertTrue(blocks["train"].isdisjoint(blocks["test"]))
        self.assertTrue(blocks["val"].isdisjoint(blocks["test"]))

    def test_split_is_seed_deterministic(self):
        rows = np.random.default_rng(1).normal(size=(400, 3))
        cfg = SplitConfig(seed=3, block_size=20, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        a, _ = block_split(rows, cfg)
        b, _ = block_split(rows, cfg)
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))

    def test_fractions_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            SplitConfig(seed=1, block_size=10, train_frac=0.8, val_frac=0.1, test_frac=0.3)


class TestModelConfig(unittest.TestCase):
    def test_layer_shapes_are_the_weight_matrices_we_decompose(self):
        cfg = ModelConfig(in_dim=21, out_dim=7, hidden=(64, 64))
        self.assertEqual(cfg.layer_shapes, [(64, 21), (64, 64), (7, 64)])

    def test_constrained_mode_validates_ranks(self):
        with self.assertRaises(ValueError):
            ModelConfig(in_dim=21, out_dim=7, hidden=(64, 64), mode="constrained")
        with self.assertRaises(ValueError):
            ModelConfig(in_dim=21, out_dim=7, hidden=(64, 64), mode="constrained", ranks=(1, 1, 9))


if __name__ == "__main__":
    unittest.main()
