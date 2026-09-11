"""Unit tests for the band/truncation logic (stdlib unittest, no new dependencies).

These guard the definitions the whole study rests on: if ``band_indices`` is wrong,
every number in the README means nothing.

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

from sarcos_svd import lowrank
from sarcos_svd.data import SplitConfig, block_split
from sarcos_svd.lowrank import band_indices, truncate_delta, truncate_matrix, truncate_state
from sarcos_svd.model import ModelConfig, build_model, numpy_state


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


class TestStateCapture(unittest.TestCase):
    """The dW studies compare a trained state against a stored init, so two things
    must hold: the init is reproducible from the seed, and capturing it does not
    alias the live parameters (`tensor.numpy()` shares storage — it silently turns
    the captured 'init' into the final weights)."""

    def test_numpy_state_does_not_alias_parameters(self):
        cfg = ModelConfig(in_dim=4, out_dim=2, hidden=(6,))
        model = build_model(cfg, 0)
        captured = numpy_state(model.state_dict())
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        moved = numpy_state(model.state_dict())
        self.assertTrue(any(not np.array_equal(captured[k], v) for k, v in moved.items()))

    def test_init_is_reproducible_from_the_seed_and_seed_dependent(self):
        cfg = ModelConfig(in_dim=21, out_dim=7, hidden=(64, 64))
        a = numpy_state(build_model(cfg, 5).state_dict())
        b = numpy_state(build_model(cfg, 5).state_dict())
        c = numpy_state(build_model(cfg, 6).state_dict())
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        self.assertFalse(all(np.array_equal(a[k], c[k]) for k in a))


class TestDeltaBands(unittest.TestCase):
    """LoRA's object is the increment, not the matrix: truncating dW must leave the
    frozen base intact and must be measured against ||dW||, never ||W||."""

    def setUp(self):
        rng = np.random.default_rng(2)
        self.w0 = rng.normal(size=(12, 30))
        # a deliberately low-rank increment, so band truncation has something to find
        self.w1 = self.w0 + rng.normal(size=(12, 4)) @ rng.normal(size=(4, 30))

    def test_full_rank_delta_truncation_reproduces_the_trained_matrix(self):
        w_new, rep = truncate_delta(self.w1, self.w0, min(self.w1.shape), "leading")
        self.assertLess(float(np.linalg.norm(w_new - self.w1)), 1e-9)
        self.assertAlmostEqual(rep["delta_retained_energy"], 1.0, places=9)

    def test_rank_one_delta_keeps_the_base_and_one_direction_of_movement(self):
        w_new, rep = truncate_delta(self.w1, self.w0, 1, "leading")
        s = np.linalg.svd(w_new - self.w0, compute_uv=False)
        self.assertLess(s[1] / s[0], 1e-10)
        self.assertEqual(rep["rank_effective"], 1)
        self.assertGreater(rep["delta_over_base_fro"], 0.0)

    def test_delta_bands_separate_the_four_directions_the_increment_actually_has(self):
        # dW has rank 4 inside a 12x30 matrix: leading-4 keeps it all, leading-2 keeps
        # most, and the trailing band falls into the null space of the movement.
        _, lead4 = truncate_delta(self.w1, self.w0, 4, "leading")
        _, lead2 = truncate_delta(self.w1, self.w0, 2, "leading")
        _, tail2 = truncate_delta(self.w1, self.w0, 2, "trailing")
        self.assertGreater(lead4["delta_retained_energy"], 0.999)
        self.assertTrue(0.4 < lead2["delta_retained_energy"] < 1.0)
        self.assertLess(tail2["delta_retained_energy"], 1e-6)


class TestShiftTask(unittest.TestCase):
    """The downstream task must be a *partition* with no contact between what an arm
    fits and what it is scored on - otherwise 'adaptation helped' is leakage."""

    def test_nn_distance_matches_a_brute_force_loop(self):
        from sarcos_svd.data import _nn_distance

        rng = np.random.default_rng(4)
        a, b = rng.normal(size=(7, 3)), rng.normal(size=(11, 3))
        got = _nn_distance(a, b, chunk=3)
        want = np.array([np.min(np.linalg.norm(b - row, axis=1)) for row in a])
        self.assertTrue(np.allclose(got, want))

    def test_partitions_are_disjoint_and_cover_the_test_blocks(self):
        from sarcos_svd.data import ShiftConfig, shift_task

        mat = REPO / "data" / "sarcos_inv.mat"
        if not mat.exists():
            self.skipTest("data/ not fetched")
        cfg = ShiftConfig(seed=13, block_size=256, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        task = shift_task(cfg, REPO / "data")
        fit, sel = set(task["fit"].tolist()), set(task["select"].tolist())
        down, indist = set(task["eval_downstream"].tolist()), set(task["eval_indistribution"].tolist())
        self.assertTrue(fit.isdisjoint(sel))
        self.assertTrue(down.isdisjoint(indist))
        self.assertEqual(len(down | indist), 4608)  # the whole test blocks, no row dropped
        self.assertTrue(fit.isdisjoint(down | indist))  # never fit on what is scored


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
