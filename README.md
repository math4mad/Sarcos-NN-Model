# SARCOS Neural Network × SVD Low-Rank Analysis

Baseline MLP regression on the SARCOS robot arm dataset, plus an SVD-based
probe of what the network actually learns: compare the training effect of
keeping only the dominant rank-1 component of the weight matrices vs. the
middle (bulk) vs. the minor (tail) singular directions.

## Dataset

SARCOS inverse dynamics, from Rasmussen & Williams' *GPML* data
(<https://gaussianprocess.org/gpml/data/>). Verified against the files
(`python -m sarcos_svd.data --report` recomputes all of this):

| File | Variable | Shape | Notes |
| ---- | -------- | ----- | ----- |
| `sarcos_inv.mat` | `sarcos_inv` | 44,484 × 28 | float64, no NaNs |
| `sarcos_inv_test.mat` | `sarcos_inv_test` | 4,449 × 28 | float64, no NaNs |

* 28 columns = **21 inputs** (7 positions, 7 velocities, 7 accelerations) →
  **7 joint torques** (N·m). Pooled unique rows: **44,565**.
* **The published pair is not a usable train/test split.** `sarcos_inv_test.mat`
  is *every 10th row of* `sarcos_inv.mat`, and those rows are still inside the
  training file: 4,368 of 4,449 test rows (98.2%) occur verbatim in train. Any
  MSE computed that way measures memorisation.
* Rows are in trajectory order: mean input distance is 3.6 at lag 1 vs 31.4 for
  random pairs, so even a row-wise random split leaks. We therefore split by
  **contiguous blocks of rows** (`--block-size`), never within a block.

Split / preprocessing are pinned in `src/sarcos_svd/data.py` — the only place a
split is defined. Canonical: pooled unique rows (train file in order + the 81
test-only rows appended), block size **256 rows**, fractions **0.8/0.1/0.1**
(seed 13 → 35,584 / 4,373 / 4,608 rows), inputs and targets z-scored using
**train-split statistics only**. Residual near-duplication after our own split is
reported, not assumed: 4 of 4,608 test rows (0.09%) still share an input vector
with train, minimum input distance to train 0.54.

## Hypothesis

Truncating each layer's weight matrix `W` to its top-k singular values degrades
test loss in a predictable way. The middle singular band should matter most for
generalization; the tail should matter least.

## Experiment plan

1. Train a baseline MLP → record test MSE and parameter count.
2. SVD of every layer's `W = U Σ Vᵀ`; reconstruct `W_r` at rank `r`.
3. Variants: retain the **leading** (`r = 1..k`), **middle**, or **trailing**
   singular directions only.
4. Compare test MSE against retained energy `‖W_r‖² / ‖W‖²`.
5. Sanity check: does rank-1 truncation of a *trained* net beat a net *trained*
   under a rank-1 constraint?

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 0. fetch the data (git-ignored) and print the leakage diagnostic
python -m sarcos_svd.data --fetch --report

# 1. baseline: seeded, CPU-only, torch.use_deterministic_algorithms(True)
python -m sarcos_svd.train --seed 13 --block-size 256 --fractions .8 .1 .1 \
    --hidden 64 64 --mode full --epochs 60

# 2-4. SVD truncation of that trained net, by band, over a rank ladder
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json \
    --layers all --sweep 1,2,4,8,16,32,64 --plot
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json \
    --layers 0,1,2 --sweep 1,2,4,8,16,32,64          # single-variable ablations
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json \
    --layers 0,1,2 --iso-energy 0.9,0.7,0.5,0.3,0.1  # bands at matched energy

# 5. the other regime: trained *under* a rank constraint (never merged with 2-4)
python -m sarcos_svd.train --seed 13 --block-size 256 --fractions .8 .1 .1 \
    --hidden 64 64 --mode constrained --ranks 1 1 1 --epochs 60

# aggregate every sweep into one table, with the spread across seeds
python -m sarcos_svd.report --layer all --markdown
python -m unittest discover -s tests        # band/truncation/split invariants
```

Everything that moves a result (`--seed`, `--block-size`, `--fractions`,
`--mode`, `--ranks`, `--bands`, `--layers`, `--sweep`) is an explicit argument;
run records in `results/runs/<run_id>/run.json` capture config, split manifest,
per-epoch history, per-joint MSE, weight spectra and environment.

## Results

Primary metric is **test MSE normalised per joint**: `mean_j MSE_j / var_train_j`,
so 1.0 = "no better than predicting the train mean" and no joint can hide behind
another. Raw N·m² MSE is reported alongside because the joint scales differ by
~20× (train σ = `[20.5, 15.0, 10.0, 13.8, 0.97, 1.7, 2.6]`). Each cell is the mean
over **3 seeds (13, 14, 15)**; retained energy is the global
`Σ_layers ‖W_r‖²_F / Σ_layers ‖W‖²_F` (for a single-layer row, that layer's energy).
Baseline raw MSE of the constant predictor: **135.5 N·m²**.

| Model | Rank | Band | Test MSE | Retained energy |
| ----- | ---- | ---- | -------- | --------------- |
| *post-hoc truncation of a trained net, all 3 layers at once* | | | | |
| MLP 21-64-64-7, trained unconstrained (baseline) | full (64/64/7) | — | 0.0250 (0.0216–0.0285) | 1.000 |
| ‑ ‑, every layer truncated | 32 | leading | 0.0455 | 0.957 |
| ‑ ‑ | 16 | leading | 0.1750 | 0.808 |
| ‑ ‑ | 8 | leading | 0.6166 | 0.583 |
| ‑ ‑ | 4 | leading | 1.0798 | 0.378 |
| ‑ ‑ | 1 | leading | 1.0444 | 0.128 |
| ‑ ‑ | 32 | middle | 1.0397 | 0.596 |
| ‑ ‑ | 16 | middle | 1.0913 | 0.417 |
| ‑ ‑ | 8 | middle | 1.0758 | 0.248 |
| ‑ ‑ | 1 | middle | 1.0558 | 0.031 |
| ‑ ‑ | 32 | trailing | 1.0596 | 0.475 |
| ‑ ‑ | 16 | trailing | 1.0779 | 0.281 |
| ‑ ‑ | 8 | trailing | 1.0754 | 0.148 |
| ‑ ‑ | 1 | trailing | 1.0570 | 0.006 |
| *same, but bands compared at matched energy (whole net)* | | | | |
| ‑ ‑, iso-energy 0.5 | 6 | leading | 0.8907 | 0.501 |
| ‑ ‑, iso-energy 0.5 | 24 | middle | 1.0701 | 0.548 |
| ‑ ‑, iso-energy 0.5 | 24 | trailing | 1.0711 | 0.450 |
| *single-variable ablation: one layer truncated, the rest full rank* | | | | |
| ‑ ‑, layer 1 only (64×64) | 16 | leading | 0.1477 | 0.703 |
| ‑ ‑, layer 1 only | 16 | middle | 1.0861 | 0.128 |
| ‑ ‑, layer 1 only | 16 | trailing | 1.0763 | 0.009 |
| ‑ ‑, layer 1 only, iso-energy 0.33 | 6 | leading | 0.4729 | 0.389 |
| ‑ ‑, layer 1 only, iso-energy 0.33 | 48 | middle | 0.8998 | 0.529 |
| ‑ ‑, layer 0 only (64×21) | 16 | leading | 0.0690 | 0.929 |
| ‑ ‑, layer 0 only | 16 | middle | 0.3797 | 0.734 |
| ‑ ‑, layer 0 only | 16 | trailing | 0.7256 | 0.529 |
| ‑ ‑, layer 2 only (7×64, max rank 7) | 4 | leading | 0.1126 | 0.831 |
| ‑ ‑, layer 2 only | 4 | middle | 0.2421 | 0.565 |
| ‑ ‑, layer 2 only | 4 | trailing | 0.8336 | 0.302 |
| ‑ ‑, layer 2 only, iso-energy 0.35 | 1 | leading | 0.8787 | 0.350 |
| ‑ ‑, layer 2 only, iso-energy 0.35 | 3 | middle | 0.5540 | 0.352 |
| ‑ ‑, layer 2 only, iso-energy 0.35 | 4 | trailing | 0.7450 | 0.333 |
| *trained **under** a rank constraint — different experiment, different row (rule 3)* | | | | |
| MLP 21-64-64-7, layers trained as `U Vᵀ`, rank 1 | 1/1/1 | n/a | 0.5258 | n/a (never had it) |
| ‑ ‑, rank 4 | 4/4/4 | n/a | 0.1170 | n/a |
| ‑ ‑, rank 16 (last layer capped at 7) | 16/16/7 | n/a | 0.0303 | n/a |

Full per-joint numbers, per-layer energies and the JSON for every cell are in
`results/summary.json` (regenerate with `python -m sarcos_svd.report`).

### Findings

1. **The leading band carries the learning effect.** Post-hoc truncation of the
   whole net to rank 32 *leading* keeps test MSE within 2× of baseline
   (0.0455 vs 0.0250). Rank 32 *middle* (0.596 energy) or *trailing* (0.476
   energy) lands at ≈1.04–1.06 — no better, sometimes worse, than predicting the
   train mean. The hypothesis's "middle matters most" is **rejected**; so is "tail
   matters least": at equal rank, dropping the leading directions is fatal no
   matter which band you keep instead.
2. **Retained energy does not track function** (rule 4 is what exposes this).
   Middle-32 keeps 60% of the energy and 0% of the performance. Conversely, at
   *matched* energy the bands still separate (E≈0.5: leading 0.891 vs middle
   1.070 vs trailing 1.071). Energy is necessary context, not an explanation.
3. **Band sensitivity follows each layer's spectral concentration.** Layer 1
   (64×64) has σ_max/σ_min ≈ 239 and is where truncation hurts most; layer 0
   (64×21, ratio 3.5) and layer 2 (7×64, ratio 3.7) have flat-ish spectra, so
   their bands differ far less, and layer 2 saturates by rank 7 because that is
   its full rank.
4. **The one place the middle band wins is the output layer.** At matched energy
   ≈0.35 on layer 2, middle (0.554) beats both leading (0.879) and trailing
   (0.745) — i.e. for the 7×64 read-out map, the *bulk* directions generalise
   better than the dominant ones. This is a single layer at coarse rank
   granularity (rank ≤ 7), so treat it as the lead for a follow-up, not a
   conclusion.
5. **Step 5 sanity check: no — the ordering is the other way.** Training *under*
   a rank constraint beats truncating a trained net at the same rank by a wide
   margin: rank 1 → 0.526 vs 1.044; rank 4 → 0.117 vs 1.080; rank 16 → 0.030 vs
   0.175. A net that never had the capacity finds a usable low-rank solution; a
   net that had it and lost it does not.

### Validity checks (reported so the split choice is not load-bearing in secret)

* **Determinism**: rerunning seed 13/blk 256/ep 60 reproduces `metrics` and
  `weight_spectra` bit-for-bit; `evaluate --baseline` re-scores a stored state
  and matches the training-time record.
* **Split sensitivity** (seed 13, ep 60, normalised test MSE): row-wise random
  split 0.0207 < block-256 **0.0248 (canonical)** < block-4096 0.0357. A row-wise
  random split is ~17% optimistic, which is why blocks are used; the canonical
  number is the pessimistic-but-honest one.
* **Seed spread** at the canonical config: 0.0216 / 0.0248 / 0.0285 (±14% of the
  mean) — the same order as the rank-32 leading effect, hence every row is a
  3-seed mean with min/max.
* Baseline per-joint normalised MSE: `[0.024, 0.029, 0.019, 0.008, 0.041, 0.038,
  0.015]`; the raw mean is dominated by joints 1–2, which is exactly why the
  normalised metric is the headline.

## Status

Phase 2 done: scaffold built (`src/sarcos_svd/{data,model,train,lowrank,evaluate,report}.py`,
`tests/`), environment pinned (`requirements.txt`, `.venv`), `.gitignore` added so
`data/` and `results/` stay untracked. Baseline + post-hoc truncation sweeps
(3 seeds, 3 bands × 7 ranks, whole-net and per-layer, plus iso-energy) and the
trained-under-constraint regime are all run; numbers are in the table above.
Still open: MLP depth/width was fixed at one architecture without comparison to
others, layer 2's middle-band lead needs a dedicated follow-up, and no licence
has been chosen.

## License

_Not yet chosen._


## Follow. .agents/. directives
