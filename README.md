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
# bands at matched energy; give hit_energy every rank as the ladder, or a coarse
# ladder silently turns "could not reach the target" into a full-rank row
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json \
    --layers 0,1,2 --iso-energy 0.9,0.7,0.5,0.3,0.1 \
    --rungs $(python -c "print(','.join(str(i) for i in range(1,65)))")
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json \
    --layers all --iso-energy 0.9,0.7,0.5 --rungs $(python -c "print(','.join(str(i) for i in range(1,65)))")

# 5. the other regime: trained *under* a rank constraint (never merged with 2-4)
python -m sarcos_svd.train --seed 13 --block-size 256 --fractions .8 .1 .1 \
    --hidden 64 64 --mode constrained --ranks 1 1 1 --epochs 60

# robustness: the same sweep on other shapes
for H in "32" "64" "64 64 64" "128 128"; do
  for S in 13 14 15; do
    python -m sarcos_svd.train --seed $S --block-size 256 --fractions .8 .1 .1 \
        --hidden $H --mode full --epochs 60
  done
done

# aggregate every sweep into one table, with the spread across seeds
python -m sarcos_svd.report --layer all --markdown
python -m sarcos_svd.report --layer 2 --markdown
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
| *same, but bands compared at matched energy (whole net, rank ladder = every rank 1…64)* | | | | |
| ‑ ‑, iso-energy 0.9 | 23–24 | leading | 0.0884 | 0.904 |
| ‑ ‑, iso-energy 0.9 | 59 | middle | 0.5840 | 0.909 |
| ‑ ‑, iso-energy 0.9 | 62 | trailing | 0.5842 | 0.909 |
| ‑ ‑, iso-energy 0.7 | 12 | leading | 0.3693 | 0.715 |
| ‑ ‑, iso-energy 0.7 | 45 | middle | 0.9464 | 0.711 |
| ‑ ‑, iso-energy 0.7 | 55 | trailing | 0.9450 | 0.712 |
| ‑ ‑, iso-energy 0.5 | 6–7 | leading | 0.7775 | 0.528 |
| ‑ ‑, iso-energy 0.5 | 20 | middle | 1.0948 | 0.523 |
| ‑ ‑, iso-energy 0.5 | 37–38 | trailing | 1.0530 | 0.504 |
| *single-variable ablation: one layer truncated, the rest full rank* | | | | |
| ‑ ‑, layer 1 only (64×64) | 16 | leading | 0.1477 | 0.703 |
| ‑ ‑, layer 1 only | 16 | middle | 1.0861 | 0.128 |
| ‑ ‑, layer 1 only | 16 | trailing | 1.0763 | 0.009 |
| ‑ ‑, layer 0 only (64×21) | 16 | leading | 0.0690 | 0.929 |
| ‑ ‑, layer 0 only | 16 | middle | 0.3797 | 0.734 |
| ‑ ‑, layer 0 only | 16 | trailing | 0.7256 | 0.529 |
| ‑ ‑, layer 2 only (7×64, max rank 7) | 4 | leading | 0.1126 | 0.831 |
| ‑ ‑, layer 2 only | 4 | middle | 0.2421 | 0.565 |
| ‑ ‑, layer 2 only | 4 | trailing | 0.8336 | 0.302 |
| *trained **under** a rank constraint — different experiment, different row (rule 3)* | | | | |
| MLP 21-64-64-7, layers trained as `U Vᵀ`, rank 1 | 1/1/1 | n/a | 0.5258 | n/a (never had it) |
| ‑ ‑, rank 4 | 4/4/4 | n/a | 0.1170 | n/a |
| ‑ ‑, rank 16 (last layer capped at 7) | 16/16/7 | n/a | 0.0303 | n/a |

Full per-joint numbers, per-layer energies and the JSON for every cell are in
`results/summary.json` (regenerate with `python -m sarcos_svd.report`).

### Architecture robustness

Same canonical split, optimiser and 3 seeds; only the shape changes (columns
identical to the table above, whole net truncated at one uniform rank rung).
The narrow nets have full rank below rung 32, so their last rung is closer to
untruncated by construction.

| Model | Rank | Band | Test MSE | Retained energy |
| ----- | ---- | ---- | -------- | --------------- |
| MLP 21-32-7 (1 hidden, 935 params), baseline | full (21/7) | — | 0.0412 (0.0362–0.0461) | 1.000 |
| ‑ ‑, truncated | 16 | leading | 0.0595 | 0.985 |
| ‑ ‑ | 16 | middle | 0.4168 | 0.758 |
| ‑ ‑ | 16 | trailing | 0.8901 | 0.569 |
| MLP 21-64-7 (1 hidden, 1,863 params), baseline | full (21/7) | — | 0.0311 (0.0260–0.0352) | 1.000 |
| ‑ ‑, truncated | 16 | leading | 0.0854 | 0.959 |
| ‑ ‑ | 16 | middle | 0.4318 | 0.767 |
| ‑ ‑ | 16 | trailing | 0.8247 | 0.610 |
| MLP 21-64-64-7 (canonical, 6,023 params), baseline | full (64/64/7) | — | 0.0250 (0.0216–0.0285) | 1.000 |
| ‑ ‑, truncated | 32 | leading | 0.0455 | 0.957 |
| ‑ ‑ | 32 | middle | 1.0397 | 0.596 |
| ‑ ‑ | 32 | trailing | 1.0596 | 0.475 |
| MLP 21-64-64-64-7 (3 hidden, 10,183 params), baseline | full | — | 0.0242 (0.0207–0.0284) | 1.000 |
| ‑ ‑, truncated | 32 | leading | 0.0537 | 0.950 |
| ‑ ‑ | 32 | middle | 1.0639 | 0.472 |
| ‑ ‑ | 32 | trailing | 1.0534 | 0.336 |
| MLP 21-128-128-7 (2 hidden, wide, 20,231 params), baseline | full | — | 0.0193 (0.0165–0.0227) | 1.000 |
| ‑ ‑, truncated | 64 | leading | 0.0258 | 0.960 |
| ‑ ‑ | 64 | middle | 1.0645 | 0.522 |
| ‑ ‑ | 64 | trailing | 1.0729 | 0.402 |

### Bands at matched energy, one layer at a time (canonical net)

Smallest rank reaching the target energy in *that* layer, searched over every rank
1…64 (mean of 3 seeds, test normalised MSE; the rank the search landed on across
seeds and the achieved energy are in brackets — every cell is one row of
`results/summary.json`).

| Layer (shape, σ_max/σ_min) | E target | leading | middle | trailing |
| --- | --- | --- | --- | --- |
| 0 (64×21, 3.5) | 0.7 | 0.322 (r9–10, E0.716) | 0.380 (r16, E0.734) | 0.382 (r18–19, E0.740) |
| 0 | 0.5 | 0.570 (r6, E0.539) | 0.665 (r12, E0.532) | 0.726 (r16, E0.529) |
| 0 | 0.3 | 0.887 (r3–4, E0.345) | 0.900 (r8, E0.343) | 0.962 (r12–13, E0.332) |
| 1 (64×64, 239) | 0.9 | **0.052** (r29–30, E0.905) | 0.257 (r61, E0.913) | 0.257 (r63, E0.913) |
| 1 | 0.7 | **0.145** (r16–17, E0.710) | 0.815 (r55, E0.712) | 0.816 (r60, E0.712) |
| 1 | 0.5 | **0.329** (r9, E0.507) | 0.898 (r47, E0.528) | 0.897 (r56, E0.530) |
| 1 | 0.3 | **0.524** (r5, E0.342) | 1.039 (r33, E0.310) | 1.047 (r48–49, E0.312) |
| 2 (7×64, 3.7) | 0.5 | 0.605 (r2, E0.543) | **0.242** (r4, E0.565) | **0.204** (r6, E0.650) |
| 2 | 0.3 | 0.879 (r1, E0.350) | **0.537** (r2–3, E0.339) | 0.590 (r4–5, E0.402) |

Layer 2 has only 7 directions, so targets above 0.5 are reachable only at rank 6–7
(i.e. essentially untruncated) and are omitted rather than reported as a result.

Figure: `results/plots/<run_id>_Liso-2.png` is that table drawn out — the leading
curve sits *above* middle and trailing in the read-out layer, and below them
everywhere else (`_Liso-0.png`, `_Liso-1.png`, `_Lall.png`).
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
   *matched* energy the bands still separate badly (whole net, E≈0.9: leading
   0.088 vs middle 0.584 vs trailing 0.584 — a 6.6× gap at 5% energy tolerance),
   and the gap closes only below E≈0.3, where every band is equally dead. Energy
   is necessary context, not an explanation.
3. **Band sensitivity follows each layer's spectral concentration, and it is the
   bottleneck that bites.** Layer 1 (64×64) has σ_max/σ_min ≈ 239 and dominates
   the whole-net result: at matched energy it is 2–8× better than the other bands
   (E0.5: leading 0.329 vs middle 0.898 vs trailing 0.897, in ranks 9 vs 47 vs 56
   — i.e. leading needs 5× fewer directions for the same energy *and* loses far
   less accuracy). Layer 0 (ratio 3.5) is nearly band-agnostic (E0.7: 0.322 /
   0.380 / 0.382), and at matched energy **middle and trailing are
   indistinguishable everywhere** — they are simply "not the top directions".
   The clean ordering leading < middle < trailing only appears in the flat input
   layer at mid energies (E0.5: 0.570 / 0.665 / 0.726).
4. **The band inversion lives in the read-out layer.** For layer 2 (7×64) at
   matched energy 0.5, *middle* scores **0.242** and *trailing* **0.204** while
   *leading* scores 0.605; at 0.3 it is 0.537 / 0.590 / 0.879. So for the 7-output
   read-out map the dominant singular direction is the *worst* one to keep — the
   only place the README's "middle matters most" survives, and the strongest
   reason to run a dedicated follow-up. Caveat: 7 directions means coarse energy
   matching, and the effect is seen at 2 energy targets on 3 seeds.
5. **Step 5 sanity check: no — the ordering is the other way.** Training *under*
   a rank constraint beats truncating a trained net at the same rank by a wide
   margin: rank 1 → 0.526 vs 1.044; rank 4 → 0.117 vs 1.080; rank 16 → 0.030 vs
   0.175. A net that never had the capacity finds a usable low-rank solution; a
   net that had it and lost it does not.
6. **"Leading wins, middle/trailing die" holds across shapes, and gets worse with
   depth.** Across 1-, 2- and 3-hidden-layer nets (table above), the leading band
   always recovers towards baseline while middle/trailing sit at ≈1.05 once the
   truncation is applied to more than one layer. The 1-hidden-layer nets are the
   partial exception (middle 0.42, trailing 0.89 at rung 16), i.e. a single
   band-truncated bottleneck is partly recoverable whereas stacked ones are not.
7. **Required rank grows with width, not with damage.** 21-128-128-7 needs rung 64
   to reach 0.96 energy and 0.0258 (vs its 0.0193 baseline), where the canonical
   64-wide net needs 32 — so rank must be reported *relative to layer width*, and
   absolute-rank comparisons across architectures are meaningless.

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
  3-seed mean with min/max. Note the coupling: one `--seed` drives *both* the
  block-split permutation and the initialisation, so this spread is split
  variation + init variation together, not init noise alone.
* **Scope of the per-layer ablations**: layers are indexed in forward order
  (`0` = 64×21 input map, `1` = 64×64, `2` = 7×64 read-out), and per-layer
  iso-energy / single-variable rows exist only for the canonical 21-64-64-7 net;
  the architecture table truncates the whole net at one uniform rank.
* Baseline per-joint normalised MSE: `[0.024, 0.029, 0.019, 0.008, 0.041, 0.038,
  0.015]`; the raw mean is dominated by joints 1–2, which is exactly why the
  normalised metric is the headline.

## Status

Phase 3 done: the scaffold and the first study are committed (`cda705b`), and the
headline result — **the leading singular band carries the learning effect; middle
and trailing are indistinguishable and both fatal; retained energy does not
predict function** — now rests on 5 architectures × 3 seeds rather than one net.
`data/` and `results/` stay git-ignored, so a clean checkout needs
`python -m sarcos_svd.data --fetch`.

Still open: the layer-2 middle-band inversion (only 7 directions to work with, so
energy matching is coarse — needs a dedicated run with more seeds and a wider
read-out), the fact that no architecture was *chosen* on validation for the study
(all 5 are reported, none is "the" model), dropout/regularisation untouched, and
no licence has been chosen.

## License

_Not yet chosen._


## Follow. .agents/. directives
