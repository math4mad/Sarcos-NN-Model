# AGENTS.md

Instructions for AI coding agents working in this repository.
Source of truth for intent: [`README.md`](README.md). Read it before changing anything.

## What this project is

An empirical study, not an application. We train a small MLP regressor on the
SARCOS robot-arm dataset, then take the SVD of each layer's weight matrix and
ask whether the **leading**, **middle**, or **trailing** singular directions
carry the learning effect (see README "Hypothesis" and "Experiment plan").

## Current state (read this first)

The scaffold exists and the first experiments have been run: code in
`src/sarcos_svd/`, numbers in the README **Results** table, findings written up
under it. `data/` and `results/` are present locally but git-ignored, so a clean
checkout needs `python -m sarcos_svd.data --fetch` before anything runs.

Git **is** installed and there **is** a repository (first commit `8b3ce09`).
Check the working tree with `git status` before assuming a file is committed.

## Decisions that are still open

Do not silently pick one of these — surface the choice to the user.

| Question | Status |
| -------- | ------ |
| Framework (NumPy from scratch vs. PyTorch) | **resolved: PyTorch** (per `.agents/neural-network-training`) |
| Dataset source, file format, train/val/test split | **resolved & pinned in `data.py`**: GPML `sarcos_inv*.mat`, 21→7, contiguous-block split (block 256, .8/.1/.1, seeded). The published test file is 98.2% inside train and is *not* used. |
| MLP depth/width (what counts as "each layer") | **one choice made, not compared**: 21-64-64-7, so 3 weight matrices. Still open whether the finding holds for other shapes. |
| Which MSE (normalized? per-replicator? averaged?) | **resolved: primary = per-joint normalised MSE**, raw N·m² mean + all 7 joints per joint always reported alongside |
| License | **resolved: MIT OR Apache-2.0** (dual). `LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`; the SARCOS data is third-party and is not covered |

## Hard rules for experiments

These protect the validity of the study. Violating them produces numbers that
look right and mean nothing.

1. **Fixed, seeded, and logged splits.** One canonical split defined in a
   single loader module. Every run must print/record the seed. Never regenerate
   a split between the baseline run and the truncated runs.
2. **Never tune on test.** Model size, rank `k`, and band boundaries are chosen
   on validation only.
3. **Keep the rank regimes distinct.** Post-hoc truncation of a *trained* net
   (README step 5), training *under* a rank constraint, and training a LoRA-style
   *increment* on a frozen base are three different experiments. Never merge their
   results into one table row, and never compare their numbers without saying which
   split each arm fitted.
4. **Always report retained energy** `‖W_r‖²/‖W‖²` next to test MSE. An MSE
   delta without its energy is uninterpretable. For increment studies the same rule
   applies to `‖ΔW_r‖²/‖ΔW‖²` — and say which of the two it is, because they are
   different claims.
5. **Reproducibility beats speed.** Determinism flags and pinned seeds are part
   of every runner, not optional extras.
6. **Ablations must be single-variable.** Change one band/rank at a time.

## Repository layout (proposed)

```
data/            # inputs, git-ignored
index.qmd      # site landing page (renders headline numbers from results/)
src/sarcos_svd/
  data.py        # canonical loader + split (the ONLY place splits are defined)
  model.py       # MLP definition
  train.py       # training loop, writes run records
  lowrank.py     # SVD truncation: leading / middle / trailing
  evaluate.py    # MSE + retained-energy reporting
  report.py      # aggregate sweeps over seeds into the README table
  note.py        # writes notes/results.qmd from the artifacts (Quarto presentation)
  publish.py     # renders the site into docs/ and stages it on gh-pages
notes/           # results.qmd (source, tracked) + results.html (rendered, ignored)
docs/            # site build output, git-ignored; published via the gh-pages branch
results/         # per-run JSON/CSV summaries, git-ignored
tests/           # stdlib unittest invariants for bands, truncation, splitting
README.md        # human-readable summary; results table lives here
```

Prefer adding files to this shape over inventing new top-level scripts.

## Conventions

- **Python**, 4-space indent, type hints on public functions, `snake_case`
  modules. Keep dependency list minimal and record it in `requirements.txt`.
- Small, readable functions over cleverness; this code is read by a human
  interpreting graphs more than it is executed at scale.
- in Quarto note style. 

## README-specific

- `README.md` is spelled in **all caps**. This volume is case-insensitive
  APFS, so a case-only rename (`mv readme.md README.md`) silently no-ops.
  Always go through a temporary name:
  `mv readme.md tmp.md && mv tmp.md README.md`.
- The **Results table** in README is the human-facing deliverable. When an
  experiment finishes, add a row there; do not restructure the table's columns.
- Do not overwrite the "Hypothesis" or "Experiment plan" sections as a side
  effect of unrelated edits. Update `Status` when the repo state changes.

## Non-goals

- No web UI, API, server, Dockerfile, CI, or CLI framework unless requested.
  **Requested and built since:** a Quarto note and a `gh-pages` site
  (`index.qmd`, `_quarto.yml`, `src/sarcos_svd/publish.py`). Still absent by
  choice: a CI workflow (the site is published from local artifacts, so a runner
  could not reproduce the numbers without re-running the study), and any web app.
- No dataset-wide preprocessing "improvements" (feature engineering, target
  scaling changes) mid-study — they invalidate cross-run comparison.
- Do not commit the SARCOS data or license-restricted downloads without asking.

## Definition of done for a change

1. Runs from a clean checkout with the commands in README "Usage".
2. Produces a run record capturing seed, config, test MSE, retained energy.
3. README updated (row, section, or `Status`) if behaviour or findings changed.
4. Open questions above are still open, or explicitly resolved with the user.

## data set 
Python ML project on the [SARCOS data](https://gaussianprocess.org/gpml/data/) from
Rasmussen & Williams' *GPML* book: the **inverse dynamics** problem of a 7-DOF
anthropomorphic SARCOS robot arm — map a 21-dimensional input space (7 joint
positions, 7 velocities, 7 accelerations) to the 7 joint torques (N·m).
