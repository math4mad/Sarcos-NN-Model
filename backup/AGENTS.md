# AGENTS.md

Instructions for AI coding agents working in this repository.
Source of truth for intent: [`README.md`](README.md). Read it before changing anything.

## What this project is

An empirical study, not an application. We train a small MLP regressor on the
SARCOS robot-arm dataset, then take the SVD of each layer's weight matrix and
ask whether the **leading**, **middle**, or **trailing** singular directions
carry the learning effect (see README "Hypothesis" and "Experiment plan").

## Current state (read this first)

**There is no code in this repo.** Only `README.md` and this file. If you are
asked to "fix", "refactor" or "run" something, it probably does not exist yet —
say so instead of inventing a file, and start by scaffolding it.

Also: **git is not installed** on the host and there is no repository yet. Do
not assume `git` commands will work; check with `git rev-parse` first.

## Decisions that are still open

Do not silently pick one of these — surface the choice to the user.

| Question | Status |
| -------- | ------ |
| Framework (NumPy from scratch vs. PyTorch) | undecided |
| Dataset source, file format, train/val/test split | **TBD** in README |
| MLP depth/width (what counts as "each layer") | undecided |
| Which MSE (normalized? per-replicator? averaged?) | undecided |
| License | none chosen |

## Hard rules for experiments

These protect the validity of the study. Violating them produces numbers that
look right and mean nothing.

1. **Fixed, seeded, and logged splits.** One canonical split defined in a
   single loader module. Every run must print/record the seed. Never regenerate
   a split between the baseline run and the truncated runs.
2. **Never tune on test.** Model size, rank `k`, and band boundaries are chosen
   on validation only.
3. **Keep the two rank regimes distinct.** Post-hoc truncation of a *trained*
   net (README step 5) is a different experiment from training *under* a rank
   constraint. Never merge their results into one table row.
4. **Always report retained energy** `‖W_r‖²/‖W‖²` next to test MSE. An MSE
   delta without its energy is uninterpretable.
5. **Reproducibility beats speed.** Determinism flags and pinned seeds are part
   of every runner, not optional extras.
6. **Ablations must be single-variable.** Change one band/rank at a time.

## Repository layout (proposed)

```
data/            # inputs, git-ignored
src/sarcos_svd/
  data.py        # canonical loader + split (the ONLY place splits are defined)
  model.py       # MLP definition
  train.py       # training loop, writes run records
  lowrank.py     # SVD truncation: leading / middle / trailing
  evaluate.py    # MSE + retained-energy reporting
results/         # per-run JSON/CSV summaries, git-ignored
README.md        # human-readable summary; results table lives here
```

Prefer adding files to this shape over inventing new top-level scripts.

## Conventions

- **Python**, 4-space indent, type hints on public functions, `snake_case`
  modules. Keep dependency list minimal and record it in `requirements.txt`.
- Small, readable functions over cleverness; this code is read by a human
  interpreting graphs more than it is executed at scale.
- Every script that produces numbers must accept a `--seed` and emit a run
  record describing its configuration.
- No silent defaults for anything that affects a result (rank, band, split,
  normalization). Make them explicit CLI args or config fields.
- Plots/tables go in `results/`, never committed as binaries alongside code.

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
- No dataset-wide preprocessing "improvements" (feature engineering, target
  scaling changes) mid-study — they invalidate cross-run comparison.
- Do not commit the SARCOS data or license-restricted downloads without asking.

## Definition of done for a change

1. Runs from a clean checkout with the commands in README "Usage".
2. Produces a run record capturing seed, config, test MSE, retained energy.
3. README updated (row, section, or `Status`) if behaviour or findings changed.
4. Open questions above are still open, or explicitly resolved with the user.
