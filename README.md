# SARCOS Neural Network × SVD Low-Rank Analysis

Baseline MLP regression on the SARCOS robot arm dataset, plus an SVD-based
probe of what the network actually learns: compare the training effect of
keeping only the dominant rank-1 component of the weight matrices vs. the
middle (bulk) vs. the minor (tail) singular directions.

## Dataset

- SARCOS: 7 joint-position inputs → 1 joint torque output, 7 replicators,
  ~44,933 rows.
- Split / preprocessing: **TBD** (to be pinned down in a `load_data.py` script
  so results are reproducible).

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

_Planned interface — no code committed yet._

```bash
pip install -r requirements.txt
python train.py --dataset sarcos --rank 1 --band leading
```

## Results

| Model | Rank | Band | Test MSE | Retained energy |
| ----- | ---- | ---- | -------- | --------------- |
| _tbd_ |      |      |          |                 |

## Status

Draft — no code yet. `README.md` (this file) and `AGENTS.md` (agent working
rules) exist; the experiment scaffold described above is not built.

## License

_Not yet chosen._
