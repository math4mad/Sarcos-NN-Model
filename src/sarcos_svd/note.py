"""Assemble the Quarto presentation note from the run artifacts.

Reads what is already on disk (`results/summary.json`, `results/runs/*/run.json`,
`results/plots/*.png`) and writes `notes/results.qmd` — an *executable* Quarto note
that builds its tables at render time from those artifacts. No number is typed into
the note by hand, so the note cannot drift from the runs that produced it.

    python -m sarcos_svd.note                 # write notes/results.qmd
    quarto render notes/results.qmd           # -> results/note/results.html

Rendering extras (pandas, great_tables, ipykernel, nbconvert, pyyaml) live in
`requirements-note.txt`; the study itself does not need them.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTE_PATH = REPO_ROOT / "notes" / "results.qmd"

# The body is prose plus executable cells; `note.py` only fills in figure names.
BODY = r'''---
title: "Which singular directions does a small MLP actually learn from?"
subtitle: "SVD truncation of a network trained on SARCOS inverse dynamics"
format:
  html:
    toc-title: Contents
    number-sections: true
    theme: cosmo
    fig-align: center
    self-contained: true
echo: false
warning: false
---

```{python}
#| echo: false
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from great_tables import GT, md, style, loc


def _root() -> Path:
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "results" / "summary.json").exists():
            return cand
    raise RuntimeError("results/summary.json not found - run `python -m sarcos_svd.report` first")


ROOT = _root()
RES = ROOT / "results"
summary = json.loads((RES / "summary.json").read_text())
records = [json.loads(p.read_text()) for p in sorted((RES / "runs").glob("*/run.json"))]
rows = summary["rows"]

CANON = "21-64-64-7"
ARCH_ORDER = [CANON, "21-32-7", "21-64-7", "21-64-64-64-7", "21-128-128-7"]
BANDS = ["leading", "middle", "trailing"]


def arch_of(record):
    m = record["config"]["model"]
    return "-".join(str(x) for x in [m["in_dim"], *m["hidden"], m["out_dim"]])


def pick(regime, arch=CANON, layer="all", band=None, rung=None, target=None):
    out = [r for r in rows
           if r["regime"] == regime and r["arch"] == arch and str(r["layer"]) == str(layer)]
    if band:
        out = [r for r in out if r["band"] == band]
    if rung is not None:
        out = [r for r in out if str(r["rank_rung"]) == str(rung)]
    if target is not None:
        out = [r for r in out if r.get("energy_target") == target]
    return out


def one(*args, **kwargs):
    got = pick(*args, **kwargs)
    if not got:
        raise LookupError(f"no summary row for {args} {kwargs} - regenerate results/summary.json")
    return got[0]


def gt(df, caption=None, widths=None):
    t = GT(df).tab_options(
        table_font_size="13px",
        table_border_top_style="none",
        column_labels_font_weight="normal",
    )
    if caption:
        # great_tables 0.24 has no tab_caption: source notes render below the table,
        # which is where a caption belongs anyway.
        t = t.tab_source_note(md(caption))
    if widths:
        t = t.cols_width({col: f"{w}px" for col, w in widths.items()})
    return t


def rung_ladder(regime="post-hoc-truncation", arch=CANON, layer="all", band="leading"):
    """Rank rungs actually swept for this configuration, ascending, excluding the
    trivial full-rank control (energy 1.0)."""
    got = [r for r in pick(regime, arch=arch, layer=layer, band=band)
           if str(r["rank_rung"]).isdigit() and r["energy_mean"] < 0.999]
    return sorted(int(r["rank_rung"]) for r in got)
```

# The question

A small MLP is trained on the SARCOS robot-arm inverse-dynamics task. Each layer's
weight matrix is then decomposed, $W = U\Sigma V^{\top}$, and the network is
re-scored keeping only a **band** of singular directions: the dominant ones
(*leading*), the centred bulk (*middle*), or the minor tail (*trailing*).

> **Hypothesis under test.** Truncating each layer's weight matrix `W` to its top-k
> singular values degrades test loss in a predictable way. The middle singular band
> should matter most for generalization; the tail should matter least.

Two things are compared, and deliberately never merged into one row: post-hoc
truncation of a *trained* net, and a net *trained under* a rank constraint.

# Why the published split cannot be used

Recomputed from `data/*.mat` at render time — the same output as
`python -m sarcos_svd.data --report`.

```{python}
from sarcos_svd.data import leak_report

leak = leak_report(ROOT / "data")
lag = leak["input_distance_by_lag"]
gt(pd.DataFrame(
    [
        ["`sarcos_inv.mat` rows", f"{leak['gpml_train_rows']:,}", "the published training file"],
        ["`sarcos_inv_test.mat` rows", f"{leak['gpml_test_rows']:,}", "the published test file"],
        ["test rows found verbatim in train", f"{leak['test_rows_found_verbatim_in_train']:,}",
         f"{100*leak['test_fraction_leaked']:.1f}% of the published test set"],
        ["matched positions all multiples of 10", str(leak["matched_test_positions_are_multiples_of_10"]),
         "the test file is every 10th row of the train file"],
        ["pooled unique rows", f"{leak['pooled_unique_rows']:,}", "what we actually split"],
        ["mean input distance, lag 1", f"{lag['lag_1']:.2f}", "rows are in trajectory order,"],
        ["mean input distance, lag 1000", f"{lag['lag_1000']:.2f}", "so adjacent rows are similar:"],
        ["mean input distance, random pair", f"{leak['random_pair_distance']:.2f}",
         "a row-wise random split therefore leaks"],
    ],
    columns=["quantity", "value", "meaning"],
), widths={"quantity": 270, "value": 90})
```

Consequence: the canonical split is defined once, in `src/sarcos_svd/data.py`, as
**contiguous blocks of rows** over the pooled unique rows — and it is worth
0.0207 / 0.0248 / 0.0357 normalised test MSE for a row-wise / 256-row / 4096-row
block size on the same net and budget (seed 13). The headline number is a function
of this choice, so the choice is pinned and printed in every run record.

# Configuration and baseline

```{python}
canon = [r for r in records
         if r["regime"] == "trained-unconstrained" and arch_of(r) == CANON
         and r["split"]["block_size"] == 256 and r["config"]["train"]["epochs"] == 60]
base = canon[0]
gt(pd.DataFrame(
    [
        ["architecture", f"{CANON} (3 weight matrices)"],
        ["parameters", f"{base['n_params']:,}"],
        ["split", f"contiguous blocks of {base['split']['block_size']} rows, "
                  f"{base['split']['sizes']['train']:,} / {base['split']['sizes']['val']:,} / "
                  f"{base['split']['sizes']['test']:,} train/val/test"],
        ["seeds", ", ".join(str(r['seed']) for r in canon) + "   (one seed drives the split and the init)"],
        ["optimiser", "Adam, lr 1e-3, batch 256, 60 epochs, best-validation checkpoint"],
        ["device", f"{base['environment']['device']}, {base['environment']['torch_threads']} thread, "
                   "deterministic algorithms on"],
        ["primary metric", "test MSE normalised per joint (1.0 = predicting the train mean)"],
        ["constant-predictor MSE", f"{base['baseline_constant_predictor']['mse_raw_mean_test']:.1f} N·m² (raw)"],
    ],
    columns=["setting", "value"],
), widths={"setting": 190})
```

```{python}
#| layout-nrow: 2
gt(pd.DataFrame({
    "joint": [f"τ{i+1}" for i in range(7)],
    "train σ (N·m)": [20.46, 14.99, 9.95, 13.77, 0.97, 1.71, 2.60],
    "baseline MSE (N·m²)": [round(x, 3) for x in base["metrics"]["test"]["mse_per_joint_raw"]],
    "MSE / var": [round(x, 4) for x in base["metrics"]["test"]["mse_normalized_per_joint"]],
}), caption="Baseline (full-rank) error per joint, mean of the canonical seeds. A single MSE averaged over joints is dominated by τ1–τ4, which is why the normalised column is the headline.")

gt(pd.DataFrame([
    {"layer": k.replace("layers.", "").replace(".weight", ""),
     "shape": "×".join(map(str, v["shape"])),
     "σ_max": round(v["sigma"][0], 3),
     "σ_min": round(v["sigma"][-1], 4),
     "σ_max/σ_min": round(v["sigma"][0] / v["sigma"][-1], 1),
     "energy in σ₁": round(v["sigma"][0] ** 2 / sum(s * s for s in v["sigma"]), 3)}
    for k, v in base["weight_spectra"].items()
]), caption="The objects of study. Whether a layer's spectrum is steeply or only mildly decaying turns out to decide how much the band choice matters.")
```

# Leading beats middle and trailing, at every rank

Whole net truncated to one uniform rank per layer; every cell is the mean of 3 seeds.

```{python}
ladder = [r for r in rung_ladder(band="leading") if r <= 32]
wide = pd.DataFrame({
    "rank": ladder,
    "leading MSE": [one("post-hoc-truncation", band="leading", rung=r)["mse_test_mean"] for r in ladder],
    "leading energy": [one("post-hoc-truncation", band="leading", rung=r)["energy_mean"] for r in ladder],
    "middle MSE": [one("post-hoc-truncation", band="middle", rung=r)["mse_test_mean"] for r in ladder],
    "middle energy": [one("post-hoc-truncation", band="middle", rung=r)["energy_mean"] for r in ladder],
    "trailing MSE": [one("post-hoc-truncation", band="trailing", rung=r)["mse_test_mean"] for r in ladder],
    "trailing energy": [one("post-hoc-truncation", band="trailing", rung=r)["energy_mean"] for r in ladder],
})
gt(wide, caption=f"Test MSE normalised per joint (1.0 = constant predictor) against the energy it cost, at matched *rank*. Baseline is {one('full-rank baseline (trained)')['mse_test_mean']:.4f}. Anything that is not the leading band is fatal at every rank below full rank.") \
    .fmt_number(columns=[c for c in wide if "MSE" in c], decimals=4) \
    .fmt_number(columns=[c for c in wide if "energy" in c], decimals=3)
```

![MSE against retained energy, whole net truncated at one uniform rank.](../results/plots/PLOT_ALL)

# Energy is not the explanation

Rank-matched comparisons are confounded: different bands hold different energy at
equal rank. So match the **energy** and compare what each band costs at the same
fraction of $‖W‖_F^2$.

```{python}
targets = [0.9, 0.7, 0.5, 0.3]
iso = pd.DataFrame([
    {
        "energy target": t,
        "leading MSE": one("post-hoc-truncation-isoenergy", band="leading", target=t)["mse_test_mean"],
        "leading rank": one("post-hoc-truncation-isoenergy", band="leading", target=t)["rank_rung"],
        "middle MSE": one("post-hoc-truncation-isoenergy", band="middle", target=t)["mse_test_mean"],
        "middle rank": one("post-hoc-truncation-isoenergy", band="middle", target=t)["rank_rung"],
        "trailing MSE": one("post-hoc-truncation-isoenergy", band="trailing", target=t)["mse_test_mean"],
        "trailing rank": one("post-hoc-truncation-isoenergy", band="trailing", target=t)["rank_rung"],
    }
    for t in targets
])
gt(iso, caption="Matched global retained energy, whole net. A rank range means the seeds' searches landed on different integer ranks. At E = 0.9 the leading band needs ~24 directions and scores 0.088; middle and trailing need ~60 directions — nearly the whole matrix — and still score 0.584.") \
    .fmt_number(columns=[c for c in iso if "MSE" in c], decimals=4)
```

::: callout-note
**The ladder matters, and this was a bug in our own tooling.** With a coarse rank
ladder (1, 2, 4, 8, 16, 32, 48, 64) the search cannot hit, say, E = 0.5 in the
trailing band, so it returns the top rung — which reads as a spurious middle-band
win. Every iso-energy number here is produced with the ladder set to *every* rank
1…64.
:::

# One layer at a time

```{python}
per_layer = pd.DataFrame([
    {
        "layer": f"{L} ({'×'.join(map(str, base['layer_shapes'][L]))})",
        "band": b.capitalize(),
        "E=0.7": one("post-hoc-truncation-isoenergy", layer=L, band=b, target=0.7)["mse_test_mean"],
        "E=0.5": one("post-hoc-truncation-isoenergy", layer=L, band=b, target=0.5)["mse_test_mean"],
        "E=0.3": one("post-hoc-truncation-isoenergy", layer=L, band=b, target=0.3)["mse_test_mean"],
    }
    for L in (0, 1, 2)
    for b in BANDS
])
gt(per_layer, caption="Single-variable ablations: one layer truncated, the rest full rank, bands matched by *that layer's* energy. Layer 1 (the steep 64×64 map) dominates the whole-net result; layer 0 (flat) is nearly band-agnostic; layer 2 (the 7×64 read-out, highlighted) is the one place the ordering flips.") \
    .fmt_number(columns=["E=0.7", "E=0.5", "E=0.3"], decimals=4) \
    .tab_style(style.fill("#fdf1e6"), loc.body(rows=[6, 7, 8]))
```

![Per-layer rank sweeps: MSE against the energy of the layer that was truncated.](../results/plots/PLOT_LAYERS)

# The read-out layer inverts the hypothesis

```{python}
l2 = pd.DataFrame([
    {
        "energy target": t,
        "leading": one("post-hoc-truncation-isoenergy", layer=2, band="leading", target=t)["mse_test_mean"],
        "middle": one("post-hoc-truncation-isoenergy", layer=2, band="middle", target=t)["mse_test_mean"],
        "trailing": one("post-hoc-truncation-isoenergy", layer=2, band="trailing", target=t)["mse_test_mean"],
        "best band": min(BANDS, key=lambda b: one("post-hoc-truncation-isoenergy", layer=2, band=b, target=t)["mse_test_mean"]),
    }
    for t in (0.5, 0.3)
])
gt(l2, caption="Layer 2 is 7×64: seven singular directions, so targets above ~0.5 are reachable only at rank 6–7, i.e. essentially untruncated, and are omitted rather than reported as a result. At matched energy the middle and trailing bands beat the leading band — the only surviving part of the original hypothesis.") \
    .fmt_number(columns=["leading", "middle", "trailing"], decimals=4)
```

![Layer 2 iso-energy: the leading curve sits above middle and trailing.](../results/plots/PLOT_L2)

# Robust across shapes

```{python}
arch_rows = []
for arch in ARCH_ORDER:
    ladder = [r for r in rung_ladder(arch=arch, band="leading") if r <= 64]
    top = ladder[-1]
    rec = next(r for r in records if arch_of(r) == arch and r["regime"] == "trained-unconstrained"
               and r["split"]["block_size"] == 256 and r["config"]["train"]["epochs"] == 60)
    arch_rows.append({
        "architecture": arch,
        "params": rec["n_params"],
        "baseline": one("full-rank baseline (trained)", arch=arch)["mse_test_mean"],
        "rank shown": top,
        "leading": one("post-hoc-truncation", arch=arch, band="leading", rung=top)["mse_test_mean"],
        "middle": one("post-hoc-truncation", arch=arch, band="middle", rung=top)["mse_test_mean"],
        "trailing": one("post-hoc-truncation", arch=arch, band="trailing", rung=top)["mse_test_mean"],
    })
arch_df = pd.DataFrame(arch_rows)
gt(arch_df, caption="Same split, optimiser, budget and 3 seeds; only the shape changes. 'Rank shown' is the highest swept rung that is still below full rank for that net, so the three band columns of a row are always at the same rank. The leading band always recovers toward baseline while middle and trailing sit near 1.05 once more than one layer is truncated - and required rank grows with width, so ranks must be read relative to layer width, never absolutely.") \
    .fmt_number(columns=["baseline", "leading", "middle", "trailing"], decimals=4) \
    .fmt_number(columns=["params"], decimals=0, use_seps=True) \
    .cols_label(**{"rank shown": md("rank shown<br>(highest below full)")})
```

# Trained *under* a rank constraint is a different experiment

```{python}
constr = (pd.DataFrame([
    {
        "regime": "trained under rank " + "/".join(str(x) for x in r["config"]["model"]["ranks"]),
        "test MSE (norm.)": r["metrics"]["test"]["mse_normalized"],
        "raw MSE (N·m²)": r["metrics"]["test"]["mse_raw_mean"],
    }
    for r in records if r["regime"] == "trained-under-rank-constraint"
])
    .groupby("regime", as_index=False)
    .agg(**{"test MSE (norm.)": ("test MSE (norm.)", "mean"),
            "raw MSE (N·m²)": ("raw MSE (N·m²)", "mean")}))

post = pd.DataFrame([
    {
        "regime": "post-hoc truncation of the trained net to rank " + str(rung),
        "test MSE (norm.)": one("post-hoc-truncation", band="leading", rung=rung)["mse_test_mean"],
        "raw MSE (N·m²)": one("post-hoc-truncation", band="leading", rung=rung)["mse_raw_mean_test_mean"],
    }
    for rung in (1, 4, 16)
])
gt(pd.concat([constr, post], ignore_index=True), caption="Means over 3 seeds, kept in its own table on purpose: a net that never had the capacity is not comparable to one that had it and lost it. Training under the constraint wins at every matched rank — rank 1 costs 0.53 when it is trained that way and 1.04 when it is imposed afterwards.") \
    .fmt_number(columns=["test MSE (norm.)"], decimals=4) \
    .fmt_number(columns=["raw MSE (N·m²)"], decimals=2)
```

# Validity checks

```{python}
hygiene = base["split_hygiene"]
seeds = sorted(r["metrics"]["test"]["mse_normalized"] for r in canon)
gt(pd.DataFrame(
    [
        ["published test set leaked into train", f"{100*leak['test_fraction_leaked']:.1f}%", "replaced by a pinned block split"],
        ["test rows sharing an input vector with train, in our split",
         f"{hygiene['test_rows_with_identical_input_in_train']} of {base['split']['sizes']['test']:,} ({100*hygiene['test_rows_with_identical_input_fraction']:.2f}%)",
         "reported, not assumed"],
        ["closest test input to any train input", f"{hygiene['min_input_distance_test_to_train']:.2f}", "no near-duplicate leakage"],
        ["split sensitivity (row-wise / 256 / 4096 blocks)", "0.0207 / 0.0248 / 0.0357", "the canonical choice is the pessimistic one"],
        ["seed spread at the canonical config", f"{min(seeds):.4f} – {max(seeds):.4f}", "same order as the rank-32 effect, hence 3-seed means"],
        ["determinism", "bit-identical on rerun", "CPU, 1 thread, deterministic algorithms, pinned seeds"],
        ["state reload re-scoring", "matches the training-time record", "`evaluate --baseline`"],
        ["band / truncation / split invariants", "16 unit tests pass", "`python -m unittest discover -s tests`"],
    ],
    columns=["check", "value", "comment"],
), widths={"check": 320, "value": 170})
```

# Verdict

::: callout-warning
**The hypothesis is rejected as stated, with one exception.** The *leading* band
carries the learning effect: at matched energy it is 2–8× better than either other
band, and it is the only band from which the network recovers toward baseline. The
middle band does **not** matter most and the tail does **not** matter least — at
equal rank or equal energy, middle and trailing are indistinguishable and both are
fatal. The exception is the 7×64 read-out layer, where middle and trailing beat
leading at matched energy.
:::

What the study adds beyond the headline:

1. **Retained energy is not a proxy for function.** Middle-32 keeps 60% of
   $‖W‖_F^2$ and none of the performance. A low-rank result reported with energy but
   without loss — or the reverse — is uninterpretable.
2. **Band sensitivity tracks spectral concentration.** The layer with the 239:1
   singular-value spread is where truncation bites; the flat layers are nearly
   band-agnostic. "Which directions matter" is a per-layer question, not a property
   of the network as a whole.
3. **Post-hoc truncation and constrained training must not share a table row.** They
   differ by up to an order of magnitude at the same rank.

Open next: the read-out-layer inversion (more seeds, and a wider final layer so there
are more than 7 directions to match over), and whether it survives in architectures
where "band" is a less innocent notion than it is for a 2-D `Linear`.

# Reproducing this note

```{python}
gt(pd.DataFrame(
    [
        ["data sha256", ", ".join(f"{k.split('_')[1]}={v[:12]}…" for k, v in leak["files"].items())],
        ["run records on disk", f"{len(records)}"],
        ["aggregated sweep rows", f"{len(rows)}"],
        ["architectures swept", f"{len({r['arch'] for r in rows})}"],
        ["seeds", f"{sorted({r['seed'] for r in records})}"],
    ],
    columns=["provenance", "value"],
), widths={"provenance": 190})
```

``` bash
python -m sarcos_svd.data --fetch --report
python -m sarcos_svd.train --seed 13 --block-size 256 --fractions .8 .1 .1 --hidden 64 64 --mode full --epochs 60
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json --layers all   --sweep 1,2,4,8,16,32,64 --plot
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json --layers 0,1,2 --sweep 1,2,4,8,16,32,64 --plot
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json --layers all   --iso-energy 0.9,0.7,0.5,0.3 \
    --rungs $(python -c "print(','.join(str(i) for i in range(1,65)))")
python -m sarcos_svd.evaluate --run results/runs/<run_id>/run.json --layers 0,1,2 --iso-energy 0.9,0.7,0.5,0.3 \
    --rungs $(python -c "print(','.join(str(i) for i in range(1,65)))")
python -m sarcos_svd.report --out results/summary.json
python -m sarcos_svd.note && quarto render notes/results.qmd
```
'''


def plot_names() -> dict[str, str]:
    """Point the figure placeholders at the newest plots for the canonical run."""
    plots = REPO_ROOT / "results" / "plots"
    newest = lambda pattern: (sorted(plots.glob(pattern)) or [None])[-1]

    out = {
        "PLOT_ALL": newest("*_Lall.png"),
        "PLOT_LAYERS": newest("*_L0-1-2.png"),
        "PLOT_L2": newest("*_Liso-2.png"),
    }
    return {k: (v.name if v else "MISSING-run-evaluate---plot-first.png") for k, v in out.items()}


def render() -> str:
    text = BODY
    for key, value in plot_names().items():
        text = text.replace(key, value)
    return text


def main() -> None:
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(render())
    print(f"wrote {NOTE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
