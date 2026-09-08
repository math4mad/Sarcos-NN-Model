"""MSE + retained-energy reporting, and the post-hoc truncation sweep.

Metrics, and why there are three of them
---------------------------------------
SARCOS targets are joint torques in N·m and their scales differ by ~20x
(train std: ``[20.5, 15.0, 10.0, 13.8, 0.97, 1.7, 2.6]``). A single MSE averaged
over all seven outputs is therefore *joint 1's* score wearing a disguise: on the
constant-predictor baseline it contributes 414 of 134 total units. We report

* ``mse_raw_mean``      — mean squared error over rows and joints, N·m². Comparable
  with the literature, dominated by the big joints.
* ``mse_per_joint_raw`` — the seven numbers separately. Never aggregated away.
* ``mse_normalized``    — mean over joints of ``mse_joint / var_train_joint``; each
  joint scores against its own variance, so 1.0 means "no better than predicting
  the train mean". **This is the primary metric**; the other two are always written
  alongside it.

Run this module to truncate a trained net and score it, without retraining:

    python -m sarcos_svd.evaluate --run results/runs/<id>/run.json --sweep 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import lowrank
from .model import MLP, ModelConfig

PRIMARY_METRIC = "mse_normalized"


# ------------------------------------------------------------------- metrics


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_var: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    var = np.asarray(y_train_var, dtype=np.float64)
    se = (y_true - y_pred) ** 2
    per_joint = se.mean(axis=0)
    return {
        "n_rows": int(y_true.shape[0]),
        "mse_raw_mean": float(per_joint.mean()),
        "mse_per_joint_raw": [round(float(v), 6) for v in per_joint],
        "rmse_per_joint_raw": [round(float(np.sqrt(v)), 6) for v in per_joint],
        "mse_normalized": float((per_joint / var).mean()),
        "mse_normalized_per_joint": [round(float(v / j), 6) for v, j in zip(per_joint, var)],
        "mae_raw_mean": float(np.abs(y_true - y_pred).mean()),
    }


@torch.no_grad()
def predict(model: MLP, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), batch):
        out.append(model(xt[i : i + batch]).numpy())
    return np.concatenate(out)


def inverse_standardise(y: np.ndarray, stats: dict) -> np.ndarray:
    mu = np.asarray(stats["y_mean"], dtype=np.float64)
    sd = np.asarray(stats["y_std"], dtype=np.float64)
    return np.asarray(y, dtype=np.float64) * sd + mu


# -------------------------------------------------------------------- scoring


def model_from_config(cfg: ModelConfig, seed: int) -> MLP:
    torch.manual_seed(seed)
    return MLP(cfg)


def score_state(
    cfg: ModelConfig,
    state: dict[str, np.ndarray],
    tensors: dict[str, np.ndarray],
    seed: int,
) -> dict:
    """Load a (possibly truncated) state into a fresh copy of the architecture and score it."""
    model = model_from_config(cfg, seed)
    torch_state = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in state.items()}
    incompatible = model.load_state_dict(torch_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:  # pragma: no cover
        raise RuntimeError(f"state mismatch: {incompatible}")

    out = {}
    for split in ("val", "test"):
        idx = f"{split}_idx"
        pred_std = predict(model, tensors["x"][tensors[idx]])
        pred_raw = inverse_standardise(pred_std, tensors["stats"])
        out[split] = prediction_metrics(
            tensors["y_raw"][tensors[idx]], pred_raw, np.asarray(tensors["stats"]["y_var"])
        )
    return out


def load_tensors(run: dict) -> dict:
    from .data import SplitConfig, load_split

    sc = run["split"]
    cfg = SplitConfig(
        seed=sc["seed"],
        block_size=sc["block_size"],
        train_frac=sc["train_frac"],
        val_frac=sc["val_frac"],
        test_frac=sc["test_frac"],
    )
    data = load_split(cfg, run["config"].get("data_dir"))
    return {
        "x": data.x,
        "y_raw": data.y_raw,
        "train_idx": data.idx["train"],
        "val_idx": data.idx["val"],
        "test_idx": data.idx["test"],
        "stats": data.stats,
        "split_manifest": data.split_manifest,
    }


def state_of(run: dict, run_path: Path) -> dict[str, np.ndarray]:
    weights = np.load(run_path.parent / run["artifacts"]["weights"])
    return {k: weights[k] for k in weights.files}


def config_of(run: dict) -> ModelConfig:
    m = run["config"]["model"]
    return ModelConfig(
        in_dim=m["in_dim"],
        out_dim=m["out_dim"],
        hidden=tuple(m["hidden"]),
        mode=m["mode"],
        ranks=None if m["ranks"] is None else tuple(m["ranks"]),
    )


# ---------------------------------------------------------------- the sweeps


def truncation_sweep(
    run: dict,
    run_path: Path,
    ranks: list[int],
    bands: list[str],
    layers: list[int] | None = None,
) -> list[dict]:
    """Post-hoc truncation of a trained net at every (rank, band[, layer]) — one variable per row.

    ``layers=None`` truncates every layer at once (README step 3). Explicit indices truncate
    one layer at a time and leave the rest full-rank, which is the single-variable ablation.
    """
    cfg = config_of(run)
    state = state_of(run, run_path)
    tensors = load_tensors(run)
    full_scores = run["metrics"]
    weight_names = [k for k in state if k.endswith(".weight")]
    targets: list[int | None] = [None] if layers is None else [int(i) for i in layers]
    rows: list[dict] = []
    for chosen in targets:
        for band in bands:
            for r in ranks:
                all_names = [k for k in state if k.endswith(".weight")]
                if chosen is None:
                    caps = {k: min(r, min(state[k].shape)) for k in all_names}
                else:
                    name = weight_names[chosen]
                    caps = {
                        k: (min(r, min(state[k].shape)) if k == name else min(state[k].shape))
                        for k in all_names
                    }
                trunc, summary = lowrank.truncate_state(state, caps, band)
                scores = score_state(cfg, trunc, tensors, run["seed"])
                rows.append(
                    {
                        "regime": "post-hoc-truncation",
                        "layer": "all" if chosen is None else int(chosen),
                        "band": band,
                        "rank_rung": r,
                        "effective_ranks": summary["effective_ranks"],
                        "retained_energy_global": summary["retained_energy_global"],
                        "retained_energy_min_layer": summary["retained_energy_min_layer"],
                        "retained_energy_per_layer": {
                            k: v["retained_energy"] for k, v in summary["per_layer"].items()
                        },
                        "mse_val": scores["val"]["mse_normalized"],
                        "mse_test": scores["test"]["mse_normalized"],
                        "mse_raw_mean_test": scores["test"]["mse_raw_mean"],
                        "mse_per_joint_raw_test": scores["test"]["mse_per_joint_raw"],
                        "delta_vs_fullrank_test": round(
                            scores["test"]["mse_normalized"] - full_scores["test"]["mse_normalized"], 6
                        ),
                    }
                )
    return rows


def iso_energy_rows(
    run: dict,
    run_path: Path,
    energies: list[float],
    bands: list[str],
    rungs: list[int],
    layers: list[int] | None = None,
) -> list[dict]:
    """Same retained energy, different bands: the comparison the hypothesis actually needs.

    ``layers=None`` truncates the whole net together and matches *global* energy;
    explicit indices truncate one layer and match *that layer's* energy.
    """
    cfg = config_of(run)
    state = state_of(run, run_path)
    tensors = load_tensors(run)
    weight_names = [k for k in state if k.endswith(".weight")]
    targets: list[int | None] = [None] if layers is None else [int(i) for i in layers]
    rows = []
    for chosen in targets:
        layer_name = None if chosen is None else weight_names[chosen]
        if layer_name is None:
            ladder = sorted(rungs)
        else:
            ladder = sorted({min(r, min(state[layer_name].shape)) for r in rungs})
        for band in bands:
            for target in energies:
                r, _ = lowrank.hit_energy(state, target, band, ladder, layer=layer_name)
                if layer_name is None:
                    caps = {k: min(r, min(state[k].shape)) for k in weight_names}
                else:
                    caps = {k: (r if k == layer_name else min(state[k].shape)) for k in weight_names}
                trunc, summary = lowrank.truncate_state(state, caps, band)
                scores = score_state(cfg, trunc, tensors, run["seed"])
                rows.append(
                    {
                        "regime": "post-hoc-truncation-isoenergy",
                        "layer": "all" if chosen is None else int(chosen),
                        "band": band,
                        "energy_target": target,
                        "rank_rung": r,
                        "retained_energy_global": summary["retained_energy_global"],
                        "retained_energy_per_layer": {
                            k: v["retained_energy"] for k, v in summary["per_layer"].items()
                        },
                        "effective_ranks": summary["effective_ranks"],
                        "mse_val": scores["val"]["mse_normalized"],
                        "mse_test": scores["test"]["mse_normalized"],
                        "mse_raw_mean_test": scores["test"]["mse_raw_mean"],
                        "mse_per_joint_raw_test": scores["test"]["mse_per_joint_raw"],
                        "delta_vs_fullrank_test": round(
                            scores["test"]["mse_normalized"] - run["metrics"]["test"]["mse_normalized"], 6
                        ),
                    }
                )
    return rows


def init_state_of(run: dict, run_path: Path) -> tuple[dict[str, np.ndarray], str]:
    """The W_init that a trained state should be differenced against.

    Preferred: the `weights_init.npz` written next to the weights. If the run
    predates that, the init is rebuilt from the recorded seed — the same code path
    the trainer used, and the whole pipeline is bit-deterministic, so this is exact
    rather than approximate. `delta_over_base_fro` in the rows is the sanity check:
    a wrong init would make the increment look like an unrelated matrix (ratio ~1).
    """
    name = run.get("artifacts", {}).get("weights_init", "weights_init.npz")
    path = run_path.parent / name
    if path.exists():
        weights = np.load(path)
        return {k: weights[k] for k in weights.files}, "stored"
    model = model_from_config(config_of(run), run["seed"])
    state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path, **{k: v.astype(np.float32) for k, v in state.items()})
    return state, "regenerated-from-seed"


def delta_sweep(
    run: dict,
    run_path: Path,
    ranks: list[int],
    bands: list[str],
    layers: list[int] | None = None,
) -> list[dict]:
    """LoRA's object, post-hoc: truncate ΔW = W_trained − W_init by band, add it back.

    Energy here is relative to ‖ΔW‖², not ‖W‖² — a rank-r slice of the increment is a
    different claim from a rank-r slice of the matrix.
    """
    cfg = config_of(run)
    trained = state_of(run, run_path)
    init, provenance = init_state_of(run, run_path)
    tensors = load_tensors(run)
    weight_names = [k for k in trained if k.endswith(".weight")]
    targets: list[int | None] = [None] if layers is None else [int(i) for i in layers]
    rows = []
    for chosen in targets:
        for band in bands:
            for r in ranks:
                if chosen is None:
                    caps = {k: min(r, min(trained[k].shape)) for k in weight_names}
                else:
                    name = weight_names[chosen]
                    caps = {
                        k: (min(r, min(trained[k].shape)) if k == name else min(trained[k].shape))
                        for k in weight_names
                    }
                trunc, summary = lowrank.truncate_state_delta(trained, init, caps, band)
                scores = score_state(cfg, trunc, tensors, run["seed"])
                rows.append(
                    {
                        "regime": "post-hoc-delta-truncation",
                        "layer": "all" if chosen is None else int(chosen),
                        "band": band,
                        "rank_rung": r,
                        "init_provenance": provenance,
                        "effective_ranks": summary["effective_ranks"],
                        "retained_energy_global": summary["delta_retained_energy_global"],
                        "retained_energy_per_layer": {
                            k: v["delta_retained_energy"] for k, v in summary["per_layer"].items()
                        },
                        "delta_over_base_fro": {
                            k: v["delta_over_base_fro"] for k, v in summary["per_layer"].items()
                        },
                        "mse_val": scores["val"]["mse_normalized"],
                        "mse_test": scores["test"]["mse_normalized"],
                        "mse_raw_mean_test": scores["test"]["mse_raw_mean"],
                        "mse_per_joint_raw_test": scores["test"]["mse_per_joint_raw"],
                        "delta_vs_fullrank_test": round(
                            scores["test"]["mse_normalized"] - run["metrics"]["test"]["mse_normalized"], 6
                        ),
                    }
                )
    return rows


def delta_spectra(run: dict, run_path: Path) -> dict:
    """Is the learned deviation low-rank at all? The premise LoRA borrows, measured."""
    trained = state_of(run, run_path)
    init, provenance = init_state_of(run, run_path)
    out = {"run_id": run["run_id"], "init_provenance": provenance, "layers": {}}
    for name in [k for k in trained if k.endswith(".weight")]:
        out["layers"][name] = lowrank.delta_spectrum(trained[name], init[name])
    return out


def format_table(rows: list[dict], cols: list[str]) -> str:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(c) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols), "  ".join("-" * widths[c] for c in cols)]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def _energy_of(row: dict) -> float:
    layer, per = row.get("layer", "all"), row.get("retained_energy_per_layer")
    if isinstance(layer, int) and per:
        return per.get(f"layers.{layer}.weight", row["retained_energy_global"])
    return row["retained_energy_global"]


def make_plot(rows: list[dict], out_png: Path, label: str) -> None:
    """One panel per truncated layer: test MSE against retained energy, a line per band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({str(r.get("layer", "all")) for r in rows})
    fig, axes = plt.subplots(1, len(layers), figsize=(5.2 * len(layers), 4.2), sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], layers):
        sub = [r for r in rows if str(r.get("layer", "all")) == layer]
        for band, marker in zip(lowrank.BANDS, ("o", "s", "^")):
            pts = sorted((r for r in sub if r["band"] == band), key=_energy_of)
            if not pts:
                continue
            ax.plot(
                [_energy_of(p) for p in pts],
                [p["mse_test"] for p in pts],
                marker=marker,
                label=band,
            )
        ax.axhline(1.0, ls="--", c="grey", lw=1, label="constant predictor")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("log")
        ax.set_xlabel(f"retained energy of truncated part (layer {layer})")
        ax.set_title(f"{label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("test MSE, normalised per joint")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="path to run.json written by sarcos_svd.train")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep", help="comma list of per-layer rank rungs, e.g. 1,2,4,8,16,32")
    g.add_argument("--iso-energy", help="comma list of target global retained energies, e.g. 0.9,0.5,0.1")
    g.add_argument("--single", nargs=2, metavar=("BAND", "RANK"), help="evaluate one truncation")
    g.add_argument("--delta-sweep", help="same ladder, on dW = W_trained - W_init (LoRA's object)")
    g.add_argument("--delta-spectra", action="store_true", help="report the rank/energy profile of dW and exit")
    g.add_argument("--baseline", action="store_true", help="re-score the untruncated run (sanity check)")
    p.add_argument("--bands", default=",".join(lowrank.BANDS), help="comma list subset of leading,middle,trailing")
    p.add_argument(
        "--layers",
        default="all",
        help="'all' (every layer at once) or comma list of layer indices for the single-variable ablation",
    )
    p.add_argument("--rungs", default="1,2,3,4,6,8,12,16,24,32,48,64", help="ladder used by --iso-energy")
    p.add_argument("--plot", action="store_true", help="write results/plots/<run_id>.png")
    p.add_argument("--out", default=None, help="output JSON path (default: next to run.json)")
    args = p.parse_args()

    run_path = Path(args.run).resolve()
    run = json.loads(run_path.read_text())
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    for b in bands:
        if b not in lowrank.BANDS:
            raise SystemExit(f"unknown band {b!r}; choose from {lowrank.BANDS}")

    if args.layers.strip().lower() == "all":
        layers: list[int] | None = None
        layer_tag = "all"
    else:
        layers = [int(i) for i in args.layers.split(",") if i.strip()]
        layer_tag = "-".join(str(i) for i in layers)

    if args.baseline:
        cfg = config_of(run)
        tensors = load_tensors(run)
        state = state_of(run, run_path)
        recheck = score_state(cfg, state, tensors, run["seed"])
        same = np.allclose(recheck["test"]["mse_raw_mean"], run["metrics"]["test"]["mse_raw_mean"], rtol=1e-6)
        print(json.dumps(recheck, indent=2))
        print(f"re-score matches the training-time record: {same}")
        return

    if args.delta_spectra:
        spec = delta_spectra(run, run_path)
        print(json.dumps(spec, indent=2))
        (run_path.parent / "delta_spectra.json").write_text(json.dumps(spec, indent=2))
        return

    if args.single:
        band, rank = args.single[0], int(args.single[1])
        rows = truncation_sweep(run, run_path, [rank], [band], layers=layers)
    elif args.delta_sweep:
        ranks = [int(x) for x in args.delta_sweep.split(",") if x.strip()]
        rows = delta_sweep(run, run_path, ranks, bands, layers=layers)
        layer_tag = "delta-" + layer_tag
    elif args.sweep:
        ranks = [int(x) for x in args.sweep.split(",") if x.strip()]
        rows = truncation_sweep(run, run_path, ranks, bands, layers=layers)
    else:
        energies = [float(x) for x in args.iso_energy.split(",") if x.strip()]
        rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
        rows = iso_energy_rows(run, run_path, energies, bands, rungs, layers=layers)
        layer_tag = "iso-all" if layers is None else "iso-" + "-".join(str(i) for i in layers)

    cols = [
        "regime",
        "layer",
        "band",
        "rank_rung",
        "retained_energy_global",
        "mse_val",
        "mse_test",
        "delta_vs_fullrank_test",
        "mse_raw_mean_test",
    ]
    if rows and "energy_target" in rows[0]:
        cols.insert(4, "energy_target")
    print(format_table(rows, [c for c in cols if c in rows[0]]))

    if args.out:
        out_path = Path(args.out)
    else:
        legs = args.sweep or args.iso_energy or args.delta_sweep or f"{args.single[1]}"
        out_path = run_path.parent / f"truncation_L{layer_tag}_{'_'.join(legs.split(','))}.json"
    out_path.write_text(json.dumps({"run_id": run["run_id"], "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")

    if args.plot:
        png = run_path.parents[2] / "plots" / f"{run['run_id']}_L{layer_tag}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        make_plot(rows, png, run["run_id"])
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
