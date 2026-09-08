"""Training runner. Writes a run record capturing everything that shaped the numbers.

Determinism is not an option flag here (AGENTS.md rule 5): CPU-only float32,
``torch.use_deterministic_algorithms(True)`` and a single thread, so a rerun with the
same ``--seed`` and split args reproduces the record bitwise. MPS is deliberately left
alone; this net is small enough that CPU is not a bottleneck.

    python -m sarcos_svd.train --seed 13 --block-size 256 --fractions .8 .1 .1 \
        --hidden 64 64 --epochs 30 --mode full

    # the *other* regime: trained with each layer capped at rank r
    python -m sarcos_svd.train --seed 13 --block-size 256 --fractions .8 .1 .1 \
        --hidden 64 64 --epochs 30 --mode constrained --ranks 1 4 7
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import SplitConfig, check_split_hygiene, fetch, load_split
from .evaluate import PRIMARY_METRIC, inverse_standardise, prediction_metrics
from .model import ModelConfig, build_model, numpy_state, spectra, state_from_model

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def env_record() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "deterministic_algorithms": True,
        "torch_threads": torch.get_num_threads(),
    }


def make_run_id(args: argparse.Namespace) -> str:
    arch = "h".join(str(h) for h in args.hidden)
    ranks = "none" if args.ranks is None else "-".join(str(r) for r in args.ranks)
    tag = f"_{args.tag}" if args.tag else ""
    frac = "-".join(str(int(round(f * 100))) for f in args.fractions)
    return (
        f"{args.mode}-{arch}-seed{args.seed}-blk{args.block_size}-f{frac}"
        f"-ep{args.epochs}-lr{str(args.lr).rstrip('0').rstrip('.')}{tag}-ranks{ranks}"
    )


def evaluate_split(model: torch.nn.Module, data, split: str) -> dict:
    idx = data.idx[split]
    model.eval()
    with torch.no_grad():
        pred_std = model(torch.as_tensor(data.x[idx], dtype=torch.float32)).numpy()
    pred_raw = inverse_standardise(pred_std, data.stats)
    var = np.asarray(data.stats["y_var"])
    return prediction_metrics(data.y_raw[idx], pred_raw, var)


def train(args: argparse.Namespace) -> dict:
    configure_determinism(args.seed)
    t0 = time.time()

    split_cfg = SplitConfig(
        seed=args.seed,
        block_size=args.block_size,
        train_frac=args.fractions[0],
        val_frac=args.fractions[1],
        test_frac=args.fractions[2],
    )
    data = load_split(split_cfg, args.data_dir)
    model_cfg = ModelConfig(
        in_dim=data.x.shape[1],
        out_dim=data.y.shape[1],
        hidden=tuple(args.hidden),
        mode=args.mode,
        ranks=None if args.ranks is None else tuple(args.ranks),
    )
    model = build_model(model_cfg, args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    y = torch.as_tensor(data.y[data.idx["train"]], dtype=torch.float32)
    x = torch.as_tensor(data.x[data.idx["train"]], dtype=torch.float32)
    n = x.shape[0]
    step_rng = np.random.default_rng(args.seed + 1)  # shuffling stream, separate from init

    best = {"mse_val": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = step_rng.permutation(n)
        totals, seen = 0.0, 0
        for start in range(0, n, args.batch_size):
            b = torch.as_tensor(perm[start : start + args.batch_size], dtype=torch.long)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x[b]), y[b])
            loss.backward()
            opt.step()
            totals += loss.detach().item() * len(b)
            seen += len(b)
        val = evaluate_split(model, data, "val")
        history.append({"epoch": epoch, "train_loss_std_mse": totals / seen, "mse_val": val["mse_normalized"]})
        if val["mse_normalized"] < best["mse_val"]:
            best = {"mse_val": val["mse_normalized"], "epoch": epoch, "state": state_from_model(model)}

    # early-stopping checkpoint: the *best validation* state is the run's model
    model.load_state_dict({k: v.clone() for k, v in best["state"].items()})
    state = numpy_state(best["state"])
    metrics = {s: evaluate_split(model, data, s) for s in ("train", "val", "test")}

    run_id = args.run_id or make_run_id(args)
    out_dir = RESULTS_ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.npz"
    np.savez(weights_path, **state)

    record = {
        "run_id": run_id,
        "regime": ("trained-under-rank-constraint" if args.mode == "constrained" else "trained-unconstrained"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t0, 2),
        "seed": args.seed,
        "config": {
            "model": asdict(model_cfg),
            "train": {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "optimizer": "Adam",
                "loss": "mse on standardised targets",
                "selection": "best validation normalised MSE (checkpoint, not final epoch)",
                "hidden_dropout": 0.0,
            },
            "data_dir": str(args.data_dir or "data/"),
            "primary_metric": PRIMARY_METRIC,
        },
        "split": data.split_manifest,
        "split_hygiene": check_split_hygiene(data),
        "artifacts": {"weights": weights_path.name, "record": "run.json"},
        "n_params": model.n_params(),
        "layer_shapes": [list(s) for s in model_cfg.layer_shapes],
        "metrics": metrics,
        "baseline_constant_predictor": {
            "mse_normalized": 1.0,
            "mse_raw_mean_test": float(
                np.mean((data.y_raw[data.idx["test"]] - data.y_raw[data.idx["train"]].mean(0)) ** 2)
            ),
        },
        "best_epoch": best["epoch"],
        "history": history,
        "weight_spectra": spectra(best["state"]),
        "environment": env_record(),
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2))
    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # anything that moves a result is required, never defaulted silently
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--block-size", type=int, required=True, help="contiguous rows per split block")
    p.add_argument("--fractions", type=float, nargs=3, required=True, metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="hidden layer widths")
    p.add_argument("--mode", choices=("full", "constrained"), required=True)
    p.add_argument("--ranks", type=int, nargs="+", default=None, help="per-layer rank caps, --mode constrained only")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--fetch", action="store_true", help="download the .mat files first")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode == "constrained" and args.ranks is None:
        raise SystemExit("--mode constrained requires explicit --ranks (one per layer)")
    if args.mode == "full" and args.ranks is not None:
        raise SystemExit("--ranks belongs to --mode constrained; post-hoc truncation is evaluate.py")
    if args.fetch:
        fetch(args.data_dir)
    record = train(args)
    print(json.dumps({k: record[k] for k in (
        "run_id", "regime", "seed", "split", "split_hygiene", "n_params", "best_epoch", "metrics"
    )}, indent=2))
    print(f"\nrun record: {RESULTS_ROOT / 'runs' / record['run_id'] / 'run.json'}")


if __name__ == "__main__":
    main()
