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
from .model import (
    ModelConfig,
    build_model,
    numpy_state,
    spectra,
    state_from_model,
    trainable_params,
)

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
    scale = "" if args.mode != "lora" else f"-s{str(args.lora_scale).rstrip('0').rstrip('.')}"
    fit = "" if args.train_on == "train" else f"-fit{args.train_on}"
    warm = "" if args.warm_start is None else "-warm"
    frac = "-".join(str(int(round(f * 100))) for f in args.fractions)
    return (
        f"{args.mode}-{arch}-seed{args.seed}-blk{args.block_size}-f{frac}"
        f"-ep{args.epochs}-lr{str(args.lr).rstrip('0').rstrip('.')}{tag}-ranks{ranks}{scale}{fit}{warm}"
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
    base_record, base_state = load_base(args)
    model_cfg = ModelConfig(
        in_dim=data.x.shape[1],
        out_dim=data.y.shape[1],
        hidden=tuple(args.hidden),
        mode=args.mode,
        ranks=None if args.ranks is None else tuple(args.ranks),
        lora_scale=args.lora_scale,
    )
    model = build_model(model_cfg, args.seed, base_state)
    if args.mode == "full" and base_state is not None:  # warm start: same shapes, trainable
        model.load_state_dict({k: v.clone() for k, v in base_state.items()})
    init_state = numpy_state(model.state_dict())
    opt = torch.optim.Adam(trainable_params(model), lr=args.lr)
    fit_split = args.train_on
    select_split = "val" if fit_split == "train" else "train"
    y = torch.as_tensor(data.y[data.idx[fit_split]], dtype=torch.float32)
    x = torch.as_tensor(data.x[data.idx[fit_split]], dtype=torch.float32)
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
        val = evaluate_split(model, data, select_split)
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
    # the increment studies need W_init; store it once, next to the trained weights
    np.savez(out_dir / "weights_init.npz", **init_state)

    record = {
        "run_id": run_id,
        "regime": {
            ("full", False): "trained-unconstrained",
            ("full", True): "trained-full-finetune",
            ("constrained", False): "trained-under-rank-constraint",
            ("lora", False): "trained-lora-increment",
        }[(args.mode, args.warm_start is not None)],
        "frozen_base_run": None if base_record is None else base_record["run_id"],
        "warm_start_run": None if args.warm_start is None else base_record["run_id"],
        "fit_split": fit_split,
        "selection_split": select_split,
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
                "selection": f"best normalised MSE on the {select_split} blocks (checkpoint, not final epoch)",
                "hidden_dropout": 0.0,
            },
            "data_dir": str(args.data_dir or "data/"),
            "primary_metric": PRIMARY_METRIC,
            "lora": None if args.mode != "lora" else {
                "scale": args.lora_scale,
                "init": "A kaiming, B zero => dW = 0 at step 0",
                "frozen": "base weights and biases (buffers, no gradient)",
                "trainable": "layers.N.a / layers.N.b only",
            },
        },
        "split": data.split_manifest,
        "split_hygiene": check_split_hygiene(data),
        "artifacts": {"weights": weights_path.name, "weights_init": "weights_init.npz", "record": "run.json"},
        "n_params": model.n_params(),
        "n_trainable_params": sum(p.numel() for p in trainable_params(model)),
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


def load_base(args: argparse.Namespace) -> tuple[dict | None, dict | None]:
    """Frozen base weights for --mode lora, or warm-start weights for --mode full.

    Reads the base run's record and its trained weight matrices. For `lora` the base
    is *frozen* (buffers); for `full` it is merely the starting point, so all of it
    keeps training — that arm is the standard "full fine-tuning" reference LoRA is
    usually compared against.
    """
    path = getattr(args, "base_run", None) or getattr(args, "warm_start", None)
    if path is None:
        return None, None
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    record = json.loads(path.read_text())
    weights = np.load(path.parent / record["artifacts"]["weights"])
    state = {k: torch.as_tensor(weights[k], dtype=torch.float32) for k in weights.files}
    if tuple(record["config"]["model"]["hidden"]) != tuple(args.hidden):
        raise SystemExit(f"--hidden must match the base run {tuple(record['config']['model']['hidden'])}")
    if record["seed"] != args.seed:
        print(f"note: base run seed {record['seed']} != this seed {args.seed}")
    return record, state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # anything that moves a result is required, never defaulted silently
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--block-size", type=int, required=True, help="contiguous rows per split block")
    p.add_argument("--fractions", type=float, nargs=3, required=True, metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="hidden layer widths")
    p.add_argument("--mode", choices=("full", "constrained", "lora"), required=True)
    p.add_argument("--ranks", type=int, nargs="+", default=None, help="per-layer rank caps, constrained/lora")
    p.add_argument("--base-run", default=None, help="run.json whose weights are FROZEN, --mode lora")
    p.add_argument("--warm-start", default=None, help="run.json to initialise from, all weights trainable")
    p.add_argument("--train-on", choices=("train", "val"), default="train",
                   help="fit on the val blocks instead: a shifted downstream task for "
                        "adapter experiments (selection then falls back to the train blocks)")
    p.add_argument("--lora-scale", type=float, default=1.0, help="scale of the LoRA increment (alpha/r style)")
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
    if args.mode in ("constrained", "lora") and args.ranks is None:
        raise SystemExit(f"--mode {args.mode} requires explicit --ranks (one per layer)")
    if args.mode == "full" and args.ranks is not None:
        raise SystemExit("--ranks belongs to --mode constrained/lora; post-hoc truncation is evaluate.py")
    if args.mode == "lora" and not args.base_run:
        raise SystemExit("--mode lora requires --base-run <path/to/run.json> to freeze")
    if args.mode != "lora" and args.base_run:
        raise SystemExit("--base-run is only meaningful with --mode lora (use --warm-start)")
    if args.base_run and args.warm_start:
        raise SystemExit("give at most one of --base-run (frozen) / --warm-start (trainable)")
    if args.fetch:
        fetch(args.data_dir)
    record = train(args)
    print(json.dumps({k: record[k] for k in (
        "run_id", "regime", "frozen_base_run", "warm_start_run", "fit_split", "selection_split",
        "seed", "split", "n_params", "n_trainable_params", "best_epoch", "metrics"
    )}, indent=2))
    print(f"\nrun record: {RESULTS_ROOT / 'runs' / record['run_id'] / 'run.json'}")


if __name__ == "__main__":
    main()
