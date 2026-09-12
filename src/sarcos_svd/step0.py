"""Step 0 — the dose–response of covariate shift (Letter 011's registered calibration).

Runs the protocol exactly as `docs/LETTERS/2026-09-11-to-Kairos-floor-accepted-floor-amended.md`
§2 registers it, and nothing else:

* doses `far_frac` in {0.1, 0.25, 0.5, 0.75, 1.0}, block-level builder, canonical
  split (blk 256, .8/.1/.1), seeds 13/14/15;
* **all arms on the base's own schedule (ep60, Adam, lr 1e-3, batch 256)** — the
  exploratory cells compared ep300 adapters against ep60 bases and are therefore
  mirrored as `datum, not verdict`; this table does not repeat that;
* arms per cell: frozen base (scored, not trained), full fine-tuning (warm start),
  LoRA increments at the three registered rank rows, retrained from scratch on the
  shift fit set only;
* a dose *qualifies* iff the from-scratch arm lands within 2× of the full-FT arm
  there — learnability only. Which dose wins the arena is decided by that rule and
  by nothing that makes adaptation look good;
* **every dose is reported whatever it shows**, including "no dose qualifies".

This is calibration, not a hypothesis test: H9-S is registered later, in
`docs/PREREG.md`, after exp6's committed text (Letter 011 §3). No prediction is
scored here and no band is frozen here.

    python -m sarcos_svd.step0 --hidden 256 256
    python -m sarcos_svd.step0 --hidden 64 64
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

from . import evaluate
from .data import ShiftConfig, SplitConfig, _nn_distance, load_split, shift_task
from .train import RESULTS_ROOT, evaluate_indices, train

DOSES = (0.1, 0.25, 0.5, 0.75, 1.0)
SEEDS = (13, 14, 15)
# the three rank rows the exploratory table already used, so the two tables are
# comparable cell for cell (ranks capped per layer by min(shape) in ModelConfig)
RANK_ROWS = ((4, 4, 4), (16, 16, 7), (21, 21, 7))
BASE_EPOCHS = 60  # "bases at their own ep60" — and every adaptation arm too
QUALIFY_RATIO = 2.0  # from-scratch within 2x of full-FT => the region is learnable

PROTOCOL = "Letter 011 §2 (Sarcos→Kairos, 2026-09-11); answers Letter 009 §1 Step 0"


def _base_args(**kw) -> Namespace:
    d = dict(
        seed=13, block_size=256, fractions=(0.8, 0.1, 0.1), hidden=(64, 64),
        mode="full", ranks=None, base_run=None, warm_start=None, train_on="train",
        task="canonical", far_frac=0.5, lora_scale=1.0, epochs=BASE_EPOCHS, lr=1e-3,
        batch_size=256, data_dir=None, run_id=None, tag=None, fetch=False,
    )
    d.update(kw)
    return Namespace(**d)


def block_distance_profile(cfg: SplitConfig, data_dir) -> dict:
    """What 'far' means, printed from the data (Letter 009's warning: the room must
    be able to see the geometry, not trust a quantile)."""
    data = load_split(cfg, data_dir)
    x, bs = data.x_raw, cfg.block_size
    out = {}
    for split in ("val", "test"):
        idx = data.idx[split]
        d = _nn_distance(x[idx], x[data.idx["train"]])
        owner = idx // bs
        per_block = np.array([d[owner == b].mean() for b in np.unique(owner)])
        out[split] = {
            "n_blocks": int(len(per_block)),
            "quantiles": {q: round(float(np.quantile(per_block, q / 100)), 4) for q in (0, 10, 25, 50, 75, 90, 100)},
            "mean": round(float(per_block.mean()), 4),
        }
    return out


def score_frozen(base_run: Path, task: dict, tensors: dict, cfg) -> dict:
    """The frozen base's downstream score: it trains on nothing new, so it is a
    *score*, computed from the recorded weights on this dose's eval blocks.

    Empty slices are reported as None, never as a mean over nothing (`far_frac=1.0`
    leaves no near blocks by construction, `far_frac=0.1` leaves no select blocks).
    """
    record = json.loads(base_run.read_text())
    state = evaluate.state_of(record, base_run)
    model_cfg = evaluate.config_of(record)
    extra = {name: idx for name, idx in (("downstream", task["eval_downstream"]),
                                         ("indistribution", task["eval_indistribution"])) if len(idx)}
    out = evaluate.score_state(model_cfg, state, tensors, record["seed"], extra_splits=extra)
    return {k: out.get(k) for k in ("downstream", "indistribution")}


def run_one(args: Namespace) -> tuple[dict, Path]:
    record = train(args)
    return record, RESULTS_ROOT / "runs" / record["run_id"] / "run.json"


def _ratio(cell: dict) -> float | None:
    """from_scratch / full_finetune downstream MSE for one cell; None if either is absent."""
    arms = cell["arms"]
    a = arms.get("from_scratch", {}).get("downstream")
    b = arms.get("full_finetune", {}).get("downstream")
    return None if (a is None or not b) else round(a / b, 3)


def _score(rec: dict, which: str, rows: int) -> float | None:
    """The primary metric for a named eval slice, or None when the slice is empty.

    `far_frac = 1.0` leaves no near (in-distribution) blocks and `far_frac = 0.1`
    leaves no select blocks; a mean over zero rows is not a number, so it is
    written as null and stays null.
    """
    if rows == 0:
        return None
    return rec["metrics"][which]["mse_normalized"]


def audit_geometry(seeds=SEEDS, hidden=(256, 256), block_size=256, fractions=(.8, .1, .1),
                   data_dir=None) -> dict:
    """A provenance check on bytes that already exist, not a new test (Letter 016 law 5).

    `artifacts/results/sarcos/shift_summary_exploratory.json` `(sha256 16b95129…)` carries
    one frozen-base row per arch, and Letter 011 §1 files the whole table as
    superseded-row-level. This recomputes *both* geometries' downstream score for the same
    deterministic bases at far_frac=0.5 and reports which one produced A's number.

    Why it can differ at all: `report.py::frozen_base_rows` reads `shift_eval.json`, written
    on demand by `evaluate --eval-task shift` with whatever builder is committed *today*,
    while `report.py::shift_rows` reads each arm's own training-time `metrics`. The two row
    families of one table therefore have two code paths and two vintages.
    """
    out = {}
    for seed in seeds:
        run_path = RESULTS_ROOT / "runs" / f"step0-base-{'-'.join(map(str, hidden))}-seed{seed}"                                         f"-blk{block_size}-ep{BASE_EPOCHS}" / "run.json"
        record = json.loads(run_path.read_text())
        cfg, state = evaluate.config_of(record), evaluate.state_of(record, run_path)
        tensors = evaluate.load_tensors(record)
        data = load_split(SplitConfig(seed=seed, block_size=block_size, train_frac=fractions[0],
                                      val_frac=fractions[1], test_frac=fractions[2]), data_dir)
        test = data.idx["test"]
        d = _nn_distance(data.x_raw[test], data.x_raw[data.idx["train"]])
        row_far = test[np.argsort(-d)[: len(test) // 2]]          # the superseded row-level eval set
        blk = shift_task(ShiftConfig(seed=seed, block_size=block_size, train_frac=fractions[0],
                                     val_frac=fractions[1], test_frac=fractions[2], far_frac=0.5))
        sc = evaluate.score_state(cfg, state, tensors, seed,
                                  extra_splits={"rowlevel": row_far, "blocklevel": blk["eval_downstream"]})
        out[str(seed)] = {k: round(v["mse_normalized"], 5) for k, v in sc.items()
                          if k in ("rowlevel", "blocklevel")}
    means = {k: round(statistics.mean(v[k] for v in out.values()), 5) for k in ("rowlevel", "blocklevel")}
    return {"arch": f"21-{'-'.join(map(str, hidden))}-7", "far_frac": 0.5, "per_seed": out,
            "mean_over_seeds": means,
            "note": "compare against the frozen-base row of shift_summary_exploratory.json (16b95129…)"}



def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--fractions", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--doses", type=float, nargs="+", default=list(DOSES))
    p.add_argument("--epochs", type=int, default=BASE_EPOCHS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default=None, help="default results/step0_far<arch>.json")
    p.add_argument("--limit", type=int, default=0, help="run only the first N cells (smoke test)")
    p.add_argument("--audit-geometry", action="store_true",
                   help="run only the geometry provenance check on the step0 base runs (no training)")
    args = p.parse_args(argv)

    if args.audit_geometry:
        print(json.dumps(audit_geometry(tuple(args.seeds), tuple(args.hidden), args.block_size,
                                        tuple(args.fractions), args.data_dir), indent=2))
        return

    arch = "-".join(str(h) for h in args.hidden)
    split_cfg = SplitConfig(seed=args.seeds[0], block_size=args.block_size,
                            train_frac=args.fractions[0], val_frac=args.fractions[1],
                            test_frac=args.fractions[2])
    out_path = Path(args.out or RESULTS_ROOT / f"step0_{arch}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cells: list[dict] = []
    geometry: dict[str, dict] = {}
    for seed in args.seeds:
        sc = SplitConfig(seed=seed, block_size=args.block_size, train_frac=args.fractions[0],
                         val_frac=args.fractions[1], test_frac=args.fractions[2])
        geometry[str(seed)] = block_distance_profile(sc, args.data_dir)

    for seed in args.seeds:
        base_id = f"step0-base-{arch}-seed{seed}-blk{args.block_size}-ep{args.epochs}"
        base_record, base_path = run_one(_base_args(
            seed=seed, block_size=args.block_size, fractions=args.fractions,
            hidden=args.hidden, mode="full", epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, data_dir=args.data_dir, run_id=base_id,
            task="canonical",
        ))
        tensors = evaluate.load_tensors(base_record)
        for dose in args.doses:
            scfg = ShiftConfig(seed=seed, block_size=args.block_size, train_frac=args.fractions[0],
                               val_frac=args.fractions[1], test_frac=args.fractions[2], far_frac=dose)
            task = shift_task(scfg, args.data_dir)
            frozen = score_frozen(base_path, task, tensors, scfg)
            no_select = task["meta"]["sizes"]["select"] == 0
            row = {
                "seed": seed, "far_frac": dose,
                "fit_rows": int(task["meta"]["sizes"]["fit"]),
                "select_rows": int(task["meta"]["sizes"]["select"]),
                "eval_downstream_rows": int(task["meta"]["sizes"]["eval_downstream"]),
                "eval_indistribution_rows": int(task["meta"]["sizes"]["eval_indistribution"]),
                "arms": {
                    "frozen_base": {
                        "downstream": (frozen["downstream"]["mse_normalized"] if frozen["downstream"] else None),
                        "indistribution": (frozen["indistribution"]["mse_normalized"]
                                           if frozen["indistribution"] else None),
                        "trainable_params": 0, "run_id": base_record["run_id"],
                    }
                },
            }
            if no_select:
                # Registered rule: every arm checkpoints on the *select* blocks
                # (best normalised MSE there). At this dose the builder hands back
                # zero select blocks — fit_frac 0.75 of the 2 farthest val blocks
                # leaves nothing — so the treatment is undefined and B does not
                # invent a fallback mid-flight (law 5). The cell is reported empty.
                row["unrunnable_under_registered_rule"] = (
                    "select set is empty at this dose (fit_frac=0.75 consumes every far val "
                    f"block: {task['meta']['sizes']['fit']} fit rows, 0 select rows), so "
                    "best-on-select checkpointing is undefined; arms not trained"
                )
                cells.append(row)
                print(json.dumps({"seed": seed, "far_frac": dose, "arms": "NOT RUNNABLE: empty select set",
                                  "frozen_base_downstream": (None if not frozen["downstream"]
                                                              else round(frozen["downstream"]["mse_normalized"], 5))}))
                continue
            common = dict(block_size=args.block_size, fractions=args.fractions, hidden=args.hidden,
                          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                          data_dir=args.data_dir, far_frac=dose, task="shift")
            # (1) retrained from scratch on the shift fit set only — the learnability probe
            rec, _ = run_one(_base_args(seed=seed, mode="full", run_id=f"step0-scratch-{arch}-seed{seed}-far{int(dose*100)}", **common))
            row["arms"]["from_scratch"] = {
                "downstream": _score(rec, "shift_downstream", task["meta"]["sizes"]["eval_downstream"]),
                "indistribution": _score(rec, "shift_indistribution", task["meta"]["sizes"]["eval_indistribution"]),
                "trainable_params": rec["n_trainable_params"], "run_id": rec["run_id"],
            }
            # (2) full fine-tuning from the base — the ceiling probe
            rec, _ = run_one(_base_args(seed=seed, mode="full", warm_start=str(base_path),
                                        run_id=f"step0-fullft-{arch}-seed{seed}-far{int(dose*100)}", **common))
            row["arms"]["full_finetune"] = {
                "downstream": _score(rec, "shift_downstream", task["meta"]["sizes"]["eval_downstream"]),
                "indistribution": _score(rec, "shift_indistribution", task["meta"]["sizes"]["eval_indistribution"]),
                "trainable_params": rec["n_trainable_params"], "run_id": rec["run_id"],
            }
            # (3) LoRA increments at the three registered rank rows
            for ranks in RANK_ROWS:
                tag = "r" + "".join(str(r) for r in ranks)
                rec, _ = run_one(_base_args(seed=seed, mode="lora", ranks=list(ranks),
                                            base_run=str(base_path),
                                            run_id=f"step0-lora-{tag}-{arch}-seed{seed}-far{int(dose*100)}", **common))
                row["arms"][f"lora_{'_'.join(map(str, ranks))}"] = {
                    "downstream": _score(rec, "shift_downstream", task["meta"]["sizes"]["eval_downstream"]),
                    "indistribution": _score(rec, "shift_indistribution", task["meta"]["sizes"]["eval_indistribution"]),
                    "trainable_params": rec["n_trainable_params"], "run_id": rec["run_id"],
                }
            cells.append(row)
            print(json.dumps({"seed": seed, "far_frac": dose,
                              "downstream": {k: round(v["downstream"], 5) for k, v in row["arms"].items()}}))
            if args.limit and len(cells) >= args.limit:
                break

    # ---- aggregate: mean over seeds, then the learnability-only qualification ----
    table = []
    for dose in args.doses:
        part = [c for c in cells if c["far_frac"] == dose]
        if not part:
            continue

        def mean(arm, key="downstream"):
            vals = [c["arms"][arm][key] for c in part if c["arms"].get(arm, {}).get(key) is not None]
            return round(statistics.mean(vals), 5) if vals else None

        frozen, ft, scratch = mean("frozen_base"), mean("full_finetune"), mean("from_scratch")
        lora_rows = {k: mean(k) for k in part[0]["arms"] if k.startswith("lora_") and mean(k) is not None}
        best_lora = min(lora_rows.values()) if lora_rows else None
        trained = ft is not None and scratch is not None
        qualifies = bool(trained and scratch <= QUALIFY_RATIO * ft)
        if not trained:
            row_note = "no arm trained at this dose under the registered rule (empty select set)"
        table.append({
            "far_frac": dose, "seeds": len(part),
            "downstream_frozen": frozen, "downstream_full_ft": ft,
            "downstream_from_scratch": scratch, "downstream_lora_by_rank": lora_rows,
            "downstream_lora_best": None if best_lora is None else round(best_lora, 5),
            "ft_minus_frozen": None if (ft is None or frozen is None) else round(ft - frozen, 5),
            "best_lora_minus_frozen": None if (best_lora is None or frozen is None) else round(best_lora - frozen, 5),
            "scratch_over_ft": None if not trained else round(scratch / ft, 3),
            # per-seed ratios too: a mean can hide a seed that qualifies on its own
            "scratch_over_ft_per_seed": {str(c["seed"]): _ratio(c) for c in part},
            "seeds_qualifying": sum(1 for c in part
                                    if _ratio(c) is not None and _ratio(c) <= QUALIFY_RATIO),
            "not_run_reason": None if trained else row_note,
            "qualifies_learnable": qualifies,
            "mean_fit_rows": round(statistics.mean(c["fit_rows"] for c in part), 1),
            "mean_eval_downstream_rows": round(statistics.mean(c["eval_downstream_rows"] for c in part), 1),
        })

    blob = {
        "kind": "Sarcos Step 0 dose-response — EXPLORATORY CALIBRATION (Letter 011 §2), not a hypothesis test",
        "protocol": PROTOCOL,
        "arch": f"21-{arch}-7",
        "schedule": {
            "epochs_per_arm": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "note": ("all arms, base included, at the base's own ep60: the exploratory cells "
                     "compared ep300 adapters against ep60 bases (Letter 011 §1) and are mirrored "
                     "as shift_summary_exploratory.json for that reason"),
        },
        "split": {"block_size": args.block_size, "fractions": list(args.fractions), "seeds": list(args.seeds)},
        "ranks_rows": [list(r) for r in RANK_ROWS],
        "qualification_rule": f"far_frac qualifies iff mean(from_scratch) <= {QUALIFY_RATIO} x mean(full_finetune); learnability only, all doses reported",
        "geometry_nn_distance": geometry,
        "table": table,
        "cells": cells,
        "qualifying_doses": [t["far_frac"] for t in table if t["qualifies_learnable"]],
        "verdict": "QUALIFYING DOSE EXISTS" if any(t["qualifies_learnable"] for t in table)
                   else "NO DOSE QUALIFIES — H9-S unmeasurable-on-Sarcos (Letter 011 §2, monument clause)",
        "run_on": "m1-16g (machine B, worker bench)",
        "wall_seconds": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(blob, indent=2))
    print(f"\n{out_path}\n")
    print(json.dumps({"table": table, "qualifying_doses": blob["qualifying_doses"], "verdict": blob["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
