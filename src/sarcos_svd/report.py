"""Aggregate truncation sweeps over seeds and print the README-ready table.

One run record is one number; the seed-to-seed spread of this small net is the same
order as many of the effects we are measuring, so nothing here is reported without
the range across seeds. Rows are keyed by
``(regime, layer, band, rank_rung)`` and every metric carries ``mean/min/max/n``.

    python -m sarcos_svd.report --runs 'results/runs/*/run.json' \
        --sweeps 'results/runs/*/truncation_*.json' --markdown
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("mse_test", "mse_val", "retained_energy_global", "mse_raw_mean_test", "delta_vs_fullrank_test")
KEY = ("arch", "regime", "layer", "band", "rank_rung", "energy_target")



def _load_patterns(patterns: list[str]) -> list[Path]:
    found: dict[str, None] = {}
    for pat in patterns:
        for match in sorted(glob.glob(pat)):
            found.setdefault(match, None)
    return [Path(p) for p in found]


def arch_label(record: dict) -> str:
    """'21-64-64-7' — the architecture a row belongs to, so a wider/deeper net is
    never silently averaged together with the canonical one."""
    m = record["config"]["model"]
    return "-".join(str(x) for x in [m["in_dim"], *m["hidden"], m["out_dim"]])


def collect(runs: list[Path], sweeps: list[Path]) -> tuple[list[dict], list[dict], set[str]]:
    records = [json.loads(p.read_text()) for p in runs]
    baselines = {r["run_id"]: r for r in records}
    rows: list[dict] = []
    swept_run_ids: set[str] = set()
    for path in sweeps:
        blob = json.loads(Path(path).read_text())
        swept_run_ids.add(blob.get("run_id"))
        run = baselines.get(blob.get("run_id"))
        seed = run["seed"] if run else blob.get("seed", "?")
        arch = arch_label(run) if run else "?"
        for row in blob["rows"]:
            row.setdefault("layer", "all")
            # iso-energy rows can land on the same rung for different energy targets;
            # keeping the target in the grouping key stops those rows merging.
            row.setdefault("energy_target", "-")
            rows.append({**row, "seed": seed, "arch": arch, "run_id": blob.get("run_id")})
    return records, [r for r in rows if r["run_id"] in swept_run_ids], swept_run_ids


def selected_energy(row: dict) -> float | None:
    """Energy of what was actually truncated: the single layer for an ablation row,
    the whole net for an all-layer row. Global energy dilutes a one-layer change and
    would hide the effect being measured."""
    layer, per = row.get("layer"), row.get("retained_energy_per_layer")
    if isinstance(layer, int) and per:
        return per.get(f"layers.{layer}.weight")
    return row.get("retained_energy_global")


def group_key(row: dict) -> tuple:
    """Iso-energy rows are grouped by the energy *target*, not by the rank found: the
    ladder search lands on different integer ranks per seed, and splitting on that
    would bury the 3-seed mean under several half-populated rows."""
    rung = row["rank_rung"] if not str(row["regime"]).startswith("post-hoc-truncation-iso") else "-"
    return (row["arch"], row["regime"], row["layer"], row["band"], rung, row["energy_target"])


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    out = []
    for key, part in groups.items():
        entry = dict(zip(KEY, key))
        if str(entry.get("rank_rung")) == "-":  # iso-energy: show the rank range found
            found = sorted({int(p["rank_rung"]) for p in part})
            entry["rank_rung"] = str(found[0]) if len(found) == 1 else f"{found[0]}-{found[-1]}"
        entry["seeds"] = sorted(p["seed"] for p in part)
        for metric in METRICS:
            vals = [p[metric] for p in part if metric in p and p[metric] is not None]
            if not vals:
                continue
            entry[f"{metric}_mean"] = round(statistics.fmean(vals), 6)
            entry[f"{metric}_min"] = round(min(vals), 6)
            entry[f"{metric}_max"] = round(max(vals), 6)
        entry["n_seeds"] = len(part)
        # a band comparison is only honest if energy is comparable
        energies = [e for e in (selected_energy(p) for p in part) if e is not None] or [0.0]
        entry["energy_mean"] = round(statistics.fmean(energies), 6)
        entry["energy_spread"] = round(max(energies) - min(energies), 6)
        out.append(entry)
    order = {
        "full-rank baseline (trained)": 0,
        "post-hoc-truncation": 1,
        "post-hoc-truncation-isoenergy": 2,
        "trained-under-rank-constraint": 3,
    }
    band_order = {"leading": 0, "middle": 1, "trailing": 2, "-": 3}

    def rung_key(value):
        """Sort rank rungs numerically where possible ('full', '16/16/7' sort last/as text)."""
        try:
            return (0, int(value), "")
        except (TypeError, ValueError):
            return (1, 0, str(value))

    out.sort(
        key=lambda e: (
            str(e["arch"]),
            order.get(e["regime"], 9),
            str(e["layer"]),
            band_order.get(e["band"], 9),
            rung_key(e["rank_rung"]),
            str(e["energy_target"]),
        )
    )
    return out


def add_baseline_rows(records: list[dict], swept_run_ids: set[str]) -> list[dict]:
    """Turn run records themselves into table rows: the full-rank baseline, and the
    *trained-under-a-rank-constraint* regime (kept in its own rows, never merged with
    post-hoc truncation — AGENTS.md rule 3).

    Only records that were actually swept contribute baseline rows, so the baseline
    in the table is the same set of nets the truncations were applied to.
    """
    rows = []
    for r in records:
        if r["regime"] == "trained-unconstrained":
            if r["run_id"] not in swept_run_ids:
                continue
            rows.append(
                {
                    "regime": "full-rank baseline (trained)",
                    "arch": arch_label(r),
                    "energy_target": "-",
                    "layer": "all",
                    "band": "-",
                    "rank_rung": "full",
                    "seed": r["seed"],
                    "mse_test": r["metrics"]["test"]["mse_normalized"],
                    "mse_val": r["metrics"]["val"]["mse_normalized"],
                    "retained_energy_global": 1.0,
                    "mse_raw_mean_test": r["metrics"]["test"]["mse_raw_mean"],
                    "delta_vs_fullrank_test": 0.0,
                }
            )
        elif r["regime"] == "trained-under-rank-constraint":
            ranks = r["config"]["model"]["ranks"]
            rows.append(
                {
                    "regime": "trained-under-rank-constraint",
                    "arch": arch_label(r),
                    "energy_target": "-",
                    "layer": "all",
                    "band": "n/a (capacity was never there)",
                    "rank_rung": "/".join(str(x) for x in ranks),
                    "seed": r["seed"],
                    "mse_test": r["metrics"]["test"]["mse_normalized"],
                    "mse_val": r["metrics"]["val"]["mse_normalized"],
                    "retained_energy_global": 1.0,
                    "mse_raw_mean_test": r["metrics"]["test"]["mse_raw_mean"],
                    "delta_vs_fullrank_test": None,
                    "n_params": r["n_params"],
                }
            )
    return rows


def to_markdown(agg: list[dict], cols: list[str]) -> str:
    head = [c.replace("_", " ") for c in cols]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in agg:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="+", default=["results/runs/*/run.json"])
    p.add_argument("--sweeps", nargs="+", default=["results/runs/*/truncation_*.json"])
    p.add_argument("--out", default="results/summary.json")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--layer", default=None, help="filter to one layer key (all/0/1/2)")
    p.add_argument("--regime", default=None)
    args = p.parse_args()

    records, rows, swept_run_ids = collect(_load_patterns(args.runs), _load_patterns(args.sweeps))
    rows += add_baseline_rows(records, swept_run_ids)
    agg = aggregate(rows)
    if args.layer:
        agg = [r for r in agg if str(r["layer"]) == args.layer]
    if args.regime:
        agg = [r for r in agg if r["regime"] == args.regime]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_runs": len(records), "n_rows": len(agg), "rows": agg}, indent=2))
    print(f"{len(rows)} sweep rows from {len(records)} run records -> {out}")

    cols = [
        "arch",
        "regime",
        "layer",
        "band",
        "rank_rung",
        "energy_target",
        "n_runs",
        "energy_mean",
        "mse_test_mean",
        "mse_test_min",
        "mse_test_max",
        "mse_raw_mean_test_mean",
    ]
    agg = [{**r, "n_runs": r.pop("n_seeds")} for r in agg]
    table = to_markdown(agg, [c for c in cols if any(c in r for r in agg)])
    print("\n" + table)
    if args.markdown:
        md = out.with_suffix(".md")
        md.write_text(table + "\n")
        print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
