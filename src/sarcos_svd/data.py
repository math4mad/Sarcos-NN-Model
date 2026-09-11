"""Canonical SARCOS loader and split.

THE ONLY PLACE SPLITS ARE DEFINED. Every module that needs train/val/test indices
imports :func:`load_split`; nothing downstream may re-cut the data.

Why the split is ours and not GPML's
------------------------------------
The two files published with *GPML* (``sarcos_inv.mat``, ``sarcos_inv_test.mat``)
are not a usable train/test pair. ``sarcos_inv_test.mat`` is **every 10th row of
``sarcos_inv.mat``**, and those rows are still inside the training file: 4,368 of
4,449 test rows (98.2%) occur verbatim in train. Scoring on it measures
memorisation. See :func:`leak_report`, which prints these facts from the data
rather than asking the reader to trust this docstring.

The rows are also in trajectory order: mean input distance between lag-1 rows is
~3.6 vs ~31.4 for random pairs, so a *row-wise* random split still leaks. We
therefore split by contiguous **blocks** of rows, so no two samples from the same
stretch of trajectory land on opposite sides of a boundary.

Conventions fixed here (all explicit, none silent)
-------------------------------------------------
* pooled rows = the 44,484 rows of ``sarcos_inv.mat`` in file order, plus the 81
  rows of ``sarcos_inv_test.mat`` that are not already in it (appended at the end,
  where they break trajectory contiguity for 81 of 44,565 rows; logged).
* duplicates are keyed on targets/inputs rounded to 6 decimals (the published
  precision).
* features and targets are standardised with **train-split statistics only**.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE_URL = "https://gaussianprocess.org/gpml/data/"
N_INPUTS = 21
N_OUTPUTS = 7
PRECISION = 6  # decimals used when matching rows for de-duplication

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

_FILES = {
    "gpml_train": ("sarcos_inv.mat", "sarcos_inv"),
    "gpml_test": ("sarcos_inv_test.mat", "sarcos_inv_test"),
}


# --------------------------------------------------------------------------- io


def data_dir(path: str | Path | None = None) -> Path:
    """Resolve a data directory; relative paths are anchored at the repo root so a
    recorded ``"data/"`` means the same thing from any working directory."""
    if path is None:
        return DEFAULT_DATA_DIR
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def fetch(dirpath: str | Path | None = None, force: bool = False) -> dict:
    """Download the two GPML .mat files unless already present. Returns file manifest."""
    import urllib.request

    d = data_dir(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for logical, (fname, _) in _FILES.items():
        target = d / fname
        if force or not target.exists():
            url = BASE_URL + fname
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(target)
        manifest[logical] = {
            "file": fname,
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
    return manifest


def _load_matrix(dirpath: Path, logical: str) -> np.ndarray:
    from scipy.io import loadmat

    fname, varname = _FILES[logical]
    raw = loadmat(dirpath / fname)
    arr = np.asarray(raw[varname], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_INPUTS + N_OUTPUTS:
        raise ValueError(f"{fname}: expected (n, {N_INPUTS + N_OUTPUTS}), got {arr.shape}")
    return arr


def _row_keys(a: np.ndarray) -> list[bytes]:
    packed = np.ascontiguousarray(np.round(a, PRECISION)).view(np.uint8)
    packed = packed.reshape(a.shape[0], -1)
    return [row.tobytes() for row in packed]


# ---------------------------------------------------------------------- loading


@dataclass(frozen=True)
class SplitConfig:
    """Everything that determines who trains and who is scored. No defaults matter
    here: callers must pass them, and they are written into every run record."""

    seed: int
    block_size: int
    train_frac: float
    val_frac: float
    test_frac: float

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"fractions must sum to 1, got {total}")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SarcosData:
    """Standardised tensors plus the raw (physical-unit) arrays and the split manifest."""

    x_raw: np.ndarray
    y_raw: np.ndarray
    x: np.ndarray  # standardised inputs
    y: np.ndarray  # standardised targets
    idx: dict[str, np.ndarray]
    split_manifest: dict
    stats: dict = field(default_factory=dict)

    @property
    def n_outputs(self) -> int:
        return self.y.shape[1]


def load_pooled(dirpath: str | Path | None = None) -> np.ndarray:
    """All unique rows: gpml_train in file order, then test-only rows appended."""
    d = data_dir(dirpath)
    if not (d / _FILES["gpml_train"][0]).exists() or not (d / _FILES["gpml_test"][0]).exists():
        raise FileNotFoundError(
            f"SARCOS .mat files missing under {d}. Run: python -m sarcos_svd.data --fetch"
        )
    tr = _load_matrix(d, "gpml_train")
    te = _load_matrix(d, "gpml_test")
    seen = set(_row_keys(tr))
    extra = [row for key, row in zip(_row_keys(te), te) if key not in seen]
    if not extra:
        return tr
    return np.vstack([tr, np.asarray(extra)])


def block_split(rows: np.ndarray, cfg: SplitConfig) -> tuple[dict[str, np.ndarray], dict]:
    """Assign contiguous row-blocks to splits under a seeded permutation of blocks."""
    n = rows.shape[0]
    n_blocks = int(np.ceil(n / cfg.block_size))
    perm = np.random.default_rng(cfg.seed).permutation(n_blocks)
    n_test = int(round(n_blocks * cfg.test_frac))
    n_val = int(round(n_blocks * cfg.val_frac))
    labels = np.empty(n_blocks, dtype=object)
    labels[perm[:n_test]] = "test"
    labels[perm[n_test : n_test + n_val]] = "val"
    labels[perm[n_test + n_val :]] = "train"

    idx: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        blocks = np.flatnonzero(labels == name)
        rows_sel = np.concatenate(
            [np.arange(b * cfg.block_size, min((b + 1) * cfg.block_size, n)) for b in blocks]
        )
        idx[name] = np.sort(rows_sel)  # ascending, so ordering inside a split is stable
    manifest = {
        "kind": "contiguous-block",
        **cfg.as_dict(),
        "n_rows_pooled": n,
        "n_blocks": n_blocks,
        "block_size": cfg.block_size,
        "sizes": {k: int(len(v)) for k, v in idx.items()},
        "sha256_pooled_rows": hashlib.sha256(np.round(rows, PRECISION).tobytes()).hexdigest(),
    }
    return idx, manifest


def load_split(cfg: SplitConfig, dirpath: str | Path | None = None) -> SarcosData:
    """Canonical entry point: pooled rows -> block split -> train-standardised tensors."""
    rows = load_pooled(dirpath)
    x_raw, y_raw = rows[:, :N_INPUTS], rows[:, N_INPUTS:]
    idx, manifest = block_split(rows, cfg)

    tr = idx["train"]
    x_mu, x_sd = x_raw[tr].mean(0), x_raw[tr].std(0)
    y_mu, y_sd = y_raw[tr].mean(0), y_raw[tr].std(0)
    x_sd = np.where(x_sd == 0, 1.0, x_sd)
    y_sd = np.where(y_sd == 0, 1.0, y_sd)

    x = (x_raw - x_mu) / x_sd
    y = (y_raw - y_mu) / y_sd
    stats = {
        "x_mean": x_mu.tolist(),
        "x_std": x_sd.tolist(),
        "y_mean": y_mu.tolist(),
        "y_std": y_sd.tolist(),
        "y_var": (y_sd**2).tolist(),
        "normalisation": "z-score, fitted on train split only",
    }
    return SarcosData(x_raw=x_raw, y_raw=y_raw, x=x, y=y, idx=idx, split_manifest=manifest, stats=stats)


# ------------------------------------------------------------------- diagnostics


def leak_report(dirpath: str | Path | None = None) -> dict:
    """Quantify why the published split is unusable, straight from the data."""
    d = data_dir(dirpath)
    tr = _load_matrix(d, "gpml_train")
    te = _load_matrix(d, "gpml_test")
    first_pos: dict[bytes, int] = {}
    for i, k in enumerate(_row_keys(tr)):
        first_pos.setdefault(k, i)
    keys_te = _row_keys(te)
    matched = np.array([first_pos[k] for k in keys_te if k in first_pos])
    diffs = np.diff(np.sort(matched))
    rows = {
        "gpml_train_rows": int(tr.shape[0]),
        "gpml_test_rows": int(te.shape[0]),
        "test_rows_found_verbatim_in_train": int(len(matched)),
        "test_fraction_leaked": round(float(len(matched) / len(te)), 4),
        "pooled_unique_rows": int(len(set(keys_te) | set(first_pos))),
        "matched_test_positions_are_multiples_of_10": bool(np.all(matched % 10 == 0)),
        "modal_gap_between_matched_positions": int(np.bincount(diffs).argmax()),
    }
    lag = {}
    for k in (1, 2, 5, 10, 50, 200, 1000):
        a = tr[:, :N_INPUTS]
        lag[f"lag_{k}"] = round(float(np.mean(np.linalg.norm(a[k:] - a[:-k], axis=1))), 3)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(tr), 3000), rng.integers(0, len(tr), 3000)
    rows["random_pair_distance"] = round(float(np.mean(np.linalg.norm(tr[i, :N_INPUTS] - tr[j, :N_INPUTS], axis=1))), 3)
    report = {"files": {k: sha256(d / v[0]) for k, v in _FILES.items()}, **rows, "input_distance_by_lag": lag}
    return report


def check_split_hygiene(data: SarcosData) -> dict:
    """How much near-duplication survives our own block split (must be reported, not assumed)."""
    tr, te = data.idx["train"], data.idx["test"]
    keys_tr = set(_row_keys(data.x_raw[tr]))
    dup_in = sum(1 for k in _row_keys(data.x_raw[te]) if k in keys_tr)
    d = np.linalg.norm(data.x_raw[tr].mean(0, keepdims=True) - data.x_raw[te], axis=1)
    return {
        "test_rows_with_identical_input_in_train": int(dup_in),
        "test_rows_with_identical_input_fraction": round(float(dup_in / len(te)), 5),
        "min_input_distance_test_to_train": round(float(d.min()), 6),
    }


# ---------------------------------------------------------------------------
# A shifted *downstream* task, built only from rows we already have.
#
# LoRA's evidence comes from fine-tuning a converged model on data it was not
# trained on. Our canonical split has no such gap: the val blocks are drawn from
# the same region of trajectory space as the train blocks, so "fine-tuning" on them
# can only overwrite what the base already knows (measured: frozen 0.0161 beat full
# fine-tuning 0.0178). To get a task where adaptation can actually help, we rank the
# held-out rows by how far each one is, in input space, from the nearest training
# row - i.e. covariate shift towards the edge of what the net has seen - and use the
# far half of the val blocks as the downstream fit set and the far half of the test
# blocks as its held-out evaluation. The near half of the test blocks stays the
# in-distribution probe, so forgetting is measurable on the same run.
# ---------------------------------------------------------------------------


def _nn_distance(query: np.ndarray, reference: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Distance from each query row to its nearest reference row (exact, Gram trick).

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab' so the inner loop is one BLAS matmul per
    chunk instead of a broadcast difference tensor; the naive form needs ~1 GB for our
    sizes and is ~50x slower. Clipped at 0 because the algebraic identity is not exact
    in floating point.
    """
    q = np.ascontiguousarray(query, dtype=np.float64)
    r = np.ascontiguousarray(reference, dtype=np.float64)
    r2 = (r * r).sum(1)
    out = np.empty(len(q))
    for start in range(0, len(q), chunk):
        block = q[start : start + chunk]
        d2 = (block * block).sum(1)[:, None] + r2[None, :] - 2.0 * (block @ r.T)
        out[start : start + len(block)] = np.sqrt(np.maximum(d2, 0.0)).min(axis=1)
    return out


@dataclass(frozen=True)
class ShiftConfig:
    """The downstream task definition. Everything that changes it changes the result."""

    seed: int
    block_size: int
    train_frac: float
    val_frac: float
    test_frac: float
    far_frac: float = 0.5
    fit_frac: float = 0.75
    nn_chunk: int = 1024

    def split(self) -> SplitConfig:
        return SplitConfig(self.seed, self.block_size, self.train_frac, self.val_frac, self.test_frac)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def shift_task(cfg: ShiftConfig, dirpath: str | Path | None = None) -> dict:
    """Indices + provenance for the covariate-shifted downstream task.

    Block-level, deliberately. The first version of this picked the farthest *rows*
    of the val blocks as the fit set and the farthest rows of the test blocks as the
    evaluation - but "far from train" is not one region: those are two different
    places in input space, so an arm could only ever be damaged by the fit set, never
    helped. Here the downstream **domain** is a set of whole blocks (the blocks whose
    rows sit farthest from the training data), and fit/select/eval are different
    blocks *of the same domain*. Then improvement can actually mean something.
    """
    data = load_split(cfg.split(), dirpath)
    x, bs = data.x_raw, cfg.block_size
    train = data.idx["train"]

    def blocks_of(idx: np.ndarray) -> np.ndarray:
        return np.unique(idx // bs)

    def block_score(blocks: np.ndarray, idx: np.ndarray) -> dict[int, float]:
        d = _nn_distance(x[idx], x[train])
        owner = idx // bs
        return {int(b): float(d[owner == b].mean()) for b in blocks}

    val_blocks, test_blocks = blocks_of(data.idx["val"]), blocks_of(data.idx["test"])
    vs = block_score(val_blocks, data.idx["val"])
    ts = block_score(test_blocks, data.idx["test"])

    def split_blocks(blocks, scores, keep):
        ordered = sorted(blocks, key=lambda b: -scores[int(b)])
        n = int(round(len(ordered) * keep))
        return ordered[:n], ordered[n:]

    far_val, near_val = split_blocks(val_blocks, vs, cfg.far_frac)
    far_test, near_test = split_blocks(test_blocks, ts, cfg.far_frac)
    perm = np.random.default_rng(cfg.seed + 2).permutation(len(far_val))
    cut = int(round(len(far_val) * cfg.fit_frac))
    fit_blocks = [int(far_val[i]) for i in perm[:cut]]
    select_blocks = [int(far_val[i]) for i in perm[cut:]]
    far_val = [int(b) for b in far_val]
    far_test = [int(b) for b in far_test]
    near_test = [int(b) for b in near_test]

    def rows_of(blocks, pool):
        if not blocks:
            return np.array([], dtype=int)
        mask = np.isin(pool // bs, blocks)
        return np.sort(pool[mask])

    fit = rows_of(fit_blocks, data.idx["val"])
    select = rows_of(select_blocks, data.idx["val"])
    down = rows_of(far_test, data.idx["test"])
    indist = rows_of(near_test, data.idx["test"])
    meta = {
        "kind": "covariate-shift domain = whole blocks farthest from train, fit/eval split by block",
        **cfg.as_dict(),
        "val_block_distance_range": [round(min(vs.values()), 3), round(max(vs.values()), 3)],
        "test_block_distance_range": [round(min(ts.values()), 3), round(max(ts.values()), 3)],
        "far_test_block_distance_min": round(min(ts[b] for b in far_test), 4),
        "near_test_block_distance_max": round(max(ts[b] for b in near_test), 4) if near_test else None,
        "sizes": {"fit": int(len(fit)), "select": int(len(select)),
                  "eval_downstream": int(len(down)), "eval_indistribution": int(len(indist))},
        "blocks": {"fit": fit_blocks, "select": select_blocks,
                   "eval_downstream": far_test, "eval_indistribution": near_test},
    }
    return {"fit": fit, "select": select, "eval_downstream": down, "eval_indistribution": indist,
            "meta": meta}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fetch", action="store_true", help="download the GPML .mat files into data/")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--report", action="store_true", help="print the leakage + split diagnostic JSON")
    p.add_argument("--shift-task", action="store_true", help="print the downstream-task provenance JSON")
    p.add_argument("--seed", type=int, default=None, help="with --report: also print this split's manifest")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--fractions", type=float, nargs=3, default=None, metavar=("TRAIN", "VAL", "TEST"))
    args = p.parse_args()

    if args.fetch:
        print(json.dumps(fetch(args.data_dir), indent=2))
    if args.report:
        print(json.dumps(leak_report(args.data_dir), indent=2))
        if args.seed is not None and args.block_size is not None and args.fractions is not None:
            cfg = SplitConfig(
                seed=args.seed,
                block_size=args.block_size,
                train_frac=args.fractions[0],
                val_frac=args.fractions[1],
                test_frac=args.fractions[2],
            )
            data = load_split(cfg, args.data_dir)
            print(json.dumps({"split": data.split_manifest, "hygiene": check_split_hygiene(data)}, indent=2))
    if args.shift_task:
        if not (args.seed and args.block_size and args.fractions):
            raise SystemExit("--shift-task needs --seed, --block-size and --fractions")
        scfg = ShiftConfig(
            seed=args.seed, block_size=args.block_size, train_frac=args.fractions[0],
            val_frac=args.fractions[1], test_frac=args.fractions[2],
        )
        print(json.dumps(shift_task(scfg, args.data_dir)["meta"], indent=2))
    if not (args.fetch or args.report or args.shift_task):
        p.print_help()


if __name__ == "__main__":
    main()
