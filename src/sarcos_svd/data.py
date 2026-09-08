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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fetch", action="store_true", help="download the GPML .mat files into data/")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--report", action="store_true", help="print the leakage + split diagnostic JSON")
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
    if not (args.fetch or args.report):
        p.print_help()


if __name__ == "__main__":
    main()
