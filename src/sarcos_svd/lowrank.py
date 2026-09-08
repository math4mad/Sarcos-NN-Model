"""Post-hoc SVD truncation of a *trained* net: leading / middle / trailing bands.

Given ``W = U Σ Vᵀ`` (singular values sorted descending, as returned by SVD), a band
is just a set of indices into Σ:

===========  ==========================================================
band         retained singular directions (0-based, Σ descending)
===========  ==========================================================
leading      ``0 .. r-1``                        (dominant)
middle       ``(n-r)//2 .. (n-r)//2 + r-1``      (centred bulk)
trailing     ``n-r .. n-1``                      (minor tail)
===========  ==========================================================

``r`` is a per-layer **rank cap**; a layer with ``min(shape) < r`` keeps everything
(its band is full rank by construction, energy 1.0). That is recorded rather than
hidden, because it makes the tail of a rank sweep flat for the small last layer.

The three bands hold different *energy* at equal ``r`` — that is the experiment —
so retained energy is returned with every truncation and must be reported next to
the MSE (AGENTS.md rule 4).
"""

from __future__ import annotations

import numpy as np
import torch

BANDS = ("leading", "middle", "trailing")


def band_indices(n: int, rank: int, band: str) -> np.ndarray:
    """Indices of the ``rank`` singular directions of ``n`` that belong to ``band``."""
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    r = int(max(1, min(rank, n)))
    if band == "leading":
        idx = np.arange(0, r)
    elif band == "trailing":
        idx = np.arange(n - r, n)
    else:  # middle: centred window of width r
        start = (n - r) // 2
        idx = np.arange(start, start + r)
    return idx


def truncate_matrix(w: np.ndarray, rank: int, band: str) -> tuple[np.ndarray, dict]:
    """Return ``(W_r, report)`` for a single weight matrix."""
    w = np.asarray(w, dtype=np.float64)
    n = min(w.shape)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    idx = band_indices(n, rank, band)
    w_r = (u[:, idx] * s[idx]) @ vt[idx]
    energy_full = float((s**2).sum())
    energy_kept = float((s[idx] ** 2).sum())
    report = {
        "shape": list(w.shape),
        "rank_max": int(n),
        "rank_requested": int(rank),
        "rank_effective": int(len(idx)),
        "band": band,
        "sigma_indices": [int(i) for i in idx],
        "fro_norm_sq_before": energy_full,
        "fro_norm_sq_after": energy_kept,
        "retained_energy": round(energy_kept / energy_full, 6) if energy_full > 0 else 1.0,
    }
    return w_r, report


def truncate_state(
    state: dict[str, np.ndarray], ranks: dict[str, int], band: str
) -> tuple[dict[str, np.ndarray], dict]:
    """Truncate every weight matrix of a trained (unconstrained) state dict in place.

    ``ranks`` maps a parameter name (``"layers.0.weight"``) to that layer's rank cap.
    Non-weight tensors (biases) pass through untouched.
    """
    names = [k for k in state if k.endswith(".weight")]
    if not names:
        raise ValueError(
            "no '*.weight' entries: post-hoc truncation applies only to a mode=full run "
            "(a constrained run stores layers.N.u/.v and is a different experiment)"
        )
    missing = [k for k in names if k not in ranks]
    if missing:
        raise ValueError(f"no rank given for {missing}")

    new_state = {k: np.array(v, dtype=np.float32) for k, v in state.items()}
    per_layer: dict[str, dict] = {}
    num = den = 0.0
    for name in names:
        w_r, rep = truncate_matrix(state[name], ranks[name], band)
        new_state[name] = w_r.astype(np.float32)
        per_layer[name] = rep
        num += rep["fro_norm_sq_after"]
        den += rep["fro_norm_sq_before"]
    summary = {
        "band": band,
        "ranks": {k: int(v) for k, v in ranks.items()},
        "per_layer": per_layer,
        "retained_energy_global": round(num / den, 6) if den else 1.0,
        "retained_energy_min_layer": round(min(p["retained_energy"] for p in per_layer.values()), 6),
        "effective_ranks": {k: p["rank_effective"] for k, p in per_layer.items()},
    }
    return new_state, summary


def state_to_torch(state: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in state.items()}


# --------------------------------------------------------------------- ΔW: LoRA's object
#
# LoRA does not constrain W, it constrains the *increment* ΔW = W_trained - W_init.
# Everything below therefore measures energy against ||ΔW||, not ||W||: a rank-r
# slice of a small increment is a different claim from a rank-r slice of the matrix.


def delta_spectrum(w_trained: np.ndarray, w_init: np.ndarray, top_k: int = 8) -> dict:
    """Is the learned deviation low-rank at all? The premise LoRA borrows, tested."""
    w = np.asarray(w_trained, dtype=np.float64) - np.asarray(w_init, dtype=np.float64)
    s = np.linalg.svd(w, compute_uv=False)
    e = s**2
    total = float(e.sum())
    return {
        "shape": list(w.shape),
        "delta_fro_norm_sq": total,
        "delta_over_base_fro": round(float(np.sqrt(total) / np.linalg.norm(w_init)), 6),
        "sigma": [round(float(x), 6) for x in s[: max(top_k, 4)]],
        "energy_top1": round(float(e[0] / total), 6) if total else 1.0,
        "energy_topk": {k: round(float(e[:k].sum() / total), 6) for k in range(1, top_k + 1)} if total else {},
        "rank_at_99pct_energy": int(np.searchsorted(np.cumsum(e) / total, 0.99) + 1) if total else 0,
    }


def truncate_delta(
    w_trained: np.ndarray, w_init: np.ndarray, rank: int, band: str
) -> tuple[np.ndarray, dict]:
    """``W_init + (ΔW restricted to the band)`` — LoRA's operation, applied post-hoc."""
    w0 = np.asarray(w_init, dtype=np.float64)
    d = np.asarray(w_trained, dtype=np.float64) - w0
    n = min(d.shape)
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    idx = band_indices(n, rank, band)
    d_r = (u[:, idx] * s[idx]) @ vt[idx]
    total = float((s**2).sum())
    kept = float((s[idx] ** 2).sum())
    report = {
        "shape": list(d.shape),
        "band": band,
        "rank_requested": int(rank),
        "rank_effective": int(len(idx)),
        "delta_fro_norm_sq": total,
        "delta_retained_energy": round(kept / total, 6) if total else 1.0,
        "delta_over_base_fro": round(float(np.sqrt(total) / np.linalg.norm(w0)), 6),
        "total_weight_retained_energy": round(
            float(np.linalg.norm(w0 + d_r) ** 2 / np.linalg.norm(w0 + d) ** 2), 6
        ),
    }
    return w0 + d_r, report


def truncate_state_delta(
    trained: dict[str, np.ndarray], init: dict[str, np.ndarray], ranks: dict[str, int], band: str
) -> tuple[dict[str, np.ndarray], dict]:
    """Band-truncate every layer's increment and add it back to the initial weights."""
    names = [k for k in trained if k.endswith(".weight")]
    missing = [k for k in names if k not in init]
    if missing:
        raise ValueError(f"no initial weights for {missing} (need weights_init.npz)")
    new_state = {k: np.array(v, dtype=np.float32) for k, v in trained.items()}
    per_layer: dict[str, dict] = {}
    num = den = 0.0
    for name in names:
        w_new, rep = truncate_delta(trained[name], init[name], ranks[name], band)
        new_state[name] = w_new.astype(np.float32)
        per_layer[name] = rep
        num += rep["delta_fro_norm_sq"] * rep["delta_retained_energy"]
        den += rep["delta_fro_norm_sq"]
    return new_state, {
        "regime": "post-hoc-delta-truncation",
        "band": band,
        "ranks": {k: int(v) for k, v in ranks.items()},
        "per_layer": per_layer,
        "delta_retained_energy_global": round(num / den, 6) if den else 1.0,
        "effective_ranks": {k: p["rank_effective"] for k, p in per_layer.items()},
    }


def rank_ladder(layer_shapes: list[tuple[int, int]], rungs: list[int]) -> list[dict[str, int]]:
    """Per-layer rank caps for each rung of a sweep, e.g. rungs=[1,2,4,8,16,32,64].

    Keys follow a ``mode=full`` state dict: ``layers.{i}.weight``.
    """
    return [{f"layers.{i}.weight": int(r) for i in range(len(layer_shapes))} for r in rungs]


def hit_energy(
    state: dict[str, np.ndarray], target: float, band: str, rungs: list[int], layer: str | None = None
) -> tuple[int, float]:
    """Smallest uniform rank rung whose retained energy reaches ``target``.

    ``layer=None`` measures energy over the whole net (every layer truncated together).
    Passing a parameter name searches that layer alone and measures *its* energy — the
    honest way to match energy for a single-variable ablation.
    """
    names = [k for k in state if k.endswith(".weight")]
    if layer is not None and layer not in names:
        raise ValueError(f"unknown layer {layer!r}; have {names}")
    ranks_full = {k: int(min(state[k].shape)) for k in names}
    best = (int(max(rungs)), 0.0)
    best_e = -1.0
    for r in sorted(int(x) for x in rungs):
        caps = {
            k: (min(r, ranks_full[k]) if (layer is None or k == layer) else ranks_full[k]) for k in names
        }
        _, summary = truncate_state(state, caps, band)
        e = summary["retained_energy_global"] if layer is None else summary["per_layer"][layer]["retained_energy"]
        if e >= target - 1e-12:
            return int(r), float(e)
        if e > best_e:
            best, best_e = (int(r), e), e
    return best
