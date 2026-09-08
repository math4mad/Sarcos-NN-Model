"""The MLP whose layers we take apart.

A plain feed-forward ReLU regressor, 21 -> hidden... -> 7. The SVD experiment is
about **weight matrices**, so the architecture keeps them plainly enumerable:
``model.layers[i].weight`` is the ``i``-th ``W`` (shape ``[out, in]``) and
``model.layers[i].bias`` is the matching bias. Biases are never truncated.

Two regimes, deliberately kept apart (AGENTS.md rule 3):

* ``mode="full"``        — the unconstrained net; truncation happens *after* training
  (see :mod:`sarcos_svd.lowrank`).
* ``mode="constrained"`` — each layer is *trained* as an explicit rank-``r``
  product ``W = U Vᵀ`` with ``U:[out,r]``, ``V:[r,in]``. A net that never had the
  capacity is not comparable to one that had it and lost it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_dim: int
    out_dim: int
    hidden: tuple[int, ...] = (64, 64)
    mode: str = "full"  # "full" | "constrained"
    ranks: tuple[int, ...] | None = None  # per-layer rank cap, constrained mode only

    def __post_init__(self) -> None:
        if self.mode not in ("full", "constrained"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "constrained":
            if self.ranks is None:
                raise ValueError("constrained mode needs explicit per-layer ranks")
            if len(self.ranks) != len(self.layer_shapes):
                raise ValueError(f"ranks has {len(self.ranks)} entries, expected {len(self.layer_shapes)}")
            for r, shape in zip(self.ranks, self.layer_shapes):
                if not 1 <= r <= min(shape):
                    raise ValueError(f"rank {r} out of range for layer {shape}")

    @property
    def dims(self) -> list[int]:
        return [self.in_dim, *self.hidden, self.out_dim]

    @property
    def layer_shapes(self) -> list[tuple[int, int]]:
        """(out, in) for each weight matrix, in forward order."""
        return list(zip(self.dims[1:], self.dims[:-1]))


class LowRankLinear(nn.Module):
    """Linear layer whose weight is the product of two thin matrices."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.u = nn.Parameter(torch.empty(out_features, rank))
        self.v = nn.Parameter(torch.empty(rank, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.u, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.v, a=np.sqrt(5))
        bound = 1 / np.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        return self.u @ self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.v), self.u, self.bias)


def _make_layer(cfg: ModelConfig, index: int) -> nn.Module:
    out, inn = cfg.layer_shapes[index]
    if cfg.mode == "constrained":
        return LowRankLinear(inn, out, int(cfg.ranks[index]))
    return nn.Linear(inn, out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(_make_layer(cfg, i) for i in range(len(cfg.layer_shapes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    # -- weight-matrix accessors used by lowrank.py / evaluate.py ----------------
    def weight_matrices(self) -> dict[str, torch.Tensor]:
        return {f"layers.{i}.weight": layer.weight for i, layer in enumerate(self.layers)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: ModelConfig, seed: int) -> MLP:
    torch.manual_seed(seed)  # same seed => same init for every regime
    return MLP(cfg)


def state_from_model(model: MLP) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def numpy_state(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in state.items()}


def spectra(state: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Singular values + Frobenius norms of every weight matrix: the object of study."""
    out: dict[str, dict] = {}
    for name, w in state.items():
        if not name.endswith(".weight"):
            continue
        s = torch.linalg.svdvals(w.detach().cpu().double())
        e = (s**2)
        out[name] = {
            "shape": list(w.shape),
            "rank_max": int(min(w.shape)),
            "sigma": [round(float(x), 6) for x in s],
            "fro_norm_sq": float(e.sum()),
            "energy_fraction_cummax": round(float(e[0] / e.sum()), 6),
        }
    return out
