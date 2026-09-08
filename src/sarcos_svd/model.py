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


class LoRAIncrementLinear(nn.Module):
    """LoRA's parameterisation: ``W_eff = W_frozen + scale · A B``.

    ``A:[out,r]`` random, ``B:[r,in]`` zero, so the increment starts at exactly zero
    and the frozen base is reproduced at step 0 — that is what makes this a *third*
    regime rather than a re-run of the replacement one: the capacity that was there
    is never taken away, it is only ever edited through a rank-``r`` channel.

    ``weight`` exposes the effective matrix so spectra/truncation code works on it
    unchanged; gradients flow only to A and B (and not to the frozen base or bias).
    """

    def __init__(self, base: torch.Tensor, rank: int, scale: float = 1.0) -> None:
        super().__init__()
        out_features, in_features = int(base.shape[0]), int(base.shape[1])
        if not 1 <= rank <= min(base.shape):
            raise ValueError(f"rank {rank} out of range for base {tuple(base.shape)}")
        self.in_features, self.out_features, self.rank, self.scale = in_features, out_features, rank, float(scale)
        self.register_buffer("weight_base", base.detach().clone())
        self.register_buffer("bias_base", torch.zeros(out_features))
        self.a = nn.Parameter(torch.empty(out_features, rank))
        self.b = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.a, a=np.sqrt(5))
        nn.init.zeros_(self.b)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_base + self.scale * (self.a @ self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias_base)


@dataclass(frozen=True)
class ModelConfig:
    in_dim: int
    out_dim: int
    hidden: tuple[int, ...] = (64, 64)
    mode: str = "full"  # "full" | "constrained" | "lora"
    ranks: tuple[int, ...] | None = None  # per-layer rank cap, constrained/lora modes only
    lora_scale: float = 1.0  # alpha/r style scaling of the increment, lora mode only

    def __post_init__(self) -> None:
        if self.mode not in ("full", "constrained", "lora"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode in ("constrained", "lora"):
            if self.ranks is None:
                raise ValueError(f"{self.mode} mode needs explicit per-layer ranks")
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


def _make_layer(cfg: ModelConfig, index: int, base_state: dict[str, torch.Tensor] | None = None) -> nn.Module:
    out, inn = cfg.layer_shapes[index]
    name = f"layers.{index}.weight"
    if cfg.mode == "lora":
        if base_state is None or name not in base_state:
            raise ValueError("lora mode needs the frozen base weights (base_state)")
        return LoRAIncrementLinear(base_state[name], int(cfg.ranks[index]), cfg.lora_scale)
    if cfg.mode == "constrained":
        return LowRankLinear(inn, out, int(cfg.ranks[index]))
    return nn.Linear(inn, out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, base_state: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(_make_layer(cfg, i, base_state) for i in range(len(cfg.layer_shapes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    # -- weight-matrix accessors used by lowrank.py / evaluate.py ----------------
    def weight_matrices(self) -> dict[str, torch.Tensor]:
        return {f"layers.{i}.weight": layer.weight for i, layer in enumerate(self.layers)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: ModelConfig, seed: int, base_state: dict[str, torch.Tensor] | None = None) -> MLP:
    torch.manual_seed(seed)  # same seed => same init for every regime
    return MLP(cfg, base_state)


def trainable_params(model: MLP) -> list[torch.nn.Parameter]:
    """Parameters that actually receive gradients (LoRA freezes the base and bias)."""
    return [p for p in model.parameters() if p.requires_grad]


def state_from_model(model: MLP) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def numpy_state(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Detach to numpy. The `.copy()` is load-bearing: `tensor.numpy()` shares
    storage, so without it a captured 'initial' state silently becomes the *final*
    state as soon as training writes to those parameters."""
    return {k: np.ascontiguousarray(v.detach().cpu().numpy()).copy() for k, v in state.items()}


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
