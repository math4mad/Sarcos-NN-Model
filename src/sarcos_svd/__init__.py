"""SARCOS x SVD low-rank study.

Modules
-------
``data``       canonical loader + the only split definition in the project
``model``      the MLP, and its rank-constrained twin
``train``      seeded, deterministic training runner; writes run records
``lowrank``    SVD truncation into leading / middle / trailing bands
``evaluate``   MSE + retained-energy reporting and the post-hoc sweep
"""

__all__ = ["data", "model", "train", "lowrank", "evaluate"]
__version__ = "0.1.0"
