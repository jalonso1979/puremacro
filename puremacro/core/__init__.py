"""Core unified abstractions, protocols, and adapters for PureMacro v6.0.0."""
from __future__ import annotations

from .continuation import (
    ContinuationAttempt,
    EquilibriumBranchResult,
    EquilibriumFold,
    continue_equilibrium,
)

from .equilibrium import (
    EquilibriumLinearization,
    EquilibriumSensitivityResult,
    EquilibriumSystem,
    linearize_equilibrium,
)

from .protocol import (
    DFMDifferentiableAdapter,
    DSGEDifferentiableAdapter,
    DifferentiableModel,
    HANKDAGDifferentiableAdapter,
    SpatialDifferentiableAdapter,
    StateSpaceProtocol,
    SVARDifferentiableAdapter,
    TradeDifferentiableAdapter,
    wrap_differentiable_model,
)

__all__ = [
    "ContinuationAttempt",
    "EquilibriumBranchResult",
    "EquilibriumFold",
    "continue_equilibrium",
    "EquilibriumSystem",
    "EquilibriumLinearization",
    "EquilibriumSensitivityResult",
    "linearize_equilibrium",
    "DifferentiableModel",
    "StateSpaceProtocol",
    "DSGEDifferentiableAdapter",
    "HANKDAGDifferentiableAdapter",
    "DFMDifferentiableAdapter",
    "SVARDifferentiableAdapter",
    "TradeDifferentiableAdapter",
    "SpatialDifferentiableAdapter",
    "wrap_differentiable_model",
]
