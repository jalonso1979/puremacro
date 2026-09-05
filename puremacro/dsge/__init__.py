"""DSGE primitives for puremacro.

Includes:
- Klein (2000) QZ solver for linear rational-expectations models.
- ``build``: write equilibrium conditions as a Python function and get a
  solved first-order approximation back, with the Jacobians taken by
  complex-step differentiation (no hand-derived matrices, no Dynare).
- Sims (2002) gensys solver (equivalent, model-agnostic input form).
- Bayesian estimation engine (random-walk Metropolis-Hastings) +
  model-agnostic priors framework.
- Smets-Wouters (2007) reference model + bundled US dataset.
- Fertility DSGE (Alonso-Ortiz adjustment-costs variant) — solver only;
  Bayesian estimation queued for 0.55.0.

For likelihood-based estimation, pair the state-space form returned by
``make_state_space`` (model-specific) with ``puremacro.dsge.estimate_dsge``.
"""
from .klein import BlanchardKahnError, KleinSolution, klein_solve
from .build import LinearModel, ModelError, SteadyStateError, build
from ._results import (
    DSGEPosteriorResult, SW07PosteriorResult,
    FertilitySolution,
    DynareDR, TheoreticalMomentsResult,
)
from .estimate import estimate_dsge
from .sw07_estimate import estimate_sw07
from .fertility_adj_costs import solve_bgp, solve_fertility
from .pruning import (
    PrunedDSGESolution,
    PrunedSimulationResult,
    canonical_growth_2nd_order,
)
from .dynare import build_dynare, parse_mod, load_mod, solve_dynare_2nd_order
from . import priors, fertility_adj_costs

__all__ = [
    "klein_solve", "KleinSolution", "BlanchardKahnError",
    "build", "LinearModel", "ModelError", "SteadyStateError",
    "DSGEPosteriorResult", "SW07PosteriorResult", "FertilitySolution",
    "DynareDR", "TheoreticalMomentsResult",
    "build_dynare", "parse_mod", "load_mod", "solve_dynare_2nd_order",
    "estimate_dsge", "estimate_sw07",
    "solve_bgp", "solve_fertility",
    "PrunedDSGESolution", "PrunedSimulationResult", "canonical_growth_2nd_order",
    "priors", "fertility_adj_costs",
]
from . import smets_wouters  # re-export for back-compat with 0.50.0 callers
