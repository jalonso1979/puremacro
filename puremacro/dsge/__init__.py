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
    DynareDR, Dynare2ndDR, TheoreticalMomentsResult,
    StochSimulResult,
)
from .estimate import estimate_dsge
from .bayesian import BayesianEstimationResult, estimate_dsge_bayesian
from .sw07_estimate import estimate_sw07
from .fertility_adj_costs import solve_bgp, solve_fertility
from .pruning import (
    PrunedDSGESolution,
    PrunedSimulationResult,
    canonical_growth_2nd_order,
)
from .dynare import build_dynare, parse_mod, load_mod, load_dynare_mod, solve_dynare_2nd_order
from .perfect_foresight import PerfectForesightResult, solve_perfect_foresight
from .occbin import OccBinConstraint, OccBinResult, solve_occbin
from .gertler_karadi import (
    GK2011_PARAMS,
    GertlerKaradiResult,
    build_gertler_karadi_model,
    solve_gertler_karadi,
    solve_steady_state,
)
from .priors import (
    Prior,
    BetaPrior,
    InvGammaPrior,
    NormalPrior,
    GammaPrior,
    UniformPrior,
)
from .decomposition import (
    FEVDResult,
    ShockDecompResult,
    compute_fevd,
    compute_shock_decomposition,
)
from . import priors, fertility_adj_costs

__all__ = [
    "klein_solve", "KleinSolution", "BlanchardKahnError",
    "build", "LinearModel", "ModelError", "SteadyStateError",
    "DSGEPosteriorResult", "SW07PosteriorResult", "BayesianEstimationResult", "FertilitySolution",
    "DynareDR", "Dynare2ndDR", "TheoreticalMomentsResult", "StochSimulResult",
    "build_dynare", "parse_mod", "load_mod", "load_dynare_mod", "solve_dynare_2nd_order",
    "PerfectForesightResult", "solve_perfect_foresight",
    "OccBinConstraint", "OccBinResult", "solve_occbin",
    "solve_gertler_karadi", "GertlerKaradiResult", "GK2011_PARAMS", "solve_steady_state", "build_gertler_karadi_model",
    "estimate_dsge", "estimate_dsge_bayesian", "estimate_sw07",
    "solve_bgp", "solve_fertility",
    "PrunedDSGESolution", "PrunedSimulationResult", "canonical_growth_2nd_order",
    "FEVDResult", "ShockDecompResult", "compute_fevd", "compute_shock_decomposition",
    "Prior", "BetaPrior", "InvGammaPrior", "NormalPrior", "GammaPrior", "UniformPrior",
    "priors", "fertility_adj_costs",
]
from . import smets_wouters  # re-export for back-compat with 0.50.0 callers
from . import gertler_karadi

