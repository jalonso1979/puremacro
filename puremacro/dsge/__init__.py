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
from .steady import StructuralSingularityError, steady
from ._moments import one_sided_hp_filter
from ._results import (
    DSGEPosteriorResult, SW07PosteriorResult, NUTSResult, HANKResult,
    FertilitySolution,
    DynareDR, Dynare2ndDR, TheoreticalMomentsResult,
    StochSimulResult,
    SmootherResult, DSGEForecastResult, ModeCheckResult,
    DiagnosticFinding, EigenvalueTable, ModelDiagnosticsResult,
    IdentificationResult,
    OSRResult, PolicyResult,
    ExtendedPathResult,
    ConditionalForecastResult,
    ShockDecompositionResult,
    BayesianIRFResult,
    PriorPredictiveResult,
    ModelParityResult,
    ParityDashboardResult,
)
from .hank import HANKModel, load_hank_mod, solve_hank_bridge
from .parity import verify_dynare_parity, compare_model_to_dynare, run_parity_suite
from .load_dynare import load_dynare_dr, load_dynare_moments, load_irfs, load_fevd
from .conditional import conditional_forecast
from .shock_groups import shock_groups_decomposition
from .bayesian import bayesian_irf, prior_predictive
from .diagnostics import check, resid, model_diagnostics
from .identification import identification
from .policy import osr, discretionary_policy, lq_commitment
from .estimate import estimate_dsge
from .nuts import nuts_sample
from ._gradients import ScoreDiagnosticsResult
from .bayesian import BayesianEstimationResult, estimate_dsge_bayesian
from .sw07_estimate import estimate_sw07
from .fertility_adj_costs import solve_bgp, solve_fertility
from .pruning import (
    PrunedDSGESolution,
    PrunedSimulationResult,
    canonical_growth_2nd_order,
)
from .macro import DynareMacroError, Scope, preprocess_macro
from .widgets import InteractiveIRFResult, interactive_irf
from .dynare import (
    DynareFeatureError,
    build_dynare, parse_mod, load_mod, load_dynare_mod, solve_dynare_2nd_order,
    _orig_stoch_simul, _linear_model_stoch_simul,
)
import functools as _functools
LinearModel.stoch_simul = _functools.wraps(_orig_stoch_simul)(_linear_model_stoch_simul)
from .perfect_foresight import PerfectForesightResult, solve_perfect_foresight
from .extended_path import extended_path
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
    WeibullPrior,
)
from .decomposition import (
    FEVDResult,
    ShockDecompResult,
    compute_fevd,
    compute_shock_decomposition,
)
from ._estimated_params import (
    EstimatedParams,
    EstimatedParamSpec,
    parse_estimated_params,
    parse_estimated_params_bounds,
    parse_estimated_params_init,
)
from .observation import make_state_space_from_varobs
from .smoother import forecast_model, smooth_model
from .mode import cmaes, csminwel, find_mode, mode_check
from .marginal import (
    HarmonicMeanResult,
    harmonic_mean_mdd,
    laplace_mdd,
    model_comparison,
)
from . import priors, fertility_adj_costs
from . import marginal, mode, observation, smoother

__all__ = [
    "klein_solve", "KleinSolution", "BlanchardKahnError",
    "build", "LinearModel", "ModelError", "SteadyStateError",
    "steady", "StructuralSingularityError",
    "DSGEPosteriorResult", "SW07PosteriorResult", "NUTSResult", "BayesianEstimationResult", "FertilitySolution",
    "ScoreDiagnosticsResult", "nuts_sample",
    "DynareDR", "Dynare2ndDR", "TheoreticalMomentsResult", "StochSimulResult",
    "build_dynare", "parse_mod", "load_mod", "load_dynare_mod", "solve_dynare_2nd_order",
    "DynareFeatureError",
    "PerfectForesightResult", "solve_perfect_foresight",
    "OccBinConstraint", "OccBinResult", "solve_occbin",
    "solve_gertler_karadi", "GertlerKaradiResult", "GK2011_PARAMS", "solve_steady_state", "build_gertler_karadi_model",
    "estimate_dsge", "estimate_dsge_bayesian", "estimate_sw07",
    "solve_bgp", "solve_fertility",
    "PrunedDSGESolution", "PrunedSimulationResult", "canonical_growth_2nd_order",
    "FEVDResult", "ShockDecompResult", "compute_fevd", "compute_shock_decomposition",
    "Prior", "BetaPrior", "InvGammaPrior", "NormalPrior", "GammaPrior", "UniformPrior",
    "WeibullPrior",
    # --- 2.6.0: .mod file to posterior -----------------------------------
    "EstimatedParams", "EstimatedParamSpec", "parse_estimated_params",
    "parse_estimated_params_init", "parse_estimated_params_bounds",
    "make_state_space_from_varobs",
    "SmootherResult", "DSGEForecastResult", "smooth_model", "forecast_model",
    "find_mode", "mode_check", "csminwel", "cmaes", "ModeCheckResult",
    "laplace_mdd", "harmonic_mean_mdd", "model_comparison", "HarmonicMeanResult",
    "marginal", "mode", "observation", "smoother",
    "priors", "fertility_adj_costs",
    # --- 2.6.0: DSGE Diagnostics & Residuals ------------------------------
    "check", "resid", "model_diagnostics",
    "EigenvalueTable", "ModelDiagnosticsResult", "DiagnosticFinding",
    # --- 2.6.0: Parameter Identification Analysis ------------------------
    "identification", "IdentificationResult",
    # --- 2.6.0: Optimal Simple Rules & Policy Regimes --------------------
    "osr", "discretionary_policy", "lq_commitment",
    "OSRResult", "PolicyResult",
    # --- 2.9.0: stoch_simul Filtering & Simulation Surface ---------------
    "one_sided_hp_filter",
    # --- 2.9.0: Extended Path (Fair-Taylor 1983) -------------------------
    "extended_path", "ExtendedPathResult",
    # --- 2.9.0: Tier 3 Surface -------------------------------------------
    "conditional_forecast", "ConditionalForecastResult",
    "shock_groups_decomposition", "ShockDecompositionResult",
    "bayesian_irf", "BayesianIRFResult",
    "prior_predictive", "PriorPredictiveResult",
    # --- 2.9.0: Dynare Parity Dashboard & CLI ----------------------------
    "verify_dynare_parity", "compare_model_to_dynare", "run_parity_suite",
    "load_dynare_dr", "load_dynare_moments", "load_irfs", "load_fevd",
    "ParityDashboardResult", "ModelParityResult",
    "load_dynare", "parity",
    # --- 3.0.0: Sequence-Space HANK Bridge -------------------------------
    "HANKModel", "HANKResult", "load_hank_mod", "solve_hank_bridge",
    "hank",
    # --- Dynare Macro Processor & Interactive Widgets --------------------
    "preprocess_macro", "DynareMacroError", "Scope",
    "interactive_irf", "InteractiveIRFResult",
    "macro", "widgets",
]
from . import smets_wouters  # re-export for back-compat with 0.50.0 callers
from . import gertler_karadi
from . import load_dynare, parity
from . import hank
from . import macro, widgets



