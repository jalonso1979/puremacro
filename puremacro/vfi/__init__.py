"""puremacro.vfi -- dynamic programming and continuous projection engine.

A reusable, multi-backend (numpy/numba/mlx/cupy) solver for dynamic economic models:
- Infinite-horizon discrete-choice dynamic programs of VFIToolkit's "Case 1" form:
  define an endogenous-state grid, an exogenous Markov state, an optional decision,
  and a return function, then call .solve().
- Continuous state-space polynomial collocation (orthogonal Chebyshev polynomials
  with continuous Bellman collocation and Euler equation residual projection).
- Finite element method (FEM / Galerkin projection with piecewise linear hat shape
  functions, local curvature, and borrowing constraint handling).
- Continuous stationary distribution & general equilibrium (Young 2010 non-stochastic
  simulation, sparse Markov transition operators, invariant distribution solving,
  and continuous Aiyagari market-clearing equilibrium).
- Shape-preserving cubic B-splines and Schumaker (1983) quadratic splines with
  monotonicity and concavity preservation.
- Smolyak sparse grid collocation for multi-dimensional continuous state spaces
  with nested Clenshaw-Curtis nodes and Chebyshev basis polynomials.
- Discrete Choice Endogenous Grid Method (DC-EGM) and Upper Envelope filtering for
  dynamic models with continuous consumption/savings and discrete choices.
- Continuous-state transition dynamics and unexpected MIT shocks (backward EGM
  and forward Young 2010 distribution push with Broyden and shooting solvers).
- Exact analytic sensitivity and parameter Jacobians via the Implicit Function
  Theorem (IFT) on continuous collocation, FEM, and spline residual systems.
- Deep Macro Physics-Informed Neural Networks (PINNs) in pure NumPy for high-
  dimensional dynamic models (10+ continuous states).
"""
from __future__ import annotations

from puremacro.vfi.discretize import (
    combine_markov_chains,
    farmer_toda,
    markov_stationary,
    rouwenhorst,
    tauchen,
)
from puremacro.vfi.problem import VFIProblem, VFISolution
from puremacro.vfi.distribution import (
    joint_transition_matrix,
    lottery_distribution,
    lottery_push,
    push_distribution,
    stationary_distribution,
)
from puremacro.vfi.transition import TransitionPath, transition_path
from puremacro.vfi.aggregate import aggregate, evaluate_on_grid, lorenz_and_gini, weighted_quantile
from puremacro.vfi.equilibrium import (
    EquilibriumResult,
    PermanentTypesEquilibrium,
    stationary_equilibrium,
    stationary_equilibrium_types,
)
from puremacro.vfi.finite_horizon import (
    FiniteHorizonProblem,
    FiniteHorizonSolution,
    age_profile,
    cross_section,
    life_cycle_distribution,
)
from puremacro.vfi.examples import (
    aiyagari_steady_state,
    hopenhayn_equilibrium,
    huggett_steady_state,
    life_cycle_profile,
    neoclassical_growth,
    two_asset_profile,
)
from puremacro.vfi.olg import (
    OLGEquilibrium,
    olg_aggregate,
    olg_stationary_equilibrium,
    stationary_age_weights,
)
from puremacro.vfi.simulate import empirical_distribution, simulate_panel
from puremacro.vfi.case2 import Case2Problem, Case2Solution
from puremacro.vfi.epstein_zin import EpsteinZinProblem, EpsteinZinSolution
from puremacro.vfi.compat import value_fn_iter_case1
from puremacro.vfi.estimate import EstimationResult, estimate_method_of_moments
from puremacro.vfi.egm import EGMSolution, solve_egm
from puremacro.vfi.permanent_types import PermanentTypesSolution, solve_permanent_types
from puremacro.vfi.firm_dynamics import (
    FirmEntryExitEquilibrium,
    firm_stationary_distribution,
    firm_value_with_exit,
    free_entry_price,
)
from puremacro.vfi.model import Model
from puremacro.vfi.krusell_smith import (
    KSEquilibrium,
    krusell_smith,
    ks_exog_transition,
    ks_simulate,
)
from puremacro.vfi.collocation import (
    CollocationBasis,
    CollocationProblem,
    CollocationSolution,
    solve_collocation,
)
from puremacro.vfi.fem import (
    FEMMesh,
    FEMProblem,
    FEMSolution,
    solve_fem,
)
from puremacro.vfi.continuous_distribution import (
    AiyagariContinuousEquilibrium,
    AiyagariContinuousModel,
    ContinuousDistributionResult,
    ContinuousEquilibriumResult,
    ContinuousStationaryDistribution,
    build_continuous_transition_matrix,
    continuous_push_distribution,
    continuous_stationary_distribution,
    continuous_stationary_equilibrium,
    solve_aiyagari_continuous,
    young_lottery_weights,
    young_stationary_distribution,
    young_step,
    young_transition_matrix,
)
from puremacro.vfi.splines import (
    CubicBSplineBasis,
    SchumakerSpline,
    SplineBasis,
    SplineCollocationProblem,
    SplineCollocationSolution,
    solve_spline_collocation,
)
from puremacro.vfi.smolyak import (
    SmolyakBasis,
    SmolyakGrid,
    SmolyakProblem,
    SmolyakSolution,
    solve_smolyak,
)
from puremacro.vfi.dcegm import (
    ChoiceMapping,
    DCEGMProblem,
    DCEGMSolution,
    UpperEnvelopeResult,
    solve_dcegm,
    upper_envelope,
)
from puremacro.vfi.continuous_transition import (
    ContinuousTransitionResult,
    TransitionShock,
    continuous_mit_shock,
    solve_continuous_transition,
)
from puremacro.vfi.analytic_gradients import (
    AnalyticGradientResult,
    compute_ift_gradients,
    equilibrium_parameter_jacobian,
    gmm_objective_and_gradient,
    policy_parameter_jacobian,
)
from puremacro.vfi.deep_macro import (
    AdamOptimizer,
    DeepMacroMLP,
    DeepMacroModel,
    DeepMacroSolution,
    solve_deep_macro,
)
from puremacro.vfi.hjb_achdou import (
    AiyagariContinuousHJBResult,
    HJBSolution,
    solve_aiyagari_continuous_hjb,
    solve_hjb_achdou,
    solve_kfe_achdou,
)

__all__ = [
    "VFIProblem", "VFISolution", "tauchen", "rouwenhorst", "farmer_toda",
    "combine_markov_chains", "markov_stationary",
    "push_distribution", "stationary_distribution", "joint_transition_matrix",
    "lottery_push", "lottery_distribution",
    "aggregate", "evaluate_on_grid", "lorenz_and_gini", "weighted_quantile",
    "stationary_equilibrium", "EquilibriumResult",
    "stationary_equilibrium_types", "PermanentTypesEquilibrium",
    "FiniteHorizonProblem", "FiniteHorizonSolution",
    "life_cycle_distribution", "cross_section", "age_profile",
    "aiyagari_steady_state", "huggett_steady_state", "life_cycle_profile",
    "two_asset_profile", "neoclassical_growth", "hopenhayn_equilibrium",
    "transition_path", "TransitionPath",
    "simulate_panel", "empirical_distribution",
    "Case2Problem", "Case2Solution",
    "value_fn_iter_case1",
    "estimate_method_of_moments", "EstimationResult",
    "solve_egm", "EGMSolution",
    "solve_permanent_types", "PermanentTypesSolution",
    "stationary_age_weights", "olg_aggregate", "olg_stationary_equilibrium", "OLGEquilibrium",
    "EpsteinZinProblem", "EpsteinZinSolution",
    "firm_value_with_exit", "firm_stationary_distribution",
    "free_entry_price", "FirmEntryExitEquilibrium",
    "ks_exog_transition", "ks_simulate", "krusell_smith", "KSEquilibrium",
    "Model",
    "CollocationBasis", "CollocationProblem", "CollocationSolution", "solve_collocation",
    "FEMMesh", "FEMProblem", "FEMSolution", "solve_fem",
    "ContinuousStationaryDistribution", "ContinuousDistributionResult",
    "AiyagariContinuousEquilibrium", "ContinuousEquilibriumResult",
    "AiyagariContinuousModel", "solve_aiyagari_continuous",
    "continuous_push_distribution", "young_step",
    "continuous_stationary_distribution", "young_stationary_distribution",
    "continuous_stationary_equilibrium", "young_lottery_weights",
    "build_continuous_transition_matrix", "young_transition_matrix",
    "CubicBSplineBasis", "SplineBasis", "SchumakerSpline",
    "SplineCollocationProblem", "SplineCollocationSolution", "solve_spline_collocation",
    "SmolyakGrid", "SmolyakBasis", "SmolyakProblem", "SmolyakSolution", "solve_smolyak",
    "upper_envelope", "UpperEnvelopeResult", "ChoiceMapping",
    "DCEGMProblem", "DCEGMSolution", "solve_dcegm",
    "ContinuousTransitionResult", "solve_continuous_transition", "continuous_mit_shock", "TransitionShock",
    "AnalyticGradientResult", "compute_ift_gradients", "policy_parameter_jacobian", "equilibrium_parameter_jacobian", "gmm_objective_and_gradient",
    "DeepMacroModel", "DeepMacroMLP", "DeepMacroSolution", "solve_deep_macro", "AdamOptimizer",
    "HJBSolution", "solve_hjb_achdou", "solve_kfe_achdou", "solve_aiyagari_continuous_hjb", "AiyagariContinuousHJBResult",
]

