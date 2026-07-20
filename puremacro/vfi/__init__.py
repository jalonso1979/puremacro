"""puremacro.vfi -- general discrete value-function-iteration engine.

A reusable, multi-backend (numpy/numba/mlx/cupy) solver for infinite-horizon
discrete-choice dynamic programs of VFIToolkit's "Case 1" form: define an
endogenous-state grid, an exogenous Markov state, an optional decision, and a
return function, then call .solve().
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
from puremacro.vfi.krusell_smith import (
    KSEquilibrium,
    krusell_smith,
    ks_exog_transition,
    ks_simulate,
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
]
