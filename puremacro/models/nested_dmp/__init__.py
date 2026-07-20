"""Nested heterogeneous-firm DMP model with regime-dependent beliefs.

Phase 1 surface: the parameter object, the pluggable compute backend
(NumPy oracle + Numba + MLX), and the foundational kernels (matching,
Bayesian beliefs, AR(1) discretization, Bellman VFI). The full
equilibrium solve is built in later phases.
"""
from __future__ import annotations

from puremacro.models.nested_dmp.backend import (
    available_backends,
    backend_available,
    get_array_namespace,
    to_numpy,
)
from puremacro.models.nested_dmp.kernels import (
    belief_update,
    bellman_iterate,
    expected_discount,
    f_theta,
    matching_rate,
    q_theta,
    rouwenhorst,
    # tauchen_transition added in phase 2a task 7 (used by dynamics + exposed publicly)
    tauchen_transition,
)
from puremacro.models.nested_dmp.params import NestedDMPParameters
# dynamics surface (phase 2a task 7): perfect-foresight IRF + mu normalizer
from puremacro.models.nested_dmp.dynamics import (
    belief_path,
    calibrate_mu,
    simulate_irf,
    IRFResult,
)
from puremacro.models.nested_dmp.equilibrium import (
    ComparativeStatics,
    SteadyState,
    comparative_statics,
    effective_discount,
    free_entry_theta,
    match_value,
    nash_wage,
    participation_rate,
    sigma_scaled_process,
    solve_steady_state,
    stationary_distribution,
    worker_values,
)
# IRF-matching estimation surface (phase 2a task 7 final): empirical targets,
# calibration, model vs. data fit diagnostics, and belief-ablation counterfactual.
from puremacro.models.nested_dmp.estimation import (
    EmpiricalIRFTargets,
    load_empirical_irf_targets,
    calibrate_mu_to_f,
    model_regime_irfs,
    FitReport,
    fit_report,
    EstimationResult,
    estimate_shape_match,
    ablate_beliefs,
)
# welfare surface (phase 2a phase-5): recursive stochastic solve, ergodic welfare
# simulation, and the optimal lean-against-uncertainty (phi_sigma) policy sweep.
from puremacro.models.nested_dmp.welfare import (
    sigma_process,
    regime_transition,
    belief_transition,
    RecursiveSolution,
    solve_recursive,
    WelfareResult,
    ergodic_welfare,
    WelfareReport,
    optimal_phi_sigma,
)

__all__ = [
    "NestedDMPParameters",
    # backend
    "available_backends",
    "backend_available",
    "get_array_namespace",
    "to_numpy",
    # kernels
    "matching_rate",
    "q_theta",
    "f_theta",
    "belief_update",
    "bellman_iterate",
    "rouwenhorst",
    # kernels (phase 2a addition)
    "expected_discount",
    "tauchen_transition",
    # steady-state equilibrium (phase 2a)
    "SteadyState",
    "effective_discount",
    "sigma_scaled_process",
    "nash_wage",
    "match_value",
    "free_entry_theta",
    "stationary_distribution",
    "solve_steady_state",
    # participation surface (phase 2a task 7)
    "worker_values",
    "participation_rate",
    # comparative statics surface (phase 2a)
    "ComparativeStatics",
    "comparative_statics",
    # dynamics surface (phase 2a task 7)
    "belief_path",
    "calibrate_mu",
    "simulate_irf",
    "IRFResult",
    # estimation surface (phase 2a task 7 final)
    "EmpiricalIRFTargets",
    "load_empirical_irf_targets",
    "calibrate_mu_to_f",
    "model_regime_irfs",
    "FitReport",
    "fit_report",
    "EstimationResult",
    "estimate_shape_match",
    "ablate_beliefs",
    # welfare surface (phase 2a phase-5)
    "sigma_process",
    "regime_transition",
    "belief_transition",
    "RecursiveSolution",
    "solve_recursive",
    "WelfareResult",
    "ergodic_welfare",
    "WelfareReport",
    "optimal_phi_sigma",
]
