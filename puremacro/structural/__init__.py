"""Minimum-distance bridge from empirical moments to structural models."""

from .bridge import (
    AGSSensitivityResult,
    MomentTargets,
    StructuralFitResult,
    ags_sensitivity,
    fit_structural,
)
from .lp import lp_moment_targets
from .simulation import (
    MomentReplication,
    SimulationMomentError,
    SimulationPlan,
    SimulatedMoments,
    SMMFitResult,
    fit_simulated_moments,
    moment_error_covariance,
    simulate_moments,
)
from .robust import (
    MomentConfidenceSet,
    MomentTestGridResult,
    MomentTestResult,
    QuadraticMomentCurve,
    evaluate_moment_grid,
    invert_quadratic_moments,
    moment_restriction_test,
    simulated_moment_test,
)

__all__ = [
    "AGSSensitivityResult",
    "MomentTargets",
    "StructuralFitResult",
    "ags_sensitivity",
    "fit_structural",
    "lp_moment_targets",
    "MomentReplication",
    "SimulationMomentError",
    "SimulationPlan",
    "SimulatedMoments",
    "SMMFitResult",
    "fit_simulated_moments",
    "moment_error_covariance",
    "simulate_moments",
    "MomentConfidenceSet",
    "MomentTestGridResult",
    "MomentTestResult",
    "QuadraticMomentCurve",
    "evaluate_moment_grid",
    "invert_quadratic_moments",
    "moment_restriction_test",
    "simulated_moment_test",
]
