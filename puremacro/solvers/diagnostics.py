"""Solver capability, convergence, accuracy, and support-limit diagnostic records.

Implements Item 5 of REPORT.md:
"Users can distinguish a stable interface, solver convergence and economic accuracy...
A generic converged=True is not an audit PASS... examples demonstrate a converged but
insufficiently accurate result and an unsupported operation; neither receives an unqualified
success badge."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ConvergenceStatus:
    """Numerical solver convergence indicators.

    Attributes
    ----------
    converged : bool
        Whether the numerical solver reached its stopping criterion.
    iterations : int
        Number of solver iterations or evaluations executed.
    stopping_reason : str
        Reason or flag reported by the algorithm termination (e.g. 'tolerance_met',
        'max_iterations', 'step_failure').
    solver_method : str
        Algorithm or optimizer method used (e.g. 'newton', 'closed_form_ols', 'value_function_iteration').
    tolerance : float
        Numerical convergence tolerance criterion used during the solve.
    """

    converged: bool
    iterations: int = 0
    stopping_reason: str = ""
    solver_method: str = ""
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.converged, bool):
            object.__setattr__(self, "converged", bool(self.converged))
        if not isinstance(self.iterations, int):
            object.__setattr__(self, "iterations", int(self.iterations))
        if not isinstance(self.tolerance, float):
            object.__setattr__(self, "tolerance", float(self.tolerance))


@dataclass(frozen=True)
class AccuracyStatus:
    """Economic equation accuracy and residual evaluation.

    Attributes
    ----------
    is_accurate : bool
        Whether economic residuals (e.g. Euler KKT violations, market clearing errors,
        Bellman residuals) satisfy the target accuracy threshold.
    max_residual : float
        Maximum absolute residual observed across all equations or states.
    residual_norm : float
        L1, L2, or Linf norm of equation residuals.
    residual_type : str
        Type of residual evaluated (e.g. 'euler_kkt', 'market_clearing', 'bellman', 'normal_equations').
    tolerance : float
        Accuracy tolerance threshold required for economic certification.
    residuals : dict[str, float]
        Dictionary of named individual residual components.
    """

    is_accurate: bool
    max_residual: float = 0.0
    residual_norm: float = 0.0
    residual_type: str = ""
    tolerance: float = 0.0
    residuals: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.is_accurate, bool):
            object.__setattr__(self, "is_accurate", bool(self.is_accurate))
        if not isinstance(self.max_residual, float):
            object.__setattr__(self, "max_residual", float(self.max_residual))
        if not isinstance(self.residual_norm, float):
            object.__setattr__(self, "residual_norm", float(self.residual_norm))
        if not isinstance(self.tolerance, float):
            object.__setattr__(self, "tolerance", float(self.tolerance))
        if self.residuals is None:
            object.__setattr__(self, "residuals", {})
        elif not isinstance(self.residuals, dict):
            object.__setattr__(self, "residuals", dict(self.residuals))


@dataclass(frozen=True)
class SupportStatus:
    """Operational support limits and boundary regime indicators.

    Attributes
    ----------
    is_supported : bool
        Whether the problem state resides strictly within the mathematically and
        economically supported domain (e.g. non-explosive roots, non-saturated debt grid,
        positive prices/endowments).
    regime : str
        Identified regime (e.g. 'interior', 'stationary', 'boundary_debt_saturation',
        'non_stationary_explosive', 'infeasible_prices').
    warnings : tuple[str, ...]
        Specific warnings or caveats regarding domain boundary limits.
    boundary_metrics : dict[str, Any]
        Metrics quantifying boundary proximity (e.g. spectral radius, boundary mass,
        minimum prices).
    """

    is_supported: bool
    regime: str = "interior"
    warnings: tuple[str, ...] = field(default_factory=tuple)
    boundary_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.is_supported, bool):
            object.__setattr__(self, "is_supported", bool(self.is_supported))
        if not isinstance(self.warnings, tuple):
            object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.boundary_metrics is None:
            object.__setattr__(self, "boundary_metrics", {})
        elif not isinstance(self.boundary_metrics, dict):
            object.__setattr__(self, "boundary_metrics", dict(self.boundary_metrics))


VALID_OVERALL_STATUSES = ("PASS", "CONVERGED_INACCURATE", "UNSUPPORTED_REGIME", "FAILED")


def determine_overall_status(
    convergence: ConvergenceStatus,
    accuracy: AccuracyStatus,
    support: SupportStatus,
) -> str:
    """Determine overall status distinguishing convergence from inaccuracy and support limits.

    Hierarchy:
    1. If solver failed to converge -> 'FAILED'
    2. Else if problem is in an unsupported regime -> 'UNSUPPORTED_REGIME'
    3. Else if solution is insufficiently accurate -> 'CONVERGED_INACCURATE'
    4. Else -> 'PASS'
    """
    if not convergence.converged:
        return "FAILED"
    if not support.is_supported:
        return "UNSUPPORTED_REGIME"
    if not accuracy.is_accurate:
        return "CONVERGED_INACCURATE"
    return "PASS"


@dataclass(frozen=True)
class SolverDiagnosticRecord:
    """Standardized multi-dimensional solver audit status record.

    Exposes convergence, economic accuracy, and support limits as separate dimensions,
    preventing inaccurate or out-of-support operations from receiving an unqualified
    success badge.

    Attributes
    ----------
    convergence : ConvergenceStatus
        Convergence indicators.
    accuracy : AccuracyStatus
        Economic accuracy and residual metrics.
    support : SupportStatus
        Support boundary and regime validation.
    overall_status : str
        Must be one of 'PASS', 'CONVERGED_INACCURATE', 'UNSUPPORTED_REGIME', or 'FAILED'.
    evidence : dict[str, Any]
        Additional provenance, timings, or diagnostic evidence.
    """

    convergence: ConvergenceStatus
    accuracy: AccuracyStatus
    support: SupportStatus
    overall_status: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.overall_status not in VALID_OVERALL_STATUSES:
            raise ValueError(
                f"Invalid overall_status {self.overall_status!r}; expected one of {VALID_OVERALL_STATUSES}"
            )
        if self.evidence is None:
            object.__setattr__(self, "evidence", {})
        elif not isinstance(self.evidence, dict):
            object.__setattr__(self, "evidence", dict(self.evidence))

    @property
    def is_pass(self) -> bool:
        """True only if overall_status is 'PASS'."""
        return self.overall_status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostic record to a plain JSON-serializable dictionary."""
        return {
            "overall_status": self.overall_status,
            "convergence": {
                "converged": self.convergence.converged,
                "iterations": self.convergence.iterations,
                "stopping_reason": self.convergence.stopping_reason,
                "solver_method": self.convergence.solver_method,
                "tolerance": self.convergence.tolerance,
            },
            "accuracy": {
                "is_accurate": self.accuracy.is_accurate,
                "max_residual": self.accuracy.max_residual,
                "residual_norm": self.accuracy.residual_norm,
                "residual_type": self.accuracy.residual_type,
                "tolerance": self.accuracy.tolerance,
                "residuals": dict(self.accuracy.residuals),
            },
            "support": {
                "is_supported": self.support.is_supported,
                "regime": self.support.regime,
                "warnings": list(self.support.warnings),
                "boundary_metrics": dict(self.support.boundary_metrics),
            },
            "evidence": dict(self.evidence),
        }

    def summary(self) -> str:
        """One-line diagnostic summary."""
        parts = [
            f"status={self.overall_status}",
            f"converged={self.convergence.converged} ({self.convergence.iterations} iters)",
            f"accurate={self.accuracy.is_accurate} (max_res={self.accuracy.max_residual:.3e})",
            f"supported={self.support.is_supported} (regime={self.support.regime})",
        ]
        if self.support.warnings:
            parts.append(f"warnings={len(self.support.warnings)}")
        return "SolverDiagnosticRecord(" + ", ".join(parts) + ")"


def make_solver_diagnostic_record(
    *,
    converged: bool,
    iterations: int = 0,
    stopping_reason: str = "",
    solver_method: str = "",
    tolerance: float = 0.0,
    is_accurate: bool = True,
    max_residual: float = 0.0,
    residual_norm: float = 0.0,
    residual_type: str = "",
    accuracy_tolerance: float = 0.0,
    residuals: dict[str, float] | None = None,
    is_supported: bool = True,
    regime: str = "interior",
    warnings: Sequence[str] = (),
    boundary_metrics: dict[str, Any] | None = None,
    overall_status: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> SolverDiagnosticRecord:
    """Build a SolverDiagnosticRecord cleanly with automatic overall status determination."""
    conv = ConvergenceStatus(
        converged=converged,
        iterations=iterations,
        stopping_reason=stopping_reason,
        solver_method=solver_method,
        tolerance=tolerance,
    )
    acc = AccuracyStatus(
        is_accurate=is_accurate,
        max_residual=max_residual,
        residual_norm=residual_norm,
        residual_type=residual_type,
        tolerance=accuracy_tolerance,
        residuals=residuals or {},
    )
    supp = SupportStatus(
        is_supported=is_supported,
        regime=regime,
        warnings=tuple(warnings),
        boundary_metrics=boundary_metrics or {},
    )
    if overall_status is None:
        overall_status = determine_overall_status(conv, acc, supp)

    return SolverDiagnosticRecord(
        convergence=conv,
        accuracy=acc,
        support=supp,
        overall_status=overall_status,
        evidence=evidence or {},
    )


__all__ = [
    "ConvergenceStatus",
    "AccuracyStatus",
    "SupportStatus",
    "SolverDiagnosticRecord",
    "VALID_OVERALL_STATUSES",
    "determine_overall_status",
    "make_solver_diagnostic_record",
]
