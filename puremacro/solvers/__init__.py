"""Solvers module: diagnostics, capabilities, convergence, and accuracy indicators."""
from __future__ import annotations

from puremacro.solvers.diagnostics import (
    AccuracyStatus,
    ConvergenceStatus,
    SolverDiagnosticRecord,
    SupportStatus,
    VALID_OVERALL_STATUSES,
    determine_overall_status,
    make_solver_diagnostic_record,
)

__all__ = [
    "AccuracyStatus",
    "ConvergenceStatus",
    "SolverDiagnosticRecord",
    "SupportStatus",
    "VALID_OVERALL_STATUSES",
    "determine_overall_status",
    "make_solver_diagnostic_record",
]
