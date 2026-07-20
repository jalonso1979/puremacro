"""Core model for puremacro's validation gallery.

Pyodide-pure: imports only numpy + stdlib. Reference packages
(statsmodels/linearmodels/arch) are NEVER imported here — PACKAGE/PUBLISHED
references are read from frozen golden JSON instead.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Union

import numpy as np


class Mechanism(enum.Enum):
    """How a case's reference value is sourced."""

    PACKAGE = "package"        # statsmodels/linearmodels/arch — frozen golden + CI drift-guard
    SCIPY = "scipy"            # scipy/numpy — computed live (runtime deps)
    ANALYTICAL = "analytical"  # closed-form solution
    PUBLISHED = "published"    # numbers from a paper (+ citation), frozen golden
    INTERNAL = "internal"      # cross-method agreement / identities


class Tol(enum.Enum):
    """Comparison strictness (orthogonal to Mechanism)."""

    EXACT = "exact"
    TIGHT = "tight"
    NUMERIC = "numeric"
    COARSE = "coarse"
    QUALITATIVE = "qualitative"  # metric must be >= reference threshold (lower bound)


# (rtol, atol) per numeric tier.
_TOL_PARAMS: dict[Tol, tuple[float, float]] = {
    Tol.EXACT: (1e-10, 1e-12),
    Tol.TIGHT: (1e-6, 1e-9),
    Tol.NUMERIC: (1e-2, 1e-4),
    Tol.COARSE: (1e-1, 0.0),
}


def _compare(
    pm: Mapping[str, object], ref: Mapping[str, object], tol: Tol
) -> tuple[bool, float]:
    """Compare puremacro output ``pm`` to reference ``ref`` at tier ``tol``.

    Only keys present in ``ref`` are checked (``pm`` may carry extras).
    Values may be scalars or array-likes; comparison is elementwise.

    Returns ``(passed, worst_margin)``. For numeric tiers ``worst_margin`` is
    ``max(|pm-ref| - allowed)`` (<= 0 means pass). For QUALITATIVE it is the
    largest shortfall ``max(ref - pm)`` (> 0 means a value fell below threshold).
    """
    if tol is Tol.QUALITATIVE:
        ok = True
        worst = -math.inf
        for k, refv in ref.items():
            a = np.asarray(pm[k], dtype=float)
            b = np.asarray(refv, dtype=float)
            shortfall = float(np.max(b - a))
            worst = max(worst, shortfall)
            ok = ok and bool(np.all(a >= b - 1e-12))
        return ok, worst

    rtol, atol = _TOL_PARAMS[tol]
    ok = True
    worst = -math.inf
    for k, refv in ref.items():
        a = np.asarray(pm[k], dtype=float)
        b = np.asarray(refv, dtype=float)
        allowed = atol + rtol * np.abs(b)
        excess = np.abs(a - b) - allowed
        worst = max(worst, float(np.max(excess)))
        ok = ok and bool(np.all(excess <= 0))
    return ok, worst


@dataclass(frozen=True)
class ValidationCase:
    """One declarative validation: puremacro vs a reference, at a tolerance."""

    id: str
    subsystem: str
    title: str
    title_es: str
    mechanism: Mechanism
    compute: Callable[[], Mapping[str, object]]
    # Callable returning the reference dict, OR a golden key "subsystem:case_id".
    reference: Union[Callable[[], Mapping[str, object]], str]
    tol: Tol
    citation: str
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    id: str
    subsystem: str
    mechanism: str
    tol: str
    passed: bool
    max_margin: float
    metrics: dict
    citation: str
    error: str = ""


def _resolve_reference(case: ValidationCase) -> Mapping[str, object]:
    ref = case.reference
    if isinstance(ref, str):
        from ._goldens import load_golden

        return load_golden(ref)
    return ref()


def run(case: ValidationCase) -> CaseResult:
    """Execute a case and compare to its reference. Never raises."""
    base = dict(
        id=case.id,
        subsystem=case.subsystem,
        mechanism=case.mechanism.value,
        tol=case.tol.value,
        citation=case.citation,
    )
    try:
        pm = case.compute()
        ref = _resolve_reference(case)
        passed, margin = _compare(pm, ref, case.tol)
        metrics = {k: {"pm": pm.get(k), "ref": ref[k]} for k in ref}
        return CaseResult(passed=passed, max_margin=margin, metrics=metrics, **base)
    except Exception as exc:  # noqa: BLE001 — a broken case is a failed case, not a crash
        return CaseResult(
            passed=False,
            max_margin=math.inf,
            metrics={},
            error=f"{type(exc).__name__}: {exc}",
            **base,
        )
