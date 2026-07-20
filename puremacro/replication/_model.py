"""Core model for puremacro's replication gallery.

Pyodide-pure: numpy + stdlib only. A case reproduces a *published* result:
its ``estimate()`` runs a puremacro estimator (or model solve) and returns
metrics, which are compared to the paper's reported ``target`` at a
``target_kind``. Replication is coarse by nature (vintages/samples differ),
so the kinds encode "what counts as reproducing it".
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


class TargetKind(enum.Enum):
    POINT = "point"              # within an rtol of the published number
    SIGN = "sign"                # estimate has the published sign
    CORRELATION = "correlation"  # corr(reconstructed, published) >= threshold
    # CI_OVERLAP and SHAPE are added by the monetary-LP / fiscal plans (IRF targets).


class Tol(enum.Enum):
    TIGHT = "tight"      # POINT rtol 0.02
    MEDIUM = "medium"    # POINT rtol 0.10
    COARSE = "coarse"    # POINT rtol 0.25


_POINT_RTOL = {Tol.TIGHT: 0.02, Tol.MEDIUM: 0.10, Tol.COARSE: 0.25}


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _compare_target(
    metric: Mapping[str, object], target: Mapping[str, object], kind: TargetKind, tol: Tol
) -> tuple[bool, float]:
    """Compare ``metric`` to a published ``target`` under ``kind``.

    Only keys in ``target`` are checked. Returns ``(passed, margin)`` where
    ``margin`` is a slack figure (>0 means a failing key, for SIGN/CORRELATION;
    for POINT it is ``max(|got-target| - allowed)``, <=0 means pass).
    """
    if kind is TargetKind.SIGN:
        ok = True
        worst = 0.0
        for k, t in target.items():
            got = float(np.asarray(metric[k], dtype=float))
            match = _sign(got) == _sign(float(np.asarray(t, dtype=float)))
            ok = ok and match
            worst = max(worst, 0.0 if match else 1.0)
        return ok, worst

    if kind is TargetKind.CORRELATION:
        ok = True
        worst = -math.inf
        for k, thr in target.items():
            got = float(np.asarray(metric[k], dtype=float))
            thr_f = float(np.asarray(thr, dtype=float))
            deficit = thr_f - got
            worst = max(worst, deficit)
            ok = ok and (got >= thr_f - 1e-12)
        return ok, worst

    if kind is TargetKind.POINT:
        rtol = _POINT_RTOL[tol]
        ok = True
        worst = -math.inf
        for k, t in target.items():
            got_arr = np.asarray(metric[k], dtype=float)
            base = np.asarray(t, dtype=float)
            excess = np.abs(got_arr - base) - (rtol * np.abs(base) + 1e-9)
            worst = max(worst, float(np.max(excess)))
            ok = ok and bool(np.all(excess <= 0))
        return ok, worst

    raise ValueError(f"unhandled TargetKind: {kind}")


@dataclass(frozen=True)
class ReplicationCase:
    """One replication: a puremacro estimator/model vs a paper's published result."""

    id: str
    family: str
    paper: str
    title: str
    title_es: str
    source: str  # descriptive: "model" | "bundled" | "snapshot:<file>" | "loader+network"
    estimate: Callable[[], Mapping[str, object]]  # self-contained: loads/solves → metrics
    target: Mapping[str, object]
    target_kind: TargetKind
    tol: Tol
    citation: str  # the exact table/figure the target comes from
    network: bool = False
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    id: str
    family: str
    paper: str
    source: str
    target_kind: str
    tol: str
    passed: bool
    margin: float
    metrics: dict
    citation: str
    network: bool
    error: str = ""


def run(case: ReplicationCase) -> CaseResult:
    """Run a replication case and compare to its published target. Never raises."""
    base: dict[str, Any] = dict(
        id=case.id, family=case.family, paper=case.paper, source=case.source,
        target_kind=case.target_kind.value, tol=case.tol.value,
        citation=case.citation, network=case.network,
    )
    try:
        metric = case.estimate()
        passed, margin = _compare_target(metric, case.target, case.target_kind, case.tol)
        metrics = {k: {"got": metric.get(k), "target": case.target[k]} for k in case.target}
        return CaseResult(passed=passed, margin=margin, metrics=metrics, **base)
    except Exception as exc:  # noqa: BLE001 — a broken replication is a failed case, not a crash
        return CaseResult(
            passed=False, margin=math.inf, metrics={}, error=f"{type(exc).__name__}: {exc}", **base
        )
