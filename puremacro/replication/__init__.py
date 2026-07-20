"""puremacro replication gallery — reproduce published papers' headline results.

>>> from puremacro.replication import run_all, scorecard
>>> df = scorecard()           # puremacro vs each paper's published target
>>> assert df["passed"].all()

Pyodide-pure: model-based cases solve live; data-backed cases (added by later
families) read vendored snapshots. No statsmodels needed.
"""
from __future__ import annotations

from ._model import CaseResult, ReplicationCase, TargetKind, Tol, run
from .runner import run_all, scorecard

__all__ = [
    "run",
    "run_all",
    "scorecard",
    "ReplicationCase",
    "CaseResult",
    "TargetKind",
    "Tol",
]
